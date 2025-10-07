"""LLM client abstract base class with retry mechanism."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from src.config.settings import settings


try:
    from prometheus_client import Counter, Histogram
except Exception:  # pragma: no cover - optional dependency during tests
    Counter = None  # type: ignore
    Histogram = None  # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Retry configuration with exponential backoff."""

    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class LLMError(Exception):
    """Base class for LLM-related errors."""

    pass


class RateLimitError(LLMError):
    """Rate limit exceeded error."""

    pass


class AuthenticationError(LLMError):
    """Authentication/authorization error."""

    pass


class NetworkError(LLMError):
    """Network connectivity error."""

    pass


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients with a minimal async interface."""

    def __init__(self, retry_config: RetryConfig | None = None):
        self.retry_config = retry_config or RetryConfig()
        self._request_count = 0
        self._error_count = 0
        self._last_request_time = None

        # Initialize Prometheus metrics lazily if available and enabled
        self._metrics_enabled = bool(settings.enable_metrics and Counter and Histogram)
        if self._metrics_enabled:
            # Define metrics only once per process
            global _LLM_CALLS, _LLM_LATENCY
            if "_LLM_CALLS" not in globals():
                _LLM_CALLS = Counter(
                    "llm_calls_total",
                    "Total LLM calls",
                    labelnames=("provider", "model", "method", "status"),
                )
                _LLM_LATENCY = Histogram(
                    "llm_call_duration_seconds",
                    "LLM call latency in seconds",
                    labelnames=("provider", "model", "method"),
                    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, float("inf")),
                )

    async def complete_with_retry(self, prompt: str, **kwargs) -> str:
        """Complete text generation with retry mechanism."""
        return await self._retry_on_error(self.complete, prompt, **kwargs)

    async def complete_json_with_retry(self, prompt: str, **kwargs) -> dict[str, Any]:
        """Complete JSON generation with retry mechanism."""
        result = await self._retry_on_error(self.complete, prompt, **kwargs)
        try:
            return json.loads(result)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {result}")
            raise LLMError(f"Invalid JSON response: {e!s}") from e

    async def _retry_on_error(self, func: Callable, *args, **kwargs):
        """Generic retry decorator with exponential backoff."""
        last_exception = None

        for attempt in range(self.retry_config.max_attempts):
            try:
                self._request_count += 1
                self._last_request_time = datetime.now()

                if self._metrics_enabled:
                    provider = getattr(self, "_provider_name", "unknown")
                    model = getattr(self, "_model", "unknown")
                    method = getattr(func, "__name__", "unknown")
                    import time

                    start = time.monotonic()
                    try:
                        result = await func(*args, **kwargs)
                        _LLM_CALLS.labels(provider, model, method, "success").inc()
                        return result
                    except Exception:  # re-counted below in error path as well
                        _LLM_CALLS.labels(provider, model, method, "error").inc()
                        raise
                    finally:
                        _LLM_LATENCY.labels(provider, model, method).observe(
                            time.monotonic() - start
                        )
                else:
                    result = await func(*args, **kwargs)

                if attempt > 0:
                    logger.info(f"Request succeeded on attempt {attempt + 1}")

                return result

            except Exception as e:
                last_exception = e
                self._error_count += 1

                error_type = self._classify_error(e)

                if error_type == "authentication":
                    logger.error(f"Authentication error: {e!s}")
                    raise AuthenticationError(str(e)) from e

                if error_type == "non_retryable":
                    logger.error(f"Non-retryable error: {e!s}")
                    raise e
                if attempt < self.retry_config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}): {e!s}. Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Request failed after {self.retry_config.max_attempts} attempts: {e!s}"
                    )

        raise last_exception or LLMError("All retry attempts failed")

    def _classify_error(self, error: Exception) -> str:
        """Classify error type for retry decision."""
        error_msg = str(error).lower()

        if "authentication" in error_msg or "unauthorized" in error_msg or "api key" in error_msg:
            return "authentication"
        elif "rate limit" in error_msg or "too many requests" in error_msg:
            return "rate_limit"
        elif "network" in error_msg or "connection" in error_msg or "timeout" in error_msg:
            return "network"
        else:
            return "retryable"

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        delay = self.retry_config.initial_delay * (self.retry_config.exponential_base**attempt)
        delay = min(delay, self.retry_config.max_delay)

        if self.retry_config.jitter:
            import random

            delay *= 0.5 + random.random() * 0.5

        return delay

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics and metrics."""
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(self._request_count, 1),
            "last_request_time": self._last_request_time,
            "retry_config": {
                "max_attempts": self.retry_config.max_attempts,
                "initial_delay": self.retry_config.initial_delay,
                "max_delay": self.retry_config.max_delay,
            },
        }

    @abstractmethod
    async def complete(self, prompt: str) -> str:
        """Return a completion string for the given prompt."""
        raise NotImplementedError

    @abstractmethod
    async def complete_with_functions(
        self, messages: list[dict[str, str]], functions: list[dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        """Complete with function calling support."""
        raise NotImplementedError

    async def complete_with_context(self, messages: list[dict[str, str]]) -> str:
        """Complete with conversation context (multi-round chat)."""
        # Default implementation: concatenate messages into single prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        combined_prompt = "\n\n".join(prompt_parts)
        return await self.complete(combined_prompt)

    async def complete_chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Complete chat messages and expect a JSON object response."""

        raise NotImplementedError("JSON chat completion not implemented for this client")


class OpenAICompatibleClient(BaseLLMClient):
    """Client for OpenAI-compatible APIs (e.g., DeepSeek via base_url)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        super().__init__(retry_config)
        provider = settings.llm_provider.strip().lower()

        if api_key is not None and base_url is not None:
            # 显式传参优先，直接采用
            self._api_key = api_key
            self._base_url = base_url
        elif provider == "openrouter":
            self._api_key = settings.openrouter_api_key
            self._base_url = settings.openrouter_api_base
        elif provider == "deepseek":
            self._api_key = settings.deepseek_api_key
            self._base_url = settings.llm_base_url
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")

        self._model = model or settings.llm_model
        self._provider_name = provider
        self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if hasattr(self, "_client") and self._client:
            await self._client.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        """Complete text generation using the configured model."""
        try:
            if getattr(self, "_metrics_enabled", False):
                import time

                start = time.monotonic()
                try:
                    resp: ChatCompletion = await self._client.chat.completions.create(
                        model=model or self._model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    _LLM_CALLS.labels(
                        self._provider_name, model or self._model, "complete", "success"
                    ).inc()
                    return resp.choices[0].message.content or ""
                except Exception:
                    _LLM_CALLS.labels(
                        self._provider_name, model or self._model, "complete", "error"
                    ).inc()
                    raise
                finally:
                    _LLM_LATENCY.labels(
                        self._provider_name, model or self._model, "complete"
                    ).observe(time.monotonic() - start)
            else:
                resp: ChatCompletion = await self._client.chat.completions.create(
                    model=model or self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.debug(f"OpenAI API call failed: {e!s}")
            raise e

    async def validate_connection(self) -> bool:
        """Validate if the connection to LLM service is working."""
        try:
            await self.complete("Hello", max_tokens=1)
            return True
        except Exception as e:
            logger.error(f"Connection validation failed: {e!s}")
            return False

    def get_token_count(self, text: str) -> int:
        """Estimate token count for given text."""
        return len(text) // 4

    async def complete_with_context(
        self, messages: list[dict[str, str]], model: str | None = None
    ) -> str:
        """Complete with conversation context using native chat format."""
        try:
            if getattr(self, "_metrics_enabled", False):
                import time

                start = time.monotonic()
                try:
                    resp: ChatCompletion = await self._client.chat.completions.create(
                        model=model or self._model,
                        messages=messages,
                        temperature=0.2,
                    )
                    _LLM_CALLS.labels(
                        self._provider_name,
                        model or self._model,
                        "complete_with_context",
                        "success",
                    ).inc()
                    return resp.choices[0].message.content or ""
                except Exception:
                    _LLM_CALLS.labels(
                        self._provider_name, model or self._model, "complete_with_context", "error"
                    ).inc()
                    raise
                finally:
                    _LLM_LATENCY.labels(
                        self._provider_name, model or self._model, "complete_with_context"
                    ).observe(time.monotonic() - start)
            else:
                resp: ChatCompletion = await self._client.chat.completions.create(
                    model=model or self._model,
                    messages=messages,
                    temperature=0.2,
                )
                return resp.choices[0].message.content or ""
        except Exception as e:
            logger.debug(f"OpenAI API call with context failed: {e!s}")
            raise e

    async def complete_chat_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """Complete chat messages and expect a JSON object response."""

        try:
            if getattr(self, "_metrics_enabled", False):
                import time

                start = time.monotonic()
                try:
                    resp: ChatCompletion = await self._client.chat.completions.create(
                        model=model or self._model,
                        messages=messages,
                        temperature=temperature,
                        response_format={"type": "json_object"},
                    )
                    _LLM_CALLS.labels(
                        self._provider_name, model or self._model, "complete_chat_json", "success"
                    ).inc()
                    content = resp.choices[0].message.content or "{}"
                    return json.loads(content)
                except Exception:
                    _LLM_CALLS.labels(
                        self._provider_name, model or self._model, "complete_chat_json", "error"
                    ).inc()
                    raise
                finally:
                    _LLM_LATENCY.labels(
                        self._provider_name, model or self._model, "complete_chat_json"
                    ).observe(time.monotonic() - start)
            else:
                resp: ChatCompletion = await self._client.chat.completions.create(
                    model=model or self._model,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or "{}"
                return json.loads(content)
        except Exception as e:
            logger.debug(f"OpenAI API json completion failed: {e!s}")
            raise e

    async def complete_with_functions(
        self,
        messages: list[dict[str, str]],
        functions: list[dict[str, Any]],
        model: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Complete with function calling support using DeepSeek API."""
        try:
            temperature = kwargs.pop("temperature", 0.1)
            if getattr(self, "_metrics_enabled", False):
                import time

                start = time.monotonic()
                try:
                    resp: ChatCompletion = await self._client.chat.completions.create(
                        model=model or self._model,
                        messages=messages,
                        tools=functions,
                        tool_choice="auto",
                        temperature=temperature,
                        **kwargs,
                    )
                except Exception:
                    _LLM_CALLS.labels(
                        self._provider_name,
                        model or self._model,
                        "complete_with_functions",
                        "error",
                    ).inc()
                    raise
                finally:
                    _LLM_LATENCY.labels(
                        self._provider_name, model or self._model, "complete_with_functions"
                    ).observe(time.monotonic() - start)
                _LLM_CALLS.labels(
                    self._provider_name, model or self._model, "complete_with_functions", "success"
                ).inc()
            else:
                resp: ChatCompletion = await self._client.chat.completions.create(
                    model=model or self._model,
                    messages=messages,
                    tools=functions,
                    tool_choice="auto",
                    temperature=temperature,
                    **kwargs,
                )

            choice = resp.choices[0]
            message = choice.message

            # 检查是否有函数调用
            if message.tool_calls:
                tool_call = message.tool_calls[0]  # 取第一个函数调用
                return {
                    "function_call": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                    "content": message.content,
                    "finish_reason": choice.finish_reason,
                }
            else:
                # 没有函数调用，返回普通文本响应
                return {"content": message.content or "", "finish_reason": choice.finish_reason}

        except Exception as e:
            logger.debug(f"OpenAI API call with functions failed: {e!s}")
            raise e
