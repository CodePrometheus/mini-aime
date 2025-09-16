import os
from typing import Any, Dict, List, Optional

import pytest

from src.core.actor_factory import ActorFactory
from src.llm.base import BaseLLMClient


class DummyLLMClient(BaseLLMClient):
    async def complete(self, prompt: str, **kwargs) -> str:  # type: ignore[override]
        return "{}"

    async def complete_with_functions(
        self,
        messages: List[Dict[str, Any]],
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:  # type: ignore[override]
        return {"choices": []}


def _ensure_env():
    # Tavily 仅在初始化时需要，传入任意字符串即可避免校验失败
    os.environ.setdefault("TAVILY_API_KEY", "dummy_tavily_key")


@pytest.mark.skipif(
    not os.getenv("EXCHANGE_RATE_API_KEY"),
    reason="EXCHANGE_RATE_API_KEY not set",
)
def test_currency_convert_tool_live():
    _ensure_env()
    factory = ActorFactory(llm_client=DummyLLMClient(), initialize_tools=True)
    travel_tools = factory.tool_bundles["travel_services"]["tools"]
    tool = next(t for t in travel_tools if t.name == "currency_convert")

    result = tool.func(from_currency="USD", to_currency="JPY", amount=100)  # type: ignore[attr-defined]
    print("result:", result)

    assert "USD" in result and "JPY" in result
    assert "->" in result


@pytest.mark.skipif(
    not os.getenv("TIMEZONEDB_API_KEY"),
    reason="TIMEZONEDB_API_KEY not set",
)
def test_get_timezone_tool_live():
    _ensure_env()
    factory = ActorFactory(llm_client=DummyLLMClient(), initialize_tools=True)
    travel_tools = factory.tool_bundles["travel_services"]["tools"]
    tool = next(t for t in travel_tools if t.name == "get_timezone")

    result = tool.func(zone="Asia/Tokyo")  # type: ignore[attr-defined]

    assert "时区:" in result
    assert "Asia/Tokyo" in result or "Tokyo" in result


def test_get_public_holidays_tool_live():
    _ensure_env()
    factory = ActorFactory(llm_client=DummyLLMClient(), initialize_tools=True)
    travel_tools = factory.tool_bundles["travel_services"]["tools"]
    tool = next(t for t in travel_tools if t.name == "get_public_holidays")

    result = tool.func(country_code="JP", year=2025)  # type: ignore[attr-defined]

    assert "公共假期" in result
    assert "2025-01-01" in result  # New Year's Day is expected


