"""Tests for LLM client functionality."""

from unittest.mock import AsyncMock, Mock

import pytest

from src.llm.base import BaseLLMClient, OpenAICompatibleClient, RetryConfig


class TestBaseLLMClient:
    """Test base LLM client functionality."""

    def test_retry_config_defaults(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_error_classification(self):
        """Test error type classification."""
        client = Mock(spec=BaseLLMClient)
        client._classify_error = BaseLLMClient._classify_error.__get__(client)

        # Authentication errors
        assert client._classify_error(Exception("authentication failed")) == "authentication"
        assert client._classify_error(Exception("unauthorized access")) == "authentication"
        assert client._classify_error(Exception("invalid api key")) == "authentication"

        # Rate limit errors
        assert client._classify_error(Exception("rate limit exceeded")) == "rate_limit"
        assert client._classify_error(Exception("too many requests")) == "rate_limit"

        # Network errors
        assert client._classify_error(Exception("network timeout")) == "network"
        assert client._classify_error(Exception("connection failed")) == "network"

        # Generic retryable errors
        assert client._classify_error(Exception("server error")) == "retryable"

    def test_delay_calculation(self):
        """Test retry delay calculation."""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, jitter=False)
        client = Mock(spec=BaseLLMClient)
        client.retry_config = config
        client._calculate_delay = BaseLLMClient._calculate_delay.__get__(client)

        # Test exponential backoff without jitter
        assert client._calculate_delay(0) == 1.0
        assert client._calculate_delay(1) == 2.0
        assert client._calculate_delay(2) == 4.0

    @pytest.mark.asyncio
    async def test_complete_with_context_default(self):
        """Test default complete_with_context implementation."""
        client = Mock(spec=BaseLLMClient)
        client.complete = AsyncMock(return_value="test response")
        client.complete_with_context = BaseLLMClient.complete_with_context.__get__(client)

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]

        result = await client.complete_with_context(messages)
        print("\nresult:", result)

        # Should call complete with concatenated prompt
        client.complete.assert_called_once()
        call_args = client.complete.call_args[0][0]

        assert "System: You are helpful" in call_args
        assert "User: Hello" in call_args
        assert "Assistant: Hi there" in call_args
        assert "User: How are you?" in call_args
        assert result == "test response"


class MockOpenAIClient:
    """Mock OpenAI client for testing."""

    def __init__(self, response_content="test response", should_fail=False):
        self.response_content = response_content
        self.should_fail = should_fail
        self.chat = Mock()
        self.chat.completions = Mock()
        self.chat.completions.create = AsyncMock()

        if should_fail:
            self.chat.completions.create.side_effect = Exception("API Error")
        else:
            # Mock successful response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = response_content
            self.chat.completions.create.return_value = mock_response


class TestOpenAICompatibleClient:
    """Test OpenAI compatible client."""

    @pytest.mark.asyncio
    async def test_complete_success(self):
        """Test successful completion."""
        mock_openai = MockOpenAIClient("Hello world")

        client = OpenAICompatibleClient(api_key="test", base_url="test")
        client._client = mock_openai

        result = await client.complete("test prompt")

        assert result == "Hello world"
        mock_openai.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_with_context_success(self):
        """Test successful context completion."""
        mock_openai = MockOpenAIClient("Context response")

        client = OpenAICompatibleClient(api_key="test", base_url="test")
        client._client = mock_openai

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

        result = await client.complete_with_context(messages)

        assert result == "Context response"
        mock_openai.chat.completions.create.assert_called_once_with(
            model="deepseek-chat", messages=messages, temperature=0.2
        )

    @pytest.mark.asyncio
    async def test_complete_failure(self):
        """Test completion failure handling."""
        mock_openai = MockOpenAIClient(should_fail=True)

        client = OpenAICompatibleClient(api_key="test", base_url="test")
        client._client = mock_openai

        with pytest.raises(Exception, match="API Error"):
            await client.complete("test prompt")

    @pytest.mark.asyncio
    async def test_validate_connection_success(self):
        """Test connection validation success."""
        mock_openai = MockOpenAIClient("ok")

        client = OpenAICompatibleClient(api_key="test", base_url="test")
        client._client = mock_openai

        result = await client.validate_connection()
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_connection_failure(self):
        """Test connection validation failure."""
        mock_openai = MockOpenAIClient(should_fail=True)

        client = OpenAICompatibleClient(api_key="test", base_url="test")
        client._client = mock_openai

        result = await client.validate_connection()
        assert result is False

    def test_token_count_estimation(self):
        """Test token count estimation."""
        client = OpenAICompatibleClient(api_key="test", base_url="test")

        # Simple estimation: divide by 4
        assert client.get_token_count("hello world") == 2  # 11 chars / 4 = 2
        assert client.get_token_count("a" * 100) == 25  # 100 chars / 4 = 25


if __name__ == "__main__":
    # Run a simple test
    print("Running LLM client tests...")

    # Test retry config
    config = RetryConfig()
    assert config.max_attempts == 3
    print("✅ RetryConfig test passed")

    # Test error classification
    client = Mock(spec=BaseLLMClient)
    client._classify_error = BaseLLMClient._classify_error.__get__(client)
    assert client._classify_error(Exception("rate limit")) == "rate_limit"
    print("✅ Error classification test passed")

    print("✅ All LLM client tests passed!")
