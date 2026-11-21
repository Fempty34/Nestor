"""Unit tests for OpenRouter LLM adapter."""

from unittest.mock import AsyncMock, MagicMock

import openai
import pytest
from pydantic import BaseModel, Field, SecretStr

from nestor.adapters.outbound.llm.openrouter import OpenRouterClient
from nestor.core.config.llm import OpenRouterConfig
from nestor.ports.llm import (
    LlmAuthenticationError,
    LlmClient,
    LlmRateLimitError,
    LlmResponse,
    LlmSchemaValidationError,
    LlmTimeoutError,
)


class ProofStep(BaseModel):
    """Sample structured model representing a mathematical deduction step."""

    step_number: int = Field(ge=1)
    statement: str
    is_valid: bool


class TestOpenRouterClientInitialization:
    """Verifies initialization invariants and credential resolution."""

    def test_protocol_conformance(self) -> None:
        """Verify OpenRouterClient satisfies the abstract LlmClient protocol."""
        config = OpenRouterConfig(api_key=SecretStr("dummy-key"))
        client = OpenRouterClient(config=config)
        assert isinstance(client, LlmClient)

    def test_missing_api_key_raises_auth_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify LlmAuthenticationError is raised when no API key is provided or found."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(LlmAuthenticationError, match="OpenRouter API key not found"):
            OpenRouterClient(config=OpenRouterConfig(api_key=None))

    def test_reads_api_key_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify API key is picked up from OPENROUTER_API_KEY environment variable."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-secret")
        client = OpenRouterClient()
        assert client.default_model == "anthropic/claude-opus-5.5"

    def test_init_with_openrouter_config_object(self) -> None:
        """Verify client initializes properly when passing a typed OpenRouterConfig object."""
        config = OpenRouterConfig(
            api_key=SecretStr("typed-config-key"),
            default_model="anthropic/claude-sonnet-5.5",
            max_retries=5,
            site_url="https://nestor.ai",
            site_name="NestorApp",
        )
        client = OpenRouterClient(config=config)
        assert client.default_model == "anthropic/claude-sonnet-5.5"
        assert client.max_retries == 5
        assert client.config.raw_api_key == "typed-config-key"


class TestOpenRouterClientGenerate:
    """Verifies text generation behavior and parameter mapping."""

    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        """Verify standard text generation passes messages and returns normalized response."""
        mock_openai = MagicMock()
        mock_response = MagicMock()

        mock_choice = MagicMock()
        mock_choice.message.content = "Proof by induction step completed."
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]

        mock_openai.chat.completions.create = AsyncMock(return_value=mock_response)

        client = OpenRouterClient(client=mock_openai)
        result = await client.generate(
            prompt="Prove formula by mathematical induction",
            system_instruction="You are a discrete math TA.",
            model="anthropic/claude-sonnet-5.5",
            temperature=0.1,
            max_tokens=2048,
        )

        assert isinstance(result, LlmResponse)
        assert result.content == "Proof by induction step completed."
        assert result.model == "anthropic/claude-sonnet-5.5"
        assert result.finish_reason == "stop"
        assert result.parsed is None

        mock_openai.chat.completions.create.assert_awaited_once()
        _, kwargs = mock_openai.chat.completions.create.call_args
        assert kwargs["model"] == "anthropic/claude-sonnet-5.5"
        assert kwargs["temperature"] == 0.1
        assert kwargs["max_tokens"] == 2048
        assert kwargs["messages"] == [
            {"role": "system", "content": "You are a discrete math TA."},
            {"role": "user", "content": "Prove formula by mathematical induction"},
        ]


class TestOpenRouterClientStructured:
    """Verifies structured schema-constrained generation."""

    @pytest.mark.asyncio
    async def test_generate_structured_success_with_json_text(self) -> None:
        """Verify raw JSON text is parsed into requested Pydantic schema."""
        mock_openai = MagicMock()
        mock_response = MagicMock()

        mock_choice = MagicMock()
        mock_choice.message.content = (
            '{"step_number": 1, "statement": "Base case n=1 holds", "is_valid": true}'
        )
        mock_choice.message.parsed = None
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]

        mock_openai.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

        client = OpenRouterClient(client=mock_openai)
        result = await client.generate_structured(
            prompt="Verify base case",
            response_model=ProofStep,
        )

        assert isinstance(result, LlmResponse)
        assert isinstance(result.parsed, ProofStep)
        assert result.parsed.step_number == 1
        assert result.parsed.statement == "Base case n=1 holds"
        assert result.parsed.is_valid is True

        _, kwargs = mock_openai.beta.chat.completions.parse.call_args
        assert kwargs["response_format"] is ProofStep

    @pytest.mark.asyncio
    async def test_generate_structured_success_with_preparsed_instance(self) -> None:
        """Verify pre-parsed model instance from OpenAI beta parse is directly propagated."""
        expected_step = ProofStep(
            step_number=2, statement="Inductive step holds for k+1", is_valid=True
        )

        mock_openai = MagicMock()
        mock_response = MagicMock()

        mock_choice = MagicMock()
        mock_choice.message.content = expected_step.model_dump_json()
        mock_choice.message.parsed = expected_step
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]

        mock_openai.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

        client = OpenRouterClient(client=mock_openai)
        result = await client.generate_structured(
            prompt="Verify inductive step",
            response_model=ProofStep,
        )

        assert result.parsed == expected_step

    @pytest.mark.asyncio
    async def test_generate_structured_invalid_json_raises_schema_error(self) -> None:
        """Verify non-JSON response raises LlmSchemaValidationError."""
        mock_openai = MagicMock()
        mock_response = MagicMock()

        mock_choice = MagicMock()
        mock_choice.message.content = "Not a valid JSON string"
        mock_choice.message.parsed = None
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]

        mock_openai.beta.chat.completions.parse = AsyncMock(return_value=mock_response)

        client = OpenRouterClient(client=mock_openai)
        with pytest.raises(LlmSchemaValidationError, match="Failed to parse OpenRouter output"):
            await client.generate_structured(
                prompt="Extract",
                response_model=ProofStep,
            )


class TestOpenRouterClientErrorHandlingAndRetries:
    """Verifies error translation and tenacity retry policy."""

    @pytest.mark.asyncio
    async def test_rate_limit_retries_and_succeeds(self) -> None:
        """Verify transient rate limit (429) triggers exponential backoff retry and recovers."""
        mock_openai = MagicMock()

        mock_choice = MagicMock()
        mock_choice.message.content = "Answer recovered after rate limit"
        mock_choice.finish_reason = "stop"
        success_response = MagicMock()
        success_response.choices = [mock_choice]

        rate_limit_err = openai.RateLimitError(
            message="Rate limit reached",
            response=MagicMock(status_code=429),
            body=None,
        )

        mock_openai.chat.completions.create = AsyncMock(
            side_effect=[rate_limit_err, success_response]
        )

        config = OpenRouterConfig(max_retries=2, base_delay=0.01, max_delay=0.05)
        client = OpenRouterClient(config=config, client=mock_openai)

        result = await client.generate("Test prompt")
        assert result.content == "Answer recovered after rate limit"
        assert mock_openai.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_exhaustion_raises_rate_limit_error(self) -> None:
        """Verify persistent rate limit (429) exhausts retries and raises LlmRateLimitError."""
        mock_openai = MagicMock()
        rate_limit_err = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None,
        )

        mock_openai.chat.completions.create = AsyncMock(side_effect=rate_limit_err)

        config = OpenRouterConfig(max_retries=2, base_delay=0.01, max_delay=0.05)
        client = OpenRouterClient(config=config, client=mock_openai)

        with pytest.raises(LlmRateLimitError, match="OpenRouter rate limit exceeded"):
            await client.generate("Test prompt")

        assert mock_openai.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_auth_error_fails_immediately_without_retry(self) -> None:
        """Verify 401 authorization error fails immediately without retrying."""
        mock_openai = MagicMock()
        auth_err = openai.AuthenticationError(
            message="Invalid API Key",
            response=MagicMock(status_code=401),
            body=None,
        )

        mock_openai.chat.completions.create = AsyncMock(side_effect=auth_err)

        config = OpenRouterConfig(max_retries=3)
        client = OpenRouterClient(config=config, client=mock_openai)

        with pytest.raises(LlmAuthenticationError, match="OpenRouter authentication failed"):
            await client.generate("Test prompt")

        assert mock_openai.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_timeout_error_retries(self) -> None:
        """Verify connection or read timeout triggers retry."""
        mock_openai = MagicMock()
        timeout_err = openai.APITimeoutError(request=MagicMock())

        mock_openai.chat.completions.create = AsyncMock(side_effect=timeout_err)

        config = OpenRouterConfig(max_retries=2, base_delay=0.01, max_delay=0.05)
        client = OpenRouterClient(config=config, client=mock_openai)

        with pytest.raises(LlmTimeoutError, match="OpenRouter connection or timeout error"):
            await client.generate("Test prompt")

        assert mock_openai.chat.completions.create.await_count == 2
