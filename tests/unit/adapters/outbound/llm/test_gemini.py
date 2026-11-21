"""Unit tests for Google Gemini LLM adapter."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import errors, types
from pydantic import BaseModel, Field, SecretStr

from nestor.adapters.outbound.llm.gemini import GeminiClient
from nestor.core.config.llm import GeminiConfig
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


class TestGeminiClientInitialization:
    """Verifies initialization invariants and credential resolution."""

    def test_protocol_conformance(self) -> None:
        """Verify GeminiClient satisfies the abstract LlmClient protocol."""
        config = GeminiConfig(api_key=SecretStr("dummy-key"))
        client = GeminiClient(config=config)
        assert isinstance(client, LlmClient)

    def test_missing_api_key_raises_auth_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify LlmAuthenticationError is raised when no API key is provided or found."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(LlmAuthenticationError, match="Gemini API key not found"):
            GeminiClient(config=GeminiConfig(api_key=None))

    def test_reads_api_key_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify API key is picked up from GEMINI_API_KEY environment variable."""
        monkeypatch.setenv("GEMINI_API_KEY", "env-secret-key")
        client = GeminiClient()
        assert client.default_model == "gemini-2.0-flash"

    def test_init_with_gemini_config_object(self) -> None:
        """Verify client initializes properly when passing a typed GeminiConfig object."""
        config = GeminiConfig(
            api_key=SecretStr("typed-config-key"),
            default_model="gemini-2.0-pro",
            max_retries=5,
        )
        client = GeminiClient(config=config)
        assert client.default_model == "gemini-2.0-pro"
        assert client.max_retries == 5
        assert client.config.raw_api_key == "typed-config-key"


class TestGeminiClientGenerate:
    """Verifies text generation behavior and parameter mapping."""

    @pytest.mark.asyncio
    async def test_generate_success(self) -> None:
        """Verify standard text generation passes configuration and returns normalized response."""
        mock_genai = MagicMock()
        mock_response = MagicMock(spec=types.GenerateContentResponse)
        mock_response.text = "The limit of sin(x)/x as x approaches 0 is 1."

        candidate = MagicMock()
        candidate.finish_reason = types.FinishReason.STOP
        mock_response.candidates = [candidate]

        mock_genai.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = GeminiClient(client=mock_genai)
        result = await client.generate(
            prompt="Compute the limit of sin(x)/x",
            system_instruction="You are a calculus professor.",
            model="gemini-2.0-pro",
            temperature=0.2,
            max_tokens=1024,
        )

        assert isinstance(result, LlmResponse)
        assert result.content == "The limit of sin(x)/x as x approaches 0 is 1."
        assert result.model == "gemini-2.0-pro"
        assert result.finish_reason == "stop"
        assert result.parsed is None

        mock_genai.aio.models.generate_content.assert_awaited_once()
        _, kwargs = mock_genai.aio.models.generate_content.call_args
        assert kwargs["model"] == "gemini-2.0-pro"
        assert kwargs["contents"] == "Compute the limit of sin(x)/x"

        config = kwargs["config"]
        assert isinstance(config, types.GenerateContentConfig)
        assert config.system_instruction == "You are a calculus professor."
        assert config.temperature == 0.2
        assert config.max_output_tokens == 1024


class TestGeminiClientStructured:
    """Verifies structured schema-constrained generation."""

    @pytest.mark.asyncio
    async def test_generate_structured_success_with_json_text(self) -> None:
        """Verify raw JSON text is parsed into the requested Pydantic schema."""
        mock_genai = MagicMock()
        mock_response = MagicMock(spec=types.GenerateContentResponse)
        mock_response.text = (
            '{"step_number": 1, "statement": "cos(x) <= sin(x)/x <= 1", "is_valid": true}'
        )
        mock_response.parsed = None
        mock_response.candidates = []

        mock_genai.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = GeminiClient(client=mock_genai)
        result = await client.generate_structured(
            prompt="Analyze inequality step",
            response_model=ProofStep,
        )

        assert isinstance(result, LlmResponse)
        assert isinstance(result.parsed, ProofStep)
        assert result.parsed.step_number == 1
        assert result.parsed.statement == "cos(x) <= sin(x)/x <= 1"
        assert result.parsed.is_valid is True

        _, kwargs = mock_genai.aio.models.generate_content.call_args
        config = kwargs["config"]
        assert config.response_mime_type == "application/json"
        assert config.response_schema is ProofStep

    @pytest.mark.asyncio
    async def test_generate_structured_success_with_preparsed_instance(self) -> None:
        """Verify pre-parsed model instance from Gemini SDK is directly propagated."""
        expected_step = ProofStep(step_number=2, statement="Apply squeeze theorem", is_valid=True)

        mock_genai = MagicMock()
        mock_response = MagicMock(spec=types.GenerateContentResponse)
        mock_response.text = expected_step.model_dump_json()
        mock_response.parsed = expected_step
        mock_response.candidates = []

        mock_genai.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = GeminiClient(client=mock_genai)
        result = await client.generate_structured(
            prompt="Next step",
            response_model=ProofStep,
        )

        assert result.parsed == expected_step

    @pytest.mark.asyncio
    async def test_generate_structured_invalid_json_raises_schema_error(self) -> None:
        """Verify non-JSON response raises LlmSchemaValidationError."""
        mock_genai = MagicMock()
        mock_response = MagicMock(spec=types.GenerateContentResponse)
        mock_response.text = "Malformed non-JSON output"
        mock_response.parsed = None
        mock_response.candidates = []

        mock_genai.aio.models.generate_content = AsyncMock(return_value=mock_response)

        client = GeminiClient(client=mock_genai)
        with pytest.raises(LlmSchemaValidationError, match="Failed to parse Gemini output"):
            await client.generate_structured(
                prompt="Extract",
                response_model=ProofStep,
            )


class TestGeminiClientErrorHandlingAndRetries:
    """Verifies error translation and tenacity retry policy."""

    @pytest.mark.asyncio
    async def test_rate_limit_retries_and_succeeds(self) -> None:
        """Verify transient rate limit (429) triggers exponential backoff retry and recovers."""
        mock_genai = MagicMock()

        success_response = MagicMock(spec=types.GenerateContentResponse)
        success_response.text = "Successful answer after rate limit retry"
        success_response.candidates = []

        rate_limit_err = errors.APIError(429, "Resource exhausted")

        mock_genai.aio.models.generate_content = AsyncMock(
            side_effect=[rate_limit_err, success_response]
        )

        config = GeminiConfig(max_retries=2, base_delay=0.01, max_delay=0.05)
        client = GeminiClient(config=config, client=mock_genai)

        result = await client.generate("Test prompt")
        assert result.content == "Successful answer after rate limit retry"
        assert mock_genai.aio.models.generate_content.await_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_exhaustion_raises_rate_limit_error(self) -> None:
        """Verify persistent rate limit (429) exhausts retries and raises LlmRateLimitError."""
        mock_genai = MagicMock()
        rate_limit_err = errors.APIError(429, "Quota exceeded")

        mock_genai.aio.models.generate_content = AsyncMock(side_effect=rate_limit_err)

        config = GeminiConfig(max_retries=2, base_delay=0.01, max_delay=0.05)
        client = GeminiClient(config=config, client=mock_genai)

        with pytest.raises(LlmRateLimitError, match="Gemini rate limit exceeded"):
            await client.generate("Test prompt")

        assert mock_genai.aio.models.generate_content.await_count == 2

    @pytest.mark.asyncio
    async def test_auth_error_fails_immediately_without_retry(self) -> None:
        """Verify 401/403 authorization error fails immediately without retrying."""
        mock_genai = MagicMock()
        auth_err = errors.APIError(401, "Invalid API Key")

        mock_genai.aio.models.generate_content = AsyncMock(side_effect=auth_err)

        config = GeminiConfig(max_retries=3)
        client = GeminiClient(config=config, client=mock_genai)

        with pytest.raises(LlmAuthenticationError, match="Gemini authentication failed"):
            await client.generate("Test prompt")

        assert mock_genai.aio.models.generate_content.await_count == 1

    @pytest.mark.asyncio
    async def test_timeout_error_retries(self) -> None:
        """Verify client timeout triggers retry."""
        mock_genai = MagicMock()
        timeout_err = TimeoutError("Network timed out")

        mock_genai.aio.models.generate_content = AsyncMock(side_effect=timeout_err)

        config = GeminiConfig(max_retries=2, base_delay=0.01, max_delay=0.05)
        client = GeminiClient(config=config, client=mock_genai)

        with pytest.raises(LlmTimeoutError, match="Gemini client timeout"):
            await client.generate("Test prompt")

        assert mock_genai.aio.models.generate_content.await_count == 2
