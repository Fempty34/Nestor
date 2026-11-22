"""Configuration schema for multimodal Gemini preprocessing."""

import os

from pydantic import ConfigDict, Field, SecretStr

from nestor.core.config.llm import BaseLlmConfig


class GeminiPreprocessConfig(BaseLlmConfig):
    """Configuration for multimodal video/audio preprocessing via Gemini Files API."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_key: SecretStr | None = Field(
        default=None,
        description="Google Gemini API key. Falls back to GEMINI_API_KEY environment variable.",
    )
    default_model: str = Field(
        default="gemini-3.1-flash-lite",
        description="Target Gemini model for multimodal blackboard and speech extraction.",
    )
    poll_interval: float = Field(
        default=2.0,
        gt=0.0,
        description="Interval in seconds between Files API state checks.",
    )
    poll_timeout: float = Field(
        default=180.0,
        gt=0.0,
        description="Maximum wait time in seconds for file processing.",
    )

    @property
    def raw_api_key(self) -> str | None:
        """Resolve API key from explicit config or environment fallback."""
        if self.api_key is not None:
            return self.api_key.get_secret_value()
        return os.environ.get("GEMINI_API_KEY")
