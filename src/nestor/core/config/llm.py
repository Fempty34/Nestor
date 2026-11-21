"""LLM provider configuration schemas and settings loaders."""

import os
import tomllib
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class BaseLlmConfig(BaseModel):
    """Base configuration shared by all LLM adapters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_retries: int = Field(
        default=3, ge=0, description="Maximum retry attempts on transient errors."
    )
    base_delay: float = Field(default=1.0, gt=0.0, description="Initial retry delay in seconds.")
    max_delay: float = Field(
        default=30.0, gt=0.0, description="Maximum retry backoff delay in seconds."
    )
    timeout: float = Field(default=60.0, gt=0.0, description="Request timeout in seconds.")


class GeminiConfig(BaseLlmConfig):
    """Configuration for Google Gemini provider."""

    api_key: SecretStr | None = Field(
        default=None,
        description="Google Gemini API key. If None, falls back to GEMINI_API_KEY environment variable.",
    )
    default_model: str = Field(
        default="gemini-2.0-flash",
        description="Default Gemini model identifier.",
    )

    @property
    def raw_api_key(self) -> str | None:
        """Resolve API key from explicit config or environment fallback."""
        if self.api_key is not None:
            return self.api_key.get_secret_value()
        return os.environ.get("GEMINI_API_KEY")


class OpenRouterConfig(BaseLlmConfig):
    """Configuration for OpenRouter provider."""

    api_key: SecretStr | None = Field(
        default=None,
        description="OpenRouter API key. If None, falls back to OPENROUTER_API_KEY environment variable.",
    )
    default_model: str = Field(
        default="anthropic/claude-opus-5.5",
        description="Default model identifier route on OpenRouter.",
    )
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for OpenRouter API.",
    )
    site_url: str | None = Field(
        default=None,
        description="Optional HTTP-Referer header for OpenRouter rankings.",
    )
    site_name: str = Field(
        default="Nestor",
        description="Optional X-Title header identifying the client application.",
    )

    @property
    def raw_api_key(self) -> str | None:
        """Resolve API key from explicit config or environment fallback."""
        if self.api_key is not None:
            return self.api_key.get_secret_value()
        return os.environ.get("OPENROUTER_API_KEY")


class LlmSettings(BaseModel):
    """Consolidated LLM configuration for the multi-agent system."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    openrouter: OpenRouterConfig = Field(default_factory=OpenRouterConfig)

    @classmethod
    def from_toml(cls, path: Path | str) -> Self:
        """Load settings from a TOML configuration file."""
        file_path = Path(path)
        with file_path.open("rb") as f:
            data = tomllib.load(f)
        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Load settings from an in-memory dictionary."""
        return cls.model_validate(data)
