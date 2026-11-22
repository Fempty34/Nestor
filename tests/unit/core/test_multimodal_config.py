"""Unit tests for Gemini multimodal preprocessing configuration."""

import pytest
from pydantic import SecretStr, ValidationError

from nestor.core.config.multimodal import GeminiPreprocessConfig


class TestGeminiPreprocessConfig:
    """Verifies defaults, validation rules, immutability, and API key resolution."""

    def test_default_values(self) -> None:
        """Verify default model is gemini-3.1-flash-lite and timeouts match specs."""
        config = GeminiPreprocessConfig()
        assert config.default_model == "gemini-3.1-flash-lite"
        assert config.poll_interval == 2.0
        assert config.poll_timeout == 180.0
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0

    def test_custom_values(self) -> None:
        """Verify custom settings are preserved."""
        config = GeminiPreprocessConfig(
            api_key=SecretStr("custom-key"),
            default_model="gemini-2.0-flash",
            poll_interval=1.0,
            poll_timeout=60.0,
            max_retries=5,
        )
        assert config.raw_api_key == "custom-key"
        assert config.default_model == "gemini-2.0-flash"
        assert config.poll_interval == 1.0
        assert config.poll_timeout == 60.0
        assert config.max_retries == 5

    def test_environment_api_key_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify fallback to GEMINI_API_KEY environment variable when api_key is None."""
        monkeypatch.setenv("GEMINI_API_KEY", "env-test-key")
        config = GeminiPreprocessConfig()
        assert config.raw_api_key == "env-test-key"

    def test_immutability(self) -> None:
        """Verify GeminiPreprocessConfig is frozen."""
        config = GeminiPreprocessConfig()
        with pytest.raises(ValidationError):
            config.poll_interval = 5.0  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Verify extra attributes are rejected."""
        with pytest.raises(ValidationError):
            GeminiPreprocessConfig(unknown_field="invalid")  # type: ignore[call-arg]
