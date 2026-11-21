"""Unit tests for LLM configuration schemas and loaders."""

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from nestor.core.config.llm import (
    GeminiConfig,
    LlmSettings,
    OpenRouterConfig,
)


class TestGeminiConfig:
    """Verifies GeminiConfig creation, validation, and secret masking."""

    def test_direct_code_configuration_with_string_key(self) -> None:
        """Verify API key can be set directly via code as a string."""
        config = GeminiConfig(
            api_key=SecretStr("custom-api-key"),
            default_model="gemini-2.0-pro",
            max_retries=5,
        )
        assert config.raw_api_key == "custom-api-key"
        assert config.default_model == "gemini-2.0-pro"
        assert config.max_retries == 5

        assert "custom-api-key" not in repr(config)
        assert "custom-api-key" not in str(config)

    def test_fallback_to_environment_when_key_is_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify raw_api_key falls back to GEMINI_API_KEY environment variable."""
        monkeypatch.setenv("GEMINI_API_KEY", "env-key-12345")
        config = GeminiConfig()
        assert config.raw_api_key == "env-key-12345"

    def test_extra_forbidden_fields(self) -> None:
        """Verify passing unrecognized fields raises ValidationError."""
        with pytest.raises(ValidationError):
            GeminiConfig(unexpected_param="invalid")  # type: ignore[call-arg]


class TestOpenRouterConfig:
    """Verifies OpenRouterConfig creation, defaults, and headers."""

    def test_direct_code_configuration(self) -> None:
        """Verify OpenRouterConfig initializes with expected defaults."""
        config = OpenRouterConfig(
            api_key=SecretStr("sk-or-test-key"),
            default_model="openai/gpt-4o",
            site_url="https://nestor.ai",
        )
        assert config.raw_api_key == "sk-or-test-key"
        assert config.default_model == "openai/gpt-4o"
        assert config.base_url == "https://openrouter.ai/api/v1"
        assert config.site_url == "https://nestor.ai"
        assert config.site_name == "Nestor"
        assert "sk-or-test-key" not in repr(config)

    def test_fallback_to_environment_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify raw_api_key falls back to OPENROUTER_API_KEY environment variable."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-987")
        config = OpenRouterConfig()
        assert config.raw_api_key == "sk-or-env-987"

    def test_extra_forbidden_fields(self) -> None:
        """Verify passing unrecognized fields raises ValidationError."""
        with pytest.raises(ValidationError):
            OpenRouterConfig(unknown_setting="bad")  # type: ignore[call-arg]


class TestLlmSettingsLoaders:
    """Verifies consolidated settings loading from code, dict, and TOML."""

    def test_direct_code_initialization(self) -> None:
        """Verify full LlmSettings can be instantiated purely in Python code."""
        settings = LlmSettings(
            gemini=GeminiConfig(
                api_key=SecretStr("gemini-code-key"),
                default_model="gemini-2.0-flash",
            ),
            openrouter=OpenRouterConfig(
                api_key=SecretStr("openrouter-code-key"),
                default_model="anthropic/claude-opus-5.5",
            ),
        )
        assert settings.gemini.raw_api_key == "gemini-code-key"
        assert settings.openrouter.raw_api_key == "openrouter-code-key"

    def test_load_from_dict(self) -> None:
        """Verify settings can be loaded from an in-memory dictionary."""
        data = {
            "gemini": {
                "api_key": "dict-gemini-key",
                "default_model": "gemini-1.5-pro",
                "max_retries": 4,
            },
            "openrouter": {
                "api_key": "dict-or-key",
                "default_model": "meta-llama/llama-3.3-70b",
            },
        }
        settings = LlmSettings.from_dict(data)
        assert settings.gemini.raw_api_key == "dict-gemini-key"
        assert settings.gemini.default_model == "gemini-1.5-pro"
        assert settings.gemini.max_retries == 4
        assert settings.openrouter.raw_api_key == "dict-or-key"
        assert settings.openrouter.default_model == "meta-llama/llama-3.3-70b"

    def test_load_from_toml_file(self, tmp_path: Path) -> None:
        """Verify settings can be loaded from a TOML configuration file."""
        toml_content = """
        [gemini]
        api_key = "toml-gemini-secret"
        default_model = "gemini-2.0-flash"
        max_retries = 2

        [openrouter]
        api_key = "toml-openrouter-secret"
        default_model = "anthropic/claude-sonnet-5.5"
        """
        config_file = tmp_path / "config.toml"
        config_file.write_text(toml_content, encoding="utf-8")

        settings = LlmSettings.from_toml(config_file)
        assert settings.gemini.raw_api_key == "toml-gemini-secret"
        assert settings.gemini.max_retries == 2
        assert settings.openrouter.raw_api_key == "toml-openrouter-secret"
        assert settings.openrouter.default_model == "anthropic/claude-sonnet-5.5"
