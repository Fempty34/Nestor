"""Unit tests for FFmpegConfig settings."""

import pytest
from pydantic import ValidationError

from nestor.core.config.audio import FFmpegConfig


class TestFFmpegConfig:
    """Verifies FFmpeg configuration validation, defaults, and environment loading."""

    def test_default_values(self) -> None:
        """Verify default configuration targets 16 kHz Mono and EBU R128 standards."""
        config = FFmpegConfig()
        assert config.binary_path == "ffmpeg"
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.segment_duration == 300.0
        assert config.loudnorm_i == -16.0
        assert config.loudnorm_tp == -1.5
        assert config.loudnorm_lra == 11.0
        assert config.timeout == 600.0

    def test_loudnorm_filter_arg_property(self) -> None:
        """Verify filter string construction matches FFmpeg syntax."""
        config = FFmpegConfig(loudnorm_i=-18.0, loudnorm_tp=-2.0, loudnorm_lra=10.0)
        assert config.loudnorm_filter_arg == "loudnorm=I=-18.0:TP=-2.0:LRA=10.0"

    def test_invalid_channels_rejected(self) -> None:
        """Verify channel counts outside [1, 2] are rejected."""
        with pytest.raises(ValidationError):
            FFmpegConfig(channels=0)
        with pytest.raises(ValidationError):
            FFmpegConfig(channels=3)

    def test_invalid_sample_rate_rejected(self) -> None:
        """Verify non-positive sample rate is rejected."""
        with pytest.raises(ValidationError):
            FFmpegConfig(sample_rate=0)
        with pytest.raises(ValidationError):
            FFmpegConfig(sample_rate=-16000)

    def test_invalid_segment_duration_rejected(self) -> None:
        """Verify non-positive segment duration is rejected."""
        with pytest.raises(ValidationError):
            FFmpegConfig(segment_duration=0.0)

    def test_invalid_loudnorm_ranges_rejected(self) -> None:
        """Verify out-of-range EBU R128 parameters are rejected."""
        with pytest.raises(ValidationError):
            FFmpegConfig(loudnorm_i=-80.0)  # below -70
        with pytest.raises(ValidationError):
            FFmpegConfig(loudnorm_tp=1.0)  # above 0.0
        with pytest.raises(ValidationError):
            FFmpegConfig(loudnorm_lra=0.5)  # below 1.0

    def test_from_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify from_env uses standard defaults when env vars are absent."""
        monkeypatch.delenv("FFMPEG_PATH", raising=False)
        monkeypatch.delenv("FFMPEG_TIMEOUT", raising=False)

        config = FFmpegConfig.from_env()
        assert config.binary_path == "ffmpeg"
        assert config.timeout == 600.0

    def test_from_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify from_env loads environment variable overrides."""
        monkeypatch.setenv("FFMPEG_PATH", "/usr/local/bin/ffmpeg")
        monkeypatch.setenv("FFMPEG_TIMEOUT", "1200.5")

        config = FFmpegConfig.from_env()
        assert config.binary_path == "/usr/local/bin/ffmpeg"
        assert config.timeout == 1200.5
