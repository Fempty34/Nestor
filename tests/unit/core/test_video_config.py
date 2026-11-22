"""Unit tests for VideoConfig settings and validation."""

import pytest
from pydantic import ValidationError

from nestor.core.config.video import VideoConfig


class TestVideoConfig:
    """Verifies default values, validations, and environment overrides for VideoConfig."""

    def test_default_values(self) -> None:
        """Verify default configuration values."""
        config = VideoConfig()
        assert config.binary_path == "ffmpeg"
        assert config.fps == 0.5
        assert config.batch_duration == 300.0
        assert config.image_format == "jpg"
        assert config.jpeg_quality == 2
        assert config.timeout == 600.0

    def test_custom_values(self) -> None:
        """Verify custom configuration attributes."""
        config = VideoConfig(
            binary_path="/opt/bin/ffmpeg",
            fps=1.0,
            batch_duration=120.0,
            image_format="png",
            jpeg_quality=5,
            timeout=120.0,
        )
        assert config.binary_path == "/opt/bin/ffmpeg"
        assert config.fps == 1.0
        assert config.batch_duration == 120.0
        assert config.image_format == "png"
        assert config.jpeg_quality == 5
        assert config.timeout == 120.0

    def test_invalid_fps_raises(self) -> None:
        """Verify non-positive fps raises ValidationError."""
        with pytest.raises(ValidationError):
            VideoConfig(fps=0.0)

        with pytest.raises(ValidationError):
            VideoConfig(fps=-0.5)

    def test_invalid_batch_duration_raises(self) -> None:
        """Verify non-positive batch_duration raises ValidationError."""
        with pytest.raises(ValidationError):
            VideoConfig(batch_duration=0.0)

    def test_invalid_jpeg_quality_raises(self) -> None:
        """Verify jpeg_quality outside 1..31 raises ValidationError."""
        with pytest.raises(ValidationError):
            VideoConfig(jpeg_quality=0)

        with pytest.raises(ValidationError):
            VideoConfig(jpeg_quality=32)

    def test_invalid_image_format_raises(self) -> None:
        """Verify unsupported image_format raises ValidationError."""
        with pytest.raises(ValidationError):
            VideoConfig(image_format="bmp")

    def test_from_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify from_env loads defaults when env vars are unset."""
        monkeypatch.delenv("FFMPEG_PATH", raising=False)
        monkeypatch.delenv("VIDEO_FPS", raising=False)
        monkeypatch.delenv("VIDEO_BATCH_DURATION", raising=False)
        monkeypatch.delenv("FFMPEG_TIMEOUT", raising=False)

        config = VideoConfig.from_env()
        assert config.binary_path == "ffmpeg"
        assert config.fps == 0.5
        assert config.batch_duration == 300.0
        assert config.timeout == 600.0

    def test_from_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify from_env loads overrides from environment variables."""
        monkeypatch.setenv("FFMPEG_PATH", "/custom/ffmpeg")
        monkeypatch.setenv("VIDEO_FPS", "0.25")
        monkeypatch.setenv("VIDEO_BATCH_DURATION", "600.0")
        monkeypatch.setenv("FFMPEG_TIMEOUT", "900.0")

        config = VideoConfig.from_env()
        assert config.binary_path == "/custom/ffmpeg"
        assert config.fps == 0.25
        assert config.batch_duration == 600.0
        assert config.timeout == 900.0
