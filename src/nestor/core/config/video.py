"""Configuration settings for video sampling and frame extraction."""

import os
from typing import Self

from pydantic import BaseModel, ConfigDict, Field


class VideoConfig(BaseModel):
    """Configuration settings for video frame sampling and batch segmentation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    binary_path: str = Field(
        default="ffmpeg",
        description="Path or binary name for the FFmpeg executable.",
    )
    fps: float = Field(
        default=0.5,
        gt=0.0,
        description="Frame sampling rate in frames per second (default 0.5 fps = 1 frame every 2s).",
    )
    batch_duration: float = Field(
        default=300.0,
        gt=0.0,
        description="Target duration of individual batches in seconds (default 300s = 5 minutes).",
    )
    image_format: str = Field(
        default="jpg",
        pattern=r"^(jpg|jpeg|png)$",
        description="Target image extension and format for extracted frames.",
    )
    jpeg_quality: int = Field(
        default=2,
        ge=1,
        le=31,
        description="FFmpeg qscale / q:v compression quality level (1 is highest quality, 31 lowest).",
    )
    timeout: float = Field(
        default=600.0,
        gt=0.0,
        description="Maximum execution timeout in seconds per video extraction process.",
    )

    @classmethod
    def from_env(cls) -> Self:
        """Create VideoConfig loading overrides from environment variables."""
        binary_path = os.environ.get("FFMPEG_PATH", "ffmpeg")
        fps_env = os.environ.get("VIDEO_FPS")
        fps = float(fps_env) if fps_env else 0.5
        batch_env = os.environ.get("VIDEO_BATCH_DURATION")
        batch_duration = float(batch_env) if batch_env else 300.0
        timeout_env = os.environ.get("FFMPEG_TIMEOUT")
        timeout = float(timeout_env) if timeout_env else 600.0
        return cls(
            binary_path=binary_path,
            fps=fps,
            batch_duration=batch_duration,
            timeout=timeout,
        )
