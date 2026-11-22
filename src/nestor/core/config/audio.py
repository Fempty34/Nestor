"""Configuration settings for audio processing and FFmpeg."""

import os
from typing import Self

from pydantic import BaseModel, ConfigDict, Field


class FFmpegConfig(BaseModel):
    """Configuration settings for FFmpeg audio extraction and normalization."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    binary_path: str = Field(
        default="ffmpeg",
        description="Path or binary name for the FFmpeg executable.",
    )
    sample_rate: int = Field(
        default=16000,
        gt=0,
        description="Target sampling frequency in Hz.",
    )
    channels: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Number of output audio channels.",
    )
    segment_duration: float = Field(
        default=300.0,
        gt=0.0,
        description="Target duration of individual sliced segments in seconds (default 5 minutes).",
    )
    loudnorm_i: float = Field(
        default=-16.0,
        ge=-70.0,
        le=-5.0,
        description="EBU R128 integrated loudness target in LUFS.",
    )
    loudnorm_tp: float = Field(
        default=-1.5,
        ge=-9.0,
        le=0.0,
        description="Maximum true peak level in dBFS.",
    )
    loudnorm_lra: float = Field(
        default=11.0,
        ge=1.0,
        le=50.0,
        description="Target loudness range in LU.",
    )
    timeout: float = Field(
        default=600.0,
        gt=0.0,
        description="Maximum execution timeout in seconds per audio extraction process.",
    )

    @property
    def loudnorm_filter_arg(self) -> str:
        """Construct the FFmpeg audio filter string for EBU R128 loudness normalization."""
        return f"loudnorm=I={self.loudnorm_i}:TP={self.loudnorm_tp}:LRA={self.loudnorm_lra}"

    @classmethod
    def from_env(cls) -> Self:
        """Create FFmpegConfig loading overrides from environment variables."""
        binary_path = os.environ.get("FFMPEG_PATH", "ffmpeg")
        timeout_env = os.environ.get("FFMPEG_TIMEOUT")
        timeout = float(timeout_env) if timeout_env else 600.0
        return cls(binary_path=binary_path, timeout=timeout)
