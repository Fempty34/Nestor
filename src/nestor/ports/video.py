"""Video processing port protocol, result types, and video exceptions.

Defines the contract for sampling video frames, associating master timestamps,
and grouping frames into temporal batches aligned with audio chunks.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from nestor.domain.contracts.dtc import TimeRange


class VideoFrame(BaseModel):
    """Standardized representation of a single sampled video frame."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(
        ge=0,
        description="0-based monotonically increasing global frame index.",
    )
    path: Path = Field(
        description="Filesystem path to the stored frame image artifact.",
    )
    timestamp: float = Field(
        ge=0.0,
        description="Exact temporal offset in seconds from the beginning of the lecture.",
    )

    @property
    def timestamp_formatted(self) -> str:
        """Return master timestamp formatted as HH:MM:SS."""
        total_sec = int(self.timestamp)
        hours = total_sec // 3600
        minutes = (total_sec % 3600) // 60
        secs = total_sec % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class VideoBatch(BaseModel):
    """Temporal batch of sampled frames matching an audio chunk duration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(
        ge=0,
        description="0-based batch index matching the corresponding AudioSegment index.",
    )
    time_range: TimeRange = Field(
        description="Temporal interval covered by this batch.",
    )
    frames: list[VideoFrame] = Field(
        default_factory=list,
        description="Ordered sequence of video frames contained within this batch.",
    )
    batch_dir: Path = Field(
        description="Isolated directory containing the frame image files for this batch.",
    )

    @property
    def frame_count(self) -> int:
        """Return the number of frames contained in this batch."""
        return len(self.frames)


class VideoProcessingResult(BaseModel):
    """Container holding extracted video batches and execution diagnostics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    batches: list[VideoBatch] = Field(
        default_factory=list,
        description="Chronologically ordered sequence of video batches.",
    )
    total_frames: int = Field(
        ge=0,
        description="Total count of sampled video frames across all batches.",
    )
    total_duration: float = Field(
        default=0.0,
        ge=0.0,
        description="Total duration in seconds covered by the extracted video batches.",
    )
    fps: float = Field(
        gt=0.0,
        description="Sampling rate in frames per second used for extraction.",
    )
    output_dir: Path = Field(
        description="Root output directory containing the isolated batch subdirectories.",
    )
    log: str = Field(
        default="",
        description="Diagnostic execution logs from the video processing backend.",
    )


class VideoProcessorError(Exception):
    """Base exception for all video processing port and adapter operations."""


class VideoNotFoundError(VideoProcessorError):
    """Raised when the input video file is missing or inaccessible."""


class VideoToolNotFoundError(VideoProcessorError):
    """Raised when the required video processing engine or binary is not found."""


class VideoExecutionError(VideoProcessorError):
    """Raised when video processing terminates with an execution failure."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        cmd: Sequence[str] | None = None,
        stderr: str = "",
        stdout: str = "",
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.cmd = list(cmd) if cmd else []
        self.stderr = stderr
        self.stdout = stdout


class VideoTimeoutError(VideoProcessorError):
    """Raised when video processing exceeds the configured execution deadline."""


class NoVideoStreamError(VideoProcessorError):
    """Raised when the input media contains no valid video streams."""


@runtime_checkable
class VideoProcessor(Protocol):
    """Abstract port for sampling lecture video frames and organizing them into batches."""

    async def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        *,
        fps: float = 0.5,
        batch_duration: float = 300.0,
        timeout: float | None = None,
    ) -> VideoProcessingResult:
        """Sample video frames at constant fps, assign master timestamps, and group into batches."""
        ...
