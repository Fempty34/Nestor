"""Audio processing port protocol, result types, and audio exceptions.

Defines the contract for extracting, normalizing, and segmenting lecture audio streams.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from nestor.domain.contracts.dtc import TimeRange


class AudioSegment(BaseModel):
    """Standardized representation of a sliced, normalized audio chunk."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int = Field(ge=0, description="0-based monotonically increasing segment index.")
    path: Path = Field(description="Filesystem path to the WAV segment artifact.")
    time_range: TimeRange = Field(description="Exact temporal interval in the master lecture.")
    sample_rate: int = Field(default=16000, description="Audio sampling rate in Hz (e.g. 16000).")
    channels: int = Field(default=1, description="Number of audio channels.")

    @property
    def duration(self) -> float:
        """Calculate segment duration in seconds from its time range."""
        return self.time_range.duration


class AudioProcessingResult(BaseModel):
    """Result container holding extracted segments and overall execution metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    segments: list[AudioSegment] = Field(
        default_factory=list, description="Ordered list of temporal audio segments."
    )
    total_duration: float = Field(ge=0.0, description="Total aggregated audio duration in seconds.")
    output_dir: Path = Field(description="Directory containing the produced audio files.")
    log: str = Field(default="", description="Diagnostic execution logs from the audio backend.")


class AudioProcessorError(Exception):
    """Base exception for all audio processing port and adapter operations."""


class AudioNotFoundError(AudioProcessorError):
    """Raised when the input media file is missing or inaccessible."""


class AudioToolNotFoundError(AudioProcessorError):
    """Raised when the required audio processing engine or binary is not found."""


class AudioExecutionError(AudioProcessorError):
    """Raised when audio processing terminates with an execution failure."""

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


class AudioTimeoutError(AudioProcessorError):
    """Raised when audio processing exceeds the configured execution deadline."""


class NoAudioStreamError(AudioProcessorError):
    """Raised when the input media contains no audio streams."""


@runtime_checkable
class AudioProcessor(Protocol):
    """Abstract port for extracting, normalizing, and slicing lecture audio."""

    async def process_lecture_audio(
        self,
        video_path: Path,
        output_dir: Path,
        *,
        segment_duration: float = 300.0,
        timeout: float | None = None,
    ) -> AudioProcessingResult:
        """Extract audio to 16 kHz Mono WAV, normalize loudness, and slice into segments."""
        ...
