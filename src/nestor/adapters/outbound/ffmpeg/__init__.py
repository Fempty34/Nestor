"""FFmpeg outbound adapter package."""

from nestor.adapters.outbound.ffmpeg.adapter import (
    AsyncSubprocessRunner,
    FFmpegAudioProcessor,
    FFmpegExecutionError,
    FFmpegNotFoundError,
    FFmpegTimeoutError,
)
from nestor.adapters.outbound.ffmpeg.fake import (
    AudioCallRecord,
    FakeAudioProcessor,
)

__all__ = [
    "AsyncSubprocessRunner",
    "AudioCallRecord",
    "FFmpegAudioProcessor",
    "FFmpegExecutionError",
    "FFmpegNotFoundError",
    "FFmpegTimeoutError",
    "FakeAudioProcessor",
]
