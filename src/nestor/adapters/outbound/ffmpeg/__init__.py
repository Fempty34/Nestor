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
from nestor.adapters.outbound.ffmpeg.fake_video import (
    FakeVideoProcessor,
    VideoCallRecord,
)
from nestor.adapters.outbound.ffmpeg.video_adapter import (
    FFmpegVideoExecutionError,
    FFmpegVideoNotFoundError,
    FFmpegVideoProcessor,
    FFmpegVideoTimeoutError,
)

__all__ = [
    "AsyncSubprocessRunner",
    "AudioCallRecord",
    "FFmpegAudioProcessor",
    "FFmpegExecutionError",
    "FFmpegNotFoundError",
    "FFmpegTimeoutError",
    "FFmpegVideoExecutionError",
    "FFmpegVideoNotFoundError",
    "FFmpegVideoProcessor",
    "FFmpegVideoTimeoutError",
    "FakeAudioProcessor",
    "FakeVideoProcessor",
    "VideoCallRecord",
]
