"""Unit tests for the video processor port protocol and domain types."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from nestor.adapters.outbound.ffmpeg.fake_video import FakeVideoProcessor
from nestor.adapters.outbound.ffmpeg.video_adapter import (
    FFmpegVideoExecutionError,
    FFmpegVideoNotFoundError,
    FFmpegVideoProcessor,
    FFmpegVideoTimeoutError,
)
from nestor.domain.contracts.dtc import TimeRange
from nestor.ports.video import (
    NoVideoStreamError,
    VideoBatch,
    VideoExecutionError,
    VideoFrame,
    VideoNotFoundError,
    VideoProcessingResult,
    VideoProcessor,
    VideoProcessorError,
    VideoTimeoutError,
    VideoToolNotFoundError,
)


class TestVideoPortProtocols:
    """Verifies interface adherence and runtime protocol validation."""

    def test_ffmpeg_adapter_implements_protocol(self) -> None:
        """Verify FFmpegVideoProcessor satisfies runtime VideoProcessor protocol."""
        adapter = FFmpegVideoProcessor()
        assert isinstance(adapter, VideoProcessor)

    def test_fake_video_processor_implements_protocol(self) -> None:
        """Verify FakeVideoProcessor satisfies runtime VideoProcessor protocol."""
        fake = FakeVideoProcessor()
        assert isinstance(fake, VideoProcessor)

    def test_arbitrary_object_does_not_implement_protocol(self) -> None:
        """Verify non-conforming objects fail protocol runtime check."""
        assert not isinstance(object(), VideoProcessor)


class TestVideoPortExceptions:
    """Verifies inheritance hierarchy of video domain and adapter errors."""

    def test_exception_hierarchy(self) -> None:
        """Ensure all video exceptions inherit from VideoProcessorError."""
        assert issubclass(VideoNotFoundError, VideoProcessorError)
        assert issubclass(VideoToolNotFoundError, VideoProcessorError)
        assert issubclass(VideoExecutionError, VideoProcessorError)
        assert issubclass(VideoTimeoutError, VideoProcessorError)
        assert issubclass(NoVideoStreamError, VideoProcessorError)

    def test_adapter_exceptions_inherit_from_port_exceptions(self) -> None:
        """Ensure FFmpeg adapter specific errors inherit from generic port errors."""
        assert issubclass(FFmpegVideoNotFoundError, VideoToolNotFoundError)
        assert issubclass(FFmpegVideoExecutionError, VideoExecutionError)
        assert issubclass(FFmpegVideoTimeoutError, VideoTimeoutError)

    def test_video_execution_error_attributes(self) -> None:
        """Verify VideoExecutionError diagnostic fields."""
        err = VideoExecutionError(
            "Crash",
            returncode=1,
            cmd=["ffmpeg", "-i", "video.mp4"],
            stderr="err log",
            stdout="out log",
        )
        assert err.returncode == 1
        assert err.cmd == ["ffmpeg", "-i", "video.mp4"]
        assert err.stderr == "err log"
        assert err.stdout == "out log"


class TestVideoFrameContract:
    """Verifies immutability, validation constraints, and formatting of VideoFrame."""

    def test_valid_video_frame(self, tmp_path: Path) -> None:
        """Verify valid VideoFrame instantiation."""
        frame = VideoFrame(
            index=0,
            path=tmp_path / "frame_000001.jpg",
            timestamp=0.0,
        )
        assert frame.index == 0
        assert frame.timestamp == 0.0
        assert frame.timestamp_formatted == "00:00:00"

    def test_timestamp_formatted_property(self, tmp_path: Path) -> None:
        """Verify timestamp_formatted property for various time offsets."""
        f0 = VideoFrame(index=0, path=tmp_path / "f0.jpg", timestamp=0.0)
        assert f0.timestamp_formatted == "00:00:00"

        f1 = VideoFrame(index=1, path=tmp_path / "f1.jpg", timestamp=2.0)
        assert f1.timestamp_formatted == "00:00:02"

        f2 = VideoFrame(index=2, path=tmp_path / "f2.jpg", timestamp=65.0)
        assert f2.timestamp_formatted == "00:01:05"

        f3 = VideoFrame(index=3, path=tmp_path / "f3.jpg", timestamp=3665.0)
        assert f3.timestamp_formatted == "01:01:05"

    def test_frame_is_frozen(self, tmp_path: Path) -> None:
        """Verify VideoFrame is immutable."""
        frame = VideoFrame(
            index=0,
            path=tmp_path / "frame.jpg",
            timestamp=2.0,
        )
        with pytest.raises(ValidationError):
            frame.timestamp = 4.0  # type: ignore[misc]

    def test_invalid_negative_index_raises(self, tmp_path: Path) -> None:
        """Verify negative index raises ValidationError."""
        with pytest.raises(ValidationError):
            VideoFrame(
                index=-1,
                path=tmp_path / "frame.jpg",
                timestamp=0.0,
            )

    def test_invalid_negative_timestamp_raises(self, tmp_path: Path) -> None:
        """Verify negative timestamp raises ValidationError."""
        with pytest.raises(ValidationError):
            VideoFrame(
                index=0,
                path=tmp_path / "frame.jpg",
                timestamp=-1.0,
            )

    def test_extra_field_forbidden(self, tmp_path: Path) -> None:
        """Verify extra unexpected fields raise ValidationError."""
        with pytest.raises(ValidationError):
            VideoFrame(
                index=0,
                path=tmp_path / "frame.jpg",
                timestamp=0.0,
                extra_field="disallowed",  # type: ignore[call-arg]
            )


class TestVideoBatchContract:
    """Verifies VideoBatch invariants and frame count helper."""

    def test_valid_video_batch(self, tmp_path: Path) -> None:
        """Verify valid VideoBatch and frame_count property."""
        frames = [
            VideoFrame(
                index=0,
                path=tmp_path / "batch_0000" / "frame_000001.jpg",
                timestamp=0.0,
            ),
            VideoFrame(
                index=1,
                path=tmp_path / "batch_0000" / "frame_000002.jpg",
                timestamp=2.0,
            ),
        ]
        batch = VideoBatch(
            index=0,
            time_range=TimeRange(start=0.0, end=300.0),
            frames=frames,
            batch_dir=tmp_path / "batch_0000",
        )
        assert batch.index == 0
        assert batch.frame_count == 2
        assert batch.time_range.start == 0.0
        assert batch.time_range.end == 300.0


class TestVideoProcessingResultContract:
    """Verifies VideoProcessingResult container model."""

    def test_valid_processing_result(self, tmp_path: Path) -> None:
        """Verify result instantiation with batches and metadata."""
        result = VideoProcessingResult(
            batches=[],
            total_frames=0,
            fps=0.5,
            output_dir=tmp_path,
            log="Execution finished",
        )
        assert result.total_frames == 0
        assert result.fps == 0.5
        assert result.log == "Execution finished"
