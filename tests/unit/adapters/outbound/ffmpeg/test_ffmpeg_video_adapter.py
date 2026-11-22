"""Unit tests for the FFmpeg video processing outbound adapter."""

import shutil
from pathlib import Path

import pytest

from nestor.adapters.outbound.ffmpeg.fake_video import MINIMAL_VALID_JPEG_BYTES
from nestor.adapters.outbound.ffmpeg.video_adapter import (
    FFmpegVideoExecutionError,
    FFmpegVideoNotFoundError,
    FFmpegVideoProcessor,
    FFmpegVideoTimeoutError,
)
from nestor.core.config.video import VideoConfig
from nestor.ports.video import (
    NoVideoStreamError,
    VideoNotFoundError,
    VideoProcessor,
    VideoProcessorError,
)


class TestFFmpegVideoProcessorInitialization:
    """Verifies protocol conformance and binary resolution."""

    def test_protocol_conformance(self) -> None:
        """Verify FFmpegVideoProcessor conforms to abstract VideoProcessor protocol."""
        processor = FFmpegVideoProcessor()
        assert isinstance(processor, VideoProcessor)

    def test_missing_binary_raises_not_found_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify FFmpegVideoNotFoundError raised when binary is not in PATH."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: None)
        processor = FFmpegVideoProcessor(config=VideoConfig(binary_path="nonexistent_ffmpeg_bin"))

        with pytest.raises(FFmpegVideoNotFoundError, match="FFmpeg binary not found"):
            processor._validate_binary_availability()


class TestFFmpegVideoProcessorExecution:
    """Tests execution flow, command args generation, batching, and error handling via mocked runners."""

    @pytest.mark.asyncio
    async def test_missing_input_video_raises_video_not_found(self, tmp_path: Path) -> None:
        """Verify VideoNotFoundError raised when input video does not exist."""
        processor = FFmpegVideoProcessor()
        with pytest.raises(VideoNotFoundError, match="Input video file not found"):
            await processor.extract_frames(tmp_path / "nonexistent.mp4", tmp_path / "out")

    @pytest.mark.asyncio
    async def test_invalid_arguments_raise_value_error(self, tmp_path: Path) -> None:
        """Verify non-positive fps and batch_duration raise ValueError."""
        video_path = tmp_path / "sample.mp4"
        video_path.touch()
        processor = FFmpegVideoProcessor()

        with pytest.raises(ValueError, match="Sampling rate 'fps' must be strictly positive"):
            await processor.extract_frames(video_path, tmp_path / "out", fps=0.0)

        with pytest.raises(ValueError, match="Batch duration must be strictly positive"):
            await processor.extract_frames(video_path, tmp_path / "out", batch_duration=-1.0)

    @pytest.mark.asyncio
    async def test_successful_sampling_and_batch_alignment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify CLI flags, frame extraction, master timestamps, and 5-minute batch grouping."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/ffmpeg")

        video_path = tmp_path / "lecture.mp4"
        video_path.touch()
        output_dir = tmp_path / "output_batches"

        captured_cmd: list[str] = []
        captured_timeout: float | None = None

        # Simulate 350 frames extracted by FFmpeg (at 0.5 fps = 700 seconds duration)
        # Batch 0: frames 0..149 (0.0s..298.0s), 150 frames
        # Batch 1: frames 150..299 (300.0s..598.0s), 150 frames
        # Batch 2: frames 300..349 (600.0s..698.0s), 50 frames
        total_test_frames = 350

        async def mock_runner(cmd: list[str], timeout: float | None) -> tuple[int, str, str]:
            nonlocal captured_cmd, captured_timeout
            captured_cmd = list(cmd)
            captured_timeout = timeout

            staging_dir = output_dir / "_staging"
            staging_dir.mkdir(parents=True, exist_ok=True)
            for i in range(1, total_test_frames + 1):
                frame_file = staging_dir / f"frame_{i:06d}.jpg"
                frame_file.write_bytes(MINIMAL_VALID_JPEG_BYTES)

            return (0, "", "FFmpeg 6.0: frame extraction complete")

        config = VideoConfig(
            fps=0.5,
            batch_duration=300.0,
            image_format="jpg",
            jpeg_quality=2,
            timeout=500.0,
        )
        processor = FFmpegVideoProcessor(config=config, runner=mock_runner)

        result = await processor.extract_frames(
            video_path,
            output_dir,
            fps=0.5,
            batch_duration=300.0,
            timeout=180.0,
        )

        # 1. Verify exact CLI arguments
        assert captured_cmd[0] == "/usr/bin/ffmpeg"
        assert "-y" in captured_cmd
        assert "-nostdin" in captured_cmd
        assert captured_cmd[captured_cmd.index("-i") + 1] == str(video_path)
        assert "-an" in captured_cmd
        assert "-sn" in captured_cmd
        assert "-dn" in captured_cmd
        assert captured_cmd[captured_cmd.index("-vf") + 1] == "fps=0.5"
        assert captured_cmd[captured_cmd.index("-q:v") + 1] == "2"
        assert "frame_%06d.jpg" in captured_cmd[-1]
        assert captured_timeout == 180.0

        # 2. Verify staging directory was cleaned up
        assert not (output_dir / "_staging").exists()

        # 3. Verify total metrics
        assert result.total_frames == 350
        assert result.fps == 0.5
        assert len(result.batches) == 3
        assert result.total_duration == 700.0

        # 4. Verify Batch 0 (00:00:00 to 00:04:58)
        batch0 = result.batches[0]
        assert batch0.index == 0
        assert batch0.frame_count == 150
        assert batch0.time_range.start == 0.0
        assert batch0.time_range.end == 300.0
        assert batch0.batch_dir == output_dir / "batch_0000"
        assert batch0.batch_dir.is_dir()

        f0_first = batch0.frames[0]
        assert f0_first.index == 0
        assert f0_first.timestamp == 0.0
        assert f0_first.timestamp_formatted == "00:00:00"
        assert f0_first.path == batch0.batch_dir / "frame_000001.jpg"
        assert f0_first.path.is_file()

        f0_last = batch0.frames[-1]
        assert f0_last.index == 149
        assert f0_last.timestamp == 298.0
        assert f0_last.timestamp_formatted == "00:04:58"
        assert f0_last.path == batch0.batch_dir / "frame_000150.jpg"
        assert f0_last.path.is_file()

        # 5. Verify Batch 1 (00:05:00 to 00:09:58)
        batch1 = result.batches[1]
        assert batch1.index == 1
        assert batch1.frame_count == 150
        assert batch1.time_range.start == 300.0
        assert batch1.time_range.end == 600.0
        assert batch1.batch_dir == output_dir / "batch_0001"
        assert batch1.batch_dir.is_dir()

        f1_first = batch1.frames[0]
        assert f1_first.index == 150
        assert f1_first.timestamp == 300.0
        assert f1_first.timestamp_formatted == "00:05:00"
        assert f1_first.path == batch1.batch_dir / "frame_000151.jpg"
        assert f1_first.path.is_file()

        f1_last = batch1.frames[-1]
        assert f1_last.index == 299
        assert f1_last.timestamp == 598.0
        assert f1_last.timestamp_formatted == "00:09:58"
        assert f1_last.path == batch1.batch_dir / "frame_000300.jpg"
        assert f1_last.path.is_file()

        # 6. Verify Batch 2 (00:10:00 to 00:11:38)
        batch2 = result.batches[2]
        assert batch2.index == 2
        assert batch2.frame_count == 50
        assert batch2.time_range.start == 600.0
        assert batch2.time_range.end == 700.0
        assert batch2.batch_dir == output_dir / "batch_0002"
        assert batch2.batch_dir.is_dir()

        f2_first = batch2.frames[0]
        assert f2_first.index == 300
        assert f2_first.timestamp == 600.0
        assert f2_first.timestamp_formatted == "00:10:00"
        assert f2_first.path == batch2.batch_dir / "frame_000301.jpg"
        assert f2_first.path.is_file()

        f2_last = batch2.frames[-1]
        assert f2_last.index == 349
        assert f2_last.timestamp == 698.0
        assert f2_last.timestamp_formatted == "00:11:38"
        assert f2_last.path == batch2.batch_dir / "frame_000350.jpg"
        assert f2_last.path.is_file()

    @pytest.mark.asyncio
    async def test_audio_only_video_raises_no_video_stream_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify NoVideoStreamError raised when file contains no video track."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/ffmpeg")
        video_path = tmp_path / "audio_only.mp4"
        video_path.touch()

        async def mock_no_video_runner(
            _cmd: list[str], _timeout: float | None
        ) -> tuple[int, str, str]:
            stderr = "Output file #0 does not contain any video stream"
            return (1, "", stderr)

        processor = FFmpegVideoProcessor(runner=mock_no_video_runner)

        with pytest.raises(NoVideoStreamError, match="does not contain a valid video stream"):
            await processor.extract_frames(video_path, tmp_path / "out")

    @pytest.mark.asyncio
    async def test_ffmpeg_execution_failure_raises_execution_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify non-zero exit code raises FFmpegVideoExecutionError with diagnostics."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/ffmpeg")
        video_path = tmp_path / "corrupt.mp4"
        video_path.touch()

        async def mock_failed_runner(
            _cmd: list[str], _timeout: float | None
        ) -> tuple[int, str, str]:
            return (137, "stdout info", "Fatal: out of memory decoding frame")

        processor = FFmpegVideoProcessor(runner=mock_failed_runner)

        with pytest.raises(FFmpegVideoExecutionError) as exc_info:
            await processor.extract_frames(video_path, tmp_path / "out")

        err = exc_info.value
        assert err.returncode == 137
        assert "Fatal: out of memory" in err.stderr
        assert err.stdout == "stdout info"

    @pytest.mark.asyncio
    async def test_timeout_raises_ffmpeg_timeout_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify timeout raises FFmpegVideoTimeoutError."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/ffmpeg")
        video_path = tmp_path / "heavy_lecture.mp4"
        video_path.touch()

        async def mock_timeout_runner(
            _cmd: list[str], _timeout: float | None
        ) -> tuple[int, str, str]:
            raise TimeoutError("Execution timed out")

        processor = FFmpegVideoProcessor(runner=mock_timeout_runner)

        with pytest.raises(FFmpegVideoTimeoutError, match="exceeded execution timeout"):
            await processor.extract_frames(video_path, tmp_path / "out", timeout=10.0)

    @pytest.mark.asyncio
    async def test_no_output_frames_produced_raises_processor_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify VideoProcessorError raised if FFmpeg exits with 0 but outputs 0 frames."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/ffmpeg")
        video_path = tmp_path / "empty.mp4"
        video_path.touch()

        async def mock_empty_runner(
            _cmd: list[str], _timeout: float | None
        ) -> tuple[int, str, str]:
            return (0, "", "Completed without frames")

        processor = FFmpegVideoProcessor(runner=mock_empty_runner)

        with pytest.raises(VideoProcessorError, match="produced no frame files"):
            await processor.extract_frames(video_path, tmp_path / "out")
