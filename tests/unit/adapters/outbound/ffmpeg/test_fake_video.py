"""Unit tests for FakeVideoProcessor adapter."""

from pathlib import Path

import pytest

from nestor.adapters.outbound.ffmpeg.fake_video import FakeVideoProcessor
from nestor.ports.video import (
    VideoProcessingResult,
    VideoTimeoutError,
)


class TestFakeVideoProcessor:
    """Verifies FakeVideoProcessor execution, batching, disk writing, and error injection."""

    @pytest.mark.asyncio
    async def test_default_execution_creates_batches_and_frames(self, tmp_path: Path) -> None:
        """Verify default mock execution produces 2 batches with 150 frames each."""
        processor = FakeVideoProcessor(write_disk_files=True)
        video_path = tmp_path / "lecture.mp4"
        video_path.touch()
        output_dir = tmp_path / "output_batches"

        result = await processor.extract_frames(
            video_path,
            output_dir,
            fps=0.5,
            batch_duration=300.0,
        )

        assert processor.call_count == 1
        record = processor.calls[0]
        assert record.video_path == video_path
        assert record.output_dir == output_dir
        assert record.fps == 0.5
        assert record.batch_duration == 300.0

        assert len(result.batches) == 2
        assert result.total_frames == 300
        assert result.fps == 0.5
        assert result.output_dir == output_dir

        batch0 = result.batches[0]
        assert batch0.index == 0
        assert batch0.frame_count == 150
        assert batch0.time_range.start == 0.0
        assert batch0.time_range.end == 300.0
        assert batch0.frames[0].timestamp == 0.0
        assert batch0.frames[0].timestamp_formatted == "00:00:00"
        assert batch0.frames[0].path.is_file()
        assert batch0.frames[-1].timestamp == 298.0
        assert batch0.frames[-1].timestamp_formatted == "00:04:58"

        batch1 = result.batches[1]
        assert batch1.index == 1
        assert batch1.frame_count == 150
        assert batch1.time_range.start == 300.0
        assert batch1.time_range.end == 600.0
        assert batch1.frames[0].timestamp == 300.0
        assert batch1.frames[0].timestamp_formatted == "00:05:00"
        assert batch1.frames[-1].timestamp == 598.0
        assert batch1.frames[-1].timestamp_formatted == "00:09:58"

    @pytest.mark.asyncio
    async def test_error_to_raise_injection(self, tmp_path: Path) -> None:
        """Verify configured error_to_raise is raised upon invocation."""
        processor = FakeVideoProcessor(error_to_raise=VideoTimeoutError("Simulated timeout"))
        video_path = tmp_path / "lecture.mp4"
        video_path.touch()

        with pytest.raises(VideoTimeoutError, match="Simulated timeout"):
            await processor.extract_frames(video_path, tmp_path / "out")

        assert processor.call_count == 1

    @pytest.mark.asyncio
    async def test_queued_result_override(self, tmp_path: Path) -> None:
        """Verify pre-enqueued result is returned."""
        custom_result = VideoProcessingResult(
            batches=[],
            total_frames=0,
            fps=1.0,
            output_dir=tmp_path / "custom",
            log="Custom queued result",
        )
        processor = FakeVideoProcessor(results=[custom_result])

        res = await processor.extract_frames(tmp_path / "vid.mp4", tmp_path / "custom")
        assert res.log == "Custom queued result"
        assert processor.call_count == 1

    @pytest.mark.asyncio
    async def test_reset_clears_history_and_queue(self, tmp_path: Path) -> None:
        """Verify reset clears calls and results queue."""
        processor = FakeVideoProcessor()
        video_path = tmp_path / "lecture.mp4"
        video_path.touch()

        await processor.extract_frames(video_path, tmp_path / "out")
        assert processor.call_count == 1

        processor.reset()
        assert processor.call_count == 0
        assert processor.calls == []
