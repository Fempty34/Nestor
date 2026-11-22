"""Unit tests for FakeAudioProcessor mock adapter."""

import wave
from pathlib import Path

import pytest

from nestor.adapters.outbound.ffmpeg.fake import FakeAudioProcessor
from nestor.ports.audio import (
    AudioNotFoundError,
    AudioProcessingResult,
    AudioProcessor,
)


class TestFakeAudioProcessor:
    """Verifies FakeAudioProcessor conforms to AudioProcessor protocol and simulates workloads."""

    def test_protocol_conformance(self) -> None:
        """Verify FakeAudioProcessor satisfies the AudioProcessor protocol."""
        processor = FakeAudioProcessor()
        assert isinstance(processor, AudioProcessor)

    @pytest.mark.asyncio
    async def test_default_processing_generates_segments(self, tmp_path: Path) -> None:
        """Verify fake generates realistic segments, monotonic TimeRanges, and WAV files."""
        processor = FakeAudioProcessor(default_segment_count=3, default_segment_duration=300.0)
        video_path = tmp_path / "lecture.mp4"
        video_path.touch()
        out_dir = tmp_path / "segments"

        result = await processor.process_lecture_audio(video_path, out_dir)

        assert len(result.segments) == 3
        assert result.total_duration == 900.0
        assert result.output_dir == out_dir
        assert processor.call_count == 1

        # Check call record
        record = processor.calls[0]
        assert record.video_path == video_path
        assert record.output_dir == out_dir
        assert record.segment_duration == 300.0

        # Verify time ranges and disk files
        for i, seg in enumerate(result.segments):
            assert seg.index == i
            assert seg.time_range.start == i * 300.0
            assert seg.time_range.end == (i + 1) * 300.0
            assert seg.duration == 300.0
            assert seg.sample_rate == 16000
            assert seg.channels == 1
            assert seg.path.is_file()

            # Verify generated file is a valid readable WAV
            with wave.open(str(seg.path), "rb") as wf:
                assert wf.getframerate() == 16000
                assert wf.getnchannels() == 1

    @pytest.mark.asyncio
    async def test_error_simulation(self, tmp_path: Path) -> None:
        """Verify error_to_raise is propagated."""
        processor = FakeAudioProcessor(error_to_raise=AudioNotFoundError("Simulated missing file"))
        video_path = tmp_path / "missing.mp4"
        out_dir = tmp_path / "out"

        with pytest.raises(AudioNotFoundError, match="Simulated missing file"):
            await processor.process_lecture_audio(video_path, out_dir)

    @pytest.mark.asyncio
    async def test_result_queue_takes_precedence(self, tmp_path: Path) -> None:
        """Verify queued results are yielded sequentially."""
        custom_result = AudioProcessingResult(
            total_duration=42.0,
            output_dir=tmp_path,
            log="Custom queued result",
        )
        processor = FakeAudioProcessor(results=[custom_result])

        res = await processor.process_lecture_audio(tmp_path / "dummy.mp4", tmp_path)
        assert res.total_duration == 42.0
        assert res.log == "Custom queued result"

    def test_reset(self, tmp_path: Path) -> None:
        """Verify reset clears call records and queued results."""
        processor = FakeAudioProcessor()
        processor._calls.append(None)  # type: ignore[arg-type]
        assert processor.call_count == 1
        processor.reset()
        assert processor.call_count == 0
