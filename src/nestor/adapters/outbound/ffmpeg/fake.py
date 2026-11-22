"""Deterministic in-memory Audio Processor adapter for testing."""

import wave
from dataclasses import dataclass
from pathlib import Path

from nestor.domain.contracts.dtc import TimeRange
from nestor.ports.audio import (
    AudioProcessingResult,
    AudioSegment,
)


@dataclass(frozen=True)
class AudioCallRecord:
    """Audit record capturing all arguments passed to process_lecture_audio."""

    video_path: Path
    output_dir: Path
    segment_duration: float
    timeout: float | None


def _write_dummy_wav(path: Path, duration_seconds: float = 1.0, sample_rate: int = 16000) -> None:
    """Write a minimal valid PCM WAV file for test fixtures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        n_frames = int(sample_rate * duration_seconds)
        wf.writeframes(b"\x00\x00" * n_frames)


class FakeAudioProcessor:
    """Deterministic mock Audio Processor for testing multi-agent pipelines."""

    def __init__(
        self,
        *,
        default_segment_count: int = 2,
        default_segment_duration: float = 300.0,
        results: list[AudioProcessingResult] | None = None,
        error_to_raise: Exception | None = None,
        write_disk_files: bool = True,
    ) -> None:
        self.default_segment_count = default_segment_count
        self.default_segment_duration = default_segment_duration
        self._result_queue: list[AudioProcessingResult] = (
            list(results) if results is not None else []
        )
        self.error_to_raise = error_to_raise
        self.write_disk_files = write_disk_files
        self._calls: list[AudioCallRecord] = []

    @property
    def calls(self) -> list[AudioCallRecord]:
        """Return shallow copy of recorded call history."""
        return list(self._calls)

    @property
    def call_count(self) -> int:
        """Total number of times process_lecture_audio was invoked."""
        return len(self._calls)

    def reset(self) -> None:
        """Clear call records and result queue."""
        self._calls.clear()
        self._result_queue.clear()

    async def process_lecture_audio(
        self,
        video_path: Path,
        output_dir: Path,
        *,
        segment_duration: float = 300.0,
        timeout: float | None = None,
    ) -> AudioProcessingResult:
        """Simulate audio extraction, normalization, and segmentation."""
        self._calls.append(
            AudioCallRecord(
                video_path=video_path,
                output_dir=output_dir,
                segment_duration=segment_duration,
                timeout=timeout,
            )
        )

        if self.error_to_raise is not None:
            raise self.error_to_raise

        if self._result_queue:
            return self._result_queue.pop(0)

        output_dir.mkdir(parents=True, exist_ok=True)
        segments: list[AudioSegment] = []
        cumulative_start = 0.0

        for idx in range(self.default_segment_count):
            seg_path = output_dir / f"segment_{idx:04d}.wav"
            if self.write_disk_files:
                _write_dummy_wav(seg_path, duration_seconds=self.default_segment_duration)

            seg_end = round(cumulative_start + self.default_segment_duration, 3)
            time_range = TimeRange(start=cumulative_start, end=seg_end)
            segments.append(
                AudioSegment(
                    index=idx,
                    path=seg_path,
                    time_range=time_range,
                    sample_rate=16000,
                    channels=1,
                )
            )
            cumulative_start = seg_end

        return AudioProcessingResult(
            segments=segments,
            total_duration=cumulative_start,
            output_dir=output_dir,
            log="Mocked FFmpeg single-pass processing successful",
        )
