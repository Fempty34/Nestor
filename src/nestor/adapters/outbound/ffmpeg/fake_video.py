"""Deterministic in-memory Video Processor adapter for testing."""

from dataclasses import dataclass
from pathlib import Path

from nestor.domain.contracts.dtc import TimeRange
from nestor.ports.video import (
    VideoBatch,
    VideoFrame,
    VideoProcessingResult,
)

MINIMAL_VALID_JPEG_BYTES = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c"
    b"\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c"
    b"\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01"
    b"\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01"
    b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08"
    b"\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
)


@dataclass(frozen=True)
class VideoCallRecord:
    """Audit record capturing all arguments passed to extract_frames."""

    video_path: Path
    output_dir: Path
    fps: float
    batch_duration: float
    timeout: float | None


class FakeVideoProcessor:
    """Deterministic mock Video Processor for testing multi-agent pipelines."""

    def __init__(
        self,
        *,
        default_batch_count: int = 2,
        default_frames_per_batch: int = 150,
        results: list[VideoProcessingResult] | None = None,
        error_to_raise: Exception | None = None,
        write_disk_files: bool = True,
    ) -> None:
        self.default_batch_count = default_batch_count
        self.default_frames_per_batch = default_frames_per_batch
        self._result_queue: list[VideoProcessingResult] = (
            list(results) if results is not None else []
        )
        self.error_to_raise = error_to_raise
        self.write_disk_files = write_disk_files
        self._calls: list[VideoCallRecord] = []

    @property
    def calls(self) -> list[VideoCallRecord]:
        """Return shallow copy of recorded call history."""
        return list(self._calls)

    @property
    def call_count(self) -> int:
        """Total number of times extract_frames was invoked."""
        return len(self._calls)

    def reset(self) -> None:
        """Clear call records and result queue."""
        self._calls.clear()
        self._result_queue.clear()

    async def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        *,
        fps: float = 0.5,
        batch_duration: float = 300.0,
        timeout: float | None = None,
    ) -> VideoProcessingResult:
        """Simulate video frame sampling, master timestamping, and batching."""
        self._calls.append(
            VideoCallRecord(
                video_path=video_path,
                output_dir=output_dir,
                fps=fps,
                batch_duration=batch_duration,
                timeout=timeout,
            )
        )

        if self.error_to_raise is not None:
            raise self.error_to_raise

        if self._result_queue:
            return self._result_queue.pop(0)

        output_dir.mkdir(parents=True, exist_ok=True)
        frame_interval = round(1.0 / fps, 6)

        batches: list[VideoBatch] = []
        global_frame_idx = 0

        for b_idx in range(self.default_batch_count):
            batch_dir = output_dir / f"batch_{b_idx:04d}"
            batch_dir.mkdir(parents=True, exist_ok=True)

            batch_frames: list[VideoFrame] = []
            for _ in range(self.default_frames_per_batch):
                frame_path = batch_dir / f"frame_{global_frame_idx + 1:06d}.jpg"
                if self.write_disk_files:
                    frame_path.write_bytes(MINIMAL_VALID_JPEG_BYTES)

                ts = round(global_frame_idx * frame_interval, 3)
                batch_frames.append(
                    VideoFrame(
                        index=global_frame_idx,
                        path=frame_path,
                        timestamp=ts,
                    )
                )
                global_frame_idx += 1

            batch_start = round(b_idx * batch_duration, 3)
            batch_end = round(
                (b_idx + 1) * batch_duration,
                3,
            )
            time_range = TimeRange(start=batch_start, end=batch_end)
            batches.append(
                VideoBatch(
                    index=b_idx,
                    time_range=time_range,
                    frames=batch_frames,
                    batch_dir=batch_dir,
                )
            )

        total_duration = batches[-1].time_range.end if batches else 0.0

        return VideoProcessingResult(
            batches=batches,
            total_frames=global_frame_idx,
            total_duration=total_duration,
            fps=fps,
            output_dir=output_dir,
            log="Mocked FFmpeg video frame extraction successful",
        )
