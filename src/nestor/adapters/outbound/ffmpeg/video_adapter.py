"""FFmpeg video frame sampling and batch segmentation outbound adapter."""

import shutil
from pathlib import Path

from nestor.adapters.outbound.ffmpeg.adapter import (
    AsyncSubprocessRunner,
    _default_async_runner,
)
from nestor.core.config.video import VideoConfig
from nestor.domain.contracts.dtc import TimeRange
from nestor.ports.video import (
    NoVideoStreamError,
    VideoBatch,
    VideoExecutionError,
    VideoFrame,
    VideoNotFoundError,
    VideoProcessingResult,
    VideoProcessorError,
    VideoTimeoutError,
    VideoToolNotFoundError,
)


class FFmpegVideoNotFoundError(VideoToolNotFoundError):
    """Raised when the FFmpeg binary is missing from PATH or configured location."""


class FFmpegVideoExecutionError(VideoExecutionError):
    """Raised when the FFmpeg video extraction process terminates with a non-zero exit code."""


class FFmpegVideoTimeoutError(VideoTimeoutError):
    """Raised when FFmpeg video extraction exceeds the configured execution deadline."""


class FFmpegVideoProcessor:
    """Outbound adapter for sampling video frames and grouping them into temporal batches."""

    def __init__(
        self,
        config: VideoConfig | None = None,
        *,
        runner: AsyncSubprocessRunner | None = None,
    ) -> None:
        self.config = config or VideoConfig()
        self._runner = runner or _default_async_runner
        self._executable_path: str | None = None

    def _validate_binary_availability(self) -> str:
        """Verify FFmpeg binary exists and return resolved executable path."""
        binary = self.config.binary_path
        resolved = shutil.which(binary)
        if resolved is None and not Path(binary).is_file():
            msg = (
                f"FFmpeg binary not found at '{binary}'. "
                "Ensure ffmpeg is installed and available in PATH, "
                "or specify an explicit path via VideoConfig."
            )
            raise FFmpegVideoNotFoundError(msg)
        self._executable_path = resolved or binary
        return self._executable_path

    async def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        *,
        fps: float = 0.5,
        batch_duration: float = 300.0,
        timeout: float | None = None,
    ) -> VideoProcessingResult:
        """Sample video frames at constant fps, assign master timestamps, and organize into batches.

        Executes a single-pass FFmpeg command filtering frames via `-vf fps=<fps>`,
        numbers them sequentially in an isolated staging directory,
        associates each frame with its exact master timestamp (HH:MM:SS),
        and groups frames into 5-minute batches matching audio chunks.
        """
        if not video_path.is_file():
            msg = f"Input video file not found at '{video_path}'."
            raise VideoNotFoundError(msg)

        if fps <= 0.0:
            msg = f"Sampling rate 'fps' must be strictly positive, got {fps}."
            raise ValueError(msg)

        if batch_duration <= 0.0:
            msg = f"Batch duration must be strictly positive, got {batch_duration}."
            raise ValueError(msg)

        executable = self._validate_binary_availability()
        output_dir.mkdir(parents=True, exist_ok=True)

        staging_dir = output_dir / "_staging"
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True, exist_ok=True)

        effective_timeout = timeout if timeout is not None else self.config.timeout
        frame_pattern = staging_dir / f"frame_%06d.{self.config.image_format}"

        cmd: list[str] = [
            executable,
            "-y",
            "-nostdin",
            "-i",
            str(video_path),
            "-an",
            "-sn",
            "-dn",
            "-vf",
            f"fps={fps}",
            "-q:v",
            str(self.config.jpeg_quality),
            str(frame_pattern),
        ]

        try:
            returncode, stdout, stderr = await self._runner(cmd, effective_timeout)
        except TimeoutError as exc:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            msg = (
                f"FFmpeg process exceeded execution timeout of {effective_timeout}s "
                f"while extracting video frames from '{video_path}'."
            )
            raise FFmpegVideoTimeoutError(msg) from exc
        except BaseException:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            raise

        if returncode != 0:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            lower_stderr = stderr.lower()
            if (
                "does not contain any stream" in lower_stderr
                or "matches no streams" in lower_stderr
                or "output file #0 does not contain any video stream" in lower_stderr
                or "stream map '0:v" in lower_stderr
                or "no video stream" in lower_stderr
            ):
                msg = f"Video file '{video_path}' does not contain a valid video stream."
                raise NoVideoStreamError(msg)

            msg = (
                f"FFmpeg failed with exit code {returncode} extracting frames from '{video_path}'."
            )
            raise FFmpegVideoExecutionError(
                msg,
                returncode=returncode,
                cmd=cmd,
                stderr=stderr,
                stdout=stdout,
            )

        extracted_files = sorted(staging_dir.glob(f"frame_*.{self.config.image_format}"))
        if not extracted_files:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            msg = f"FFmpeg completed successfully but produced no frame files in '{staging_dir}'."
            raise VideoProcessorError(msg)

        frame_interval = round(1.0 / fps, 6)

        # Group extracted files by batch index: batch_idx = int(timestamp // batch_duration)
        batches_map: dict[int, list[tuple[int, Path, float]]] = {}

        for global_idx, file_path in enumerate(extracted_files):
            timestamp = round(global_idx * frame_interval, 3)
            batch_idx = int(timestamp // batch_duration)

            if batch_idx not in batches_map:
                batches_map[batch_idx] = []
            batches_map[batch_idx].append((global_idx, file_path, timestamp))

        batches: list[VideoBatch] = []

        for b_idx in sorted(batches_map.keys()):
            items = batches_map[b_idx]
            batch_dir = output_dir / f"batch_{b_idx:04d}"
            batch_dir.mkdir(parents=True, exist_ok=True)

            batch_frames: list[VideoFrame] = []
            for g_idx, src_path, ts in items:
                dest_path = batch_dir / src_path.name
                shutil.move(str(src_path), str(dest_path))

                batch_frames.append(
                    VideoFrame(
                        index=g_idx,
                        path=dest_path,
                        timestamp=ts,
                    )
                )

            batch_start = round(b_idx * batch_duration, 3)
            last_ts = batch_frames[-1].timestamp
            batch_end = round(
                min(
                    (b_idx + 1) * batch_duration,
                    max(batch_start + frame_interval, last_ts + frame_interval),
                ),
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

        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

        total_frames = len(extracted_files)
        total_duration = batches[-1].time_range.end if batches else 0.0

        return VideoProcessingResult(
            batches=batches,
            total_frames=total_frames,
            total_duration=total_duration,
            fps=fps,
            output_dir=output_dir,
            log=stderr,
        )
