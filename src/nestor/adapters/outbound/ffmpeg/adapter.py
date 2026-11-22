"""FFmpeg audio processing outbound adapter."""

import asyncio
import shutil
import wave
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from nestor.core.config.audio import FFmpegConfig
from nestor.domain.contracts.dtc import TimeRange
from nestor.ports.audio import (
    AudioExecutionError,
    AudioNotFoundError,
    AudioProcessingResult,
    AudioProcessorError,
    AudioSegment,
    AudioTimeoutError,
    AudioToolNotFoundError,
    NoAudioStreamError,
)

type AsyncSubprocessRunner = Callable[
    [list[str], float | None],
    Coroutine[Any, Any, tuple[int, str, str]],
]


class FFmpegNotFoundError(AudioToolNotFoundError):
    """Raised when the FFmpeg binary is missing from PATH or configured location."""


class FFmpegExecutionError(AudioExecutionError):
    """Raised when FFmpeg process terminates with a non-zero exit code."""


class FFmpegTimeoutError(AudioTimeoutError):
    """Raised when FFmpeg execution exceeds the configured deadline."""


async def _default_async_runner(
    cmd: list[str],
    timeout: float | None = None,
) -> tuple[int, str, str]:
    """Execute command asynchronously via asyncio subprocess with strict timeout."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        msg = f"Executable not found for command: {cmd[0]}"
        raise FFmpegNotFoundError(msg) from exc

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    except BaseException:
        proc.kill()
        await proc.wait()
        raise

    return (
        proc.returncode if proc.returncode is not None else -1,
        stdout_bytes.decode("utf-8", errors="replace"),
        stderr_bytes.decode("utf-8", errors="replace"),
    )


def _read_wav_metadata(path: Path) -> tuple[float, int, int]:
    """Extract duration, sample rate, and channel count from a WAV file header."""
    try:
        with wave.open(str(path), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            n_frames = wf.getnframes()
            duration = round(n_frames / float(sample_rate), 3) if sample_rate > 0 else 0.0
            return duration, sample_rate, channels
    except Exception as exc:
        msg = f"Failed to read WAV header for file '{path}': {exc}"
        raise AudioProcessorError(msg) from exc


class FFmpegAudioProcessor:
    """Outbound adapter for audio extraction, EBU R128 normalization, and segmentation."""

    def __init__(
        self,
        config: FFmpegConfig | None = None,
        *,
        runner: AsyncSubprocessRunner | None = None,
    ) -> None:
        self.config = config or FFmpegConfig()
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
                "or specify an explicit path via FFmpegConfig."
            )
            raise FFmpegNotFoundError(msg)
        self._executable_path = resolved or binary
        return self._executable_path

    async def process_lecture_audio(
        self,
        video_path: Path,
        output_dir: Path,
        *,
        segment_duration: float = 300.0,
        timeout: float | None = None,
    ) -> AudioProcessingResult:
        """Extract audio to 16 kHz Mono WAV, normalize loudness, and slice into 5-minute segments.

        Executes a single-pass FFmpeg pipeline combining audio extraction,
        EBU R128 loudness normalization filter, downmixing, resampling,
        and temporal stream segmentation.
        """
        if not video_path.is_file():
            msg = f"Input video file not found at '{video_path}'."
            raise AudioNotFoundError(msg)

        executable = self._validate_binary_availability()
        output_dir.mkdir(parents=True, exist_ok=True)

        effective_timeout = timeout if timeout is not None else self.config.timeout
        segment_pattern = output_dir / "segment_%04d.wav"

        cmd: list[str] = [
            executable,
            "-y",
            "-nostdin",
            "-i",
            str(video_path),
            "-vn",
            "-sn",
            "-dn",
            "-af",
            self.config.loudnorm_filter_arg,
            "-ac",
            str(self.config.channels),
            "-ar",
            str(self.config.sample_rate),
            "-c:a",
            "pcm_s16le",
            "-f",
            "segment",
            "-segment_time",
            str(segment_duration),
            "-reset_timestamps",
            "1",
            str(segment_pattern),
        ]

        try:
            returncode, stdout, stderr = await self._runner(cmd, effective_timeout)
        except TimeoutError as exc:
            msg = (
                f"FFmpeg process exceeded execution timeout of {effective_timeout}s "
                f"while processing '{video_path}'."
            )
            raise FFmpegTimeoutError(msg) from exc

        if returncode != 0:
            lower_stderr = stderr.lower()
            if (
                "does not contain any stream" in lower_stderr
                or "matches no streams" in lower_stderr
                or "output file #0 does not contain any audio stream" in lower_stderr
            ):
                msg = f"Video file '{video_path}' does not contain a valid audio stream."
                raise NoAudioStreamError(msg)

            msg = f"FFmpeg failed with exit code {returncode} processing '{video_path}'."
            raise FFmpegExecutionError(
                msg,
                returncode=returncode,
                cmd=cmd,
                stderr=stderr,
                stdout=stdout,
            )

        segment_files = sorted(output_dir.glob("segment_*.wav"))
        if not segment_files:
            msg = f"FFmpeg completed but produced no audio segment files in '{output_dir}'."
            raise AudioProcessorError(msg)

        segments: list[AudioSegment] = []
        cumulative_start = 0.0

        for idx, file_path in enumerate(segment_files):
            duration, sample_rate, channels = _read_wav_metadata(file_path)
            if duration <= 0.0:
                msg = f"Generated audio segment '{file_path}' has non-positive duration ({duration}s)."
                raise AudioProcessorError(msg)

            seg_end = round(cumulative_start + duration, 3)
            time_range = TimeRange(start=cumulative_start, end=seg_end)

            segments.append(
                AudioSegment(
                    index=idx,
                    path=file_path,
                    time_range=time_range,
                    sample_rate=sample_rate,
                    channels=channels,
                )
            )
            cumulative_start = seg_end

        return AudioProcessingResult(
            segments=segments,
            total_duration=cumulative_start,
            output_dir=output_dir,
            log=stderr,
        )
