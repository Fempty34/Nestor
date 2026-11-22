"""Unit tests for the FFmpeg audio processing outbound adapter."""

import shutil
import wave
from pathlib import Path

import pytest

from nestor.adapters.outbound.ffmpeg.adapter import (
    FFmpegAudioProcessor,
    FFmpegExecutionError,
    FFmpegNotFoundError,
    FFmpegTimeoutError,
    _default_async_runner,
    _read_wav_metadata,
)
from nestor.core.config.audio import FFmpegConfig
from nestor.ports.audio import (
    AudioExecutionError,
    AudioNotFoundError,
    AudioProcessor,
    AudioProcessorError,
    AudioTimeoutError,
    AudioToolNotFoundError,
    NoAudioStreamError,
)


def _create_test_wav(
    path: Path,
    duration_seconds: float,
    sample_rate: int = 16000,
    channels: int = 1,
) -> None:
    """Helper to create a synthetically valid WAV PCM file for fixture validation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        n_frames = int(sample_rate * duration_seconds)
        wf.writeframes(b"\x00\x00" * n_frames * channels)


class TestFFmpegAudioProcessorInitialization:
    """Verifies adapter protocol conformance and binary validation."""

    def test_protocol_conformance(self) -> None:
        """Verify FFmpegAudioProcessor satisfies abstract AudioProcessor protocol."""
        processor = FFmpegAudioProcessor()
        assert isinstance(processor, AudioProcessor)

    def test_adapter_exceptions_inherit_from_port_exceptions(self) -> None:
        """Verify adapter-specific exceptions inherit from generic port errors."""
        assert issubclass(FFmpegNotFoundError, AudioToolNotFoundError)
        assert issubclass(FFmpegExecutionError, AudioExecutionError)
        assert issubclass(FFmpegTimeoutError, AudioTimeoutError)

    def test_missing_binary_raises_not_found_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify FFmpegNotFoundError raised when binary is not in PATH or filesystem."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: None)
        processor = FFmpegAudioProcessor(config=FFmpegConfig(binary_path="nonexistent_ffmpeg_bin"))

        with pytest.raises(FFmpegNotFoundError, match="FFmpeg binary not found"):
            processor._validate_binary_availability()


class TestWavMetadataExtraction:
    """Verifies reading audio properties directly from WAV headers."""

    def test_read_valid_wav_header(self, tmp_path: Path) -> None:
        """Verify duration, sample rate, and channels extracted accurately."""
        wav_file = tmp_path / "sample.wav"
        _create_test_wav(wav_file, duration_seconds=2.5, sample_rate=16000, channels=1)

        duration, sample_rate, channels = _read_wav_metadata(wav_file)
        assert duration == 2.5
        assert sample_rate == 16000
        assert channels == 1

    def test_corrupt_wav_file_raises_processor_error(self, tmp_path: Path) -> None:
        """Verify corrupted or truncated WAV files raise AudioProcessorError."""
        corrupt_file = tmp_path / "corrupt.wav"
        corrupt_file.write_bytes(b"not a valid wav header")

        with pytest.raises(AudioProcessorError, match="Failed to read WAV header"):
            _read_wav_metadata(corrupt_file)


class TestFFmpegAudioProcessorExecution:
    """Tests execution flow, command generation, and error handling via mocked runners."""

    @pytest.mark.asyncio
    async def test_missing_input_video_raises_audio_not_found(self, tmp_path: Path) -> None:
        """Verify AudioNotFoundError raised when target video does not exist."""
        processor = FFmpegAudioProcessor()
        with pytest.raises(AudioNotFoundError, match="Input video file not found"):
            await processor.process_lecture_audio(tmp_path / "nonexistent.mp4", tmp_path / "out")

    @pytest.mark.asyncio
    async def test_successful_single_pass_pipeline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify single-pass CLI command args, loudnorm filter, and segment aggregation."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/ffmpeg")

        video_path = tmp_path / "sample_lecture.mp4"
        video_path.touch()
        output_dir = tmp_path / "output_segments"

        captured_cmd: list[str] = []
        captured_timeout: float | None = None

        async def mock_runner(cmd: list[str], timeout: float | None) -> tuple[int, str, str]:
            nonlocal captured_cmd, captured_timeout
            captured_cmd = list(cmd)
            captured_timeout = timeout

            # Simulate FFmpeg producing 2 segments (300s and 125.4s)
            output_dir.mkdir(parents=True, exist_ok=True)
            _create_test_wav(output_dir / "segment_0000.wav", duration_seconds=300.0)
            _create_test_wav(output_dir / "segment_0001.wav", duration_seconds=125.4)
            return (0, "", "FFmpeg version 6.0 ... completed successfully")

        config = FFmpegConfig(
            sample_rate=16000,
            channels=1,
            loudnorm_i=-16.0,
            loudnorm_tp=-1.5,
            loudnorm_lra=11.0,
            timeout=300.0,
        )
        processor = FFmpegAudioProcessor(config=config, runner=mock_runner)

        result = await processor.process_lecture_audio(
            video_path, output_dir, segment_duration=300.0, timeout=120.0
        )

        # 1. Verify exact CLI arguments
        assert captured_cmd[0] == "/usr/bin/ffmpeg"
        assert "-y" in captured_cmd
        assert "-nostdin" in captured_cmd
        assert captured_cmd[captured_cmd.index("-i") + 1] == str(video_path)
        assert "-vn" in captured_cmd
        assert "-sn" in captured_cmd
        assert "-dn" in captured_cmd
        assert "-af" in captured_cmd
        assert captured_cmd[captured_cmd.index("-af") + 1] == "loudnorm=I=-16.0:TP=-1.5:LRA=11.0"
        assert captured_cmd[captured_cmd.index("-ac") + 1] == "1"
        assert captured_cmd[captured_cmd.index("-ar") + 1] == "16000"
        assert captured_cmd[captured_cmd.index("-c:a") + 1] == "pcm_s16le"
        assert captured_cmd[captured_cmd.index("-f") + 1] == "segment"
        assert captured_cmd[captured_cmd.index("-segment_time") + 1] == "300.0"
        assert captured_cmd[captured_cmd.index("-reset_timestamps") + 1] == "1"
        assert captured_timeout == 120.0

        # 2. Verify segmented results and monotonic TimeRanges
        assert len(result.segments) == 2
        assert result.total_duration == 425.4
        assert result.output_dir == output_dir

        seg0 = result.segments[0]
        assert seg0.index == 0
        assert seg0.time_range.start == 0.0
        assert seg0.time_range.end == 300.0
        assert seg0.duration == 300.0
        assert seg0.sample_rate == 16000
        assert seg0.channels == 1

        seg1 = result.segments[1]
        assert seg1.index == 1
        assert seg1.time_range.start == 300.0
        assert seg1.time_range.end == 425.4
        assert seg1.duration == 125.4
        assert seg1.sample_rate == 16000
        assert seg1.channels == 1

    @pytest.mark.asyncio
    async def test_silent_video_raises_no_audio_stream_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify NoAudioStreamError is raised when video contains no audio track."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/ffmpeg")
        video_path = tmp_path / "silent.mp4"
        video_path.touch()

        async def mock_silent_runner(
            _cmd: list[str], _timeout: float | None
        ) -> tuple[int, str, str]:
            stderr = "Output file #0 does not contain any audio stream"
            return (1, "", stderr)

        processor = FFmpegAudioProcessor(runner=mock_silent_runner)

        with pytest.raises(NoAudioStreamError, match="does not contain a valid audio stream"):
            await processor.process_lecture_audio(video_path, tmp_path / "out")

    @pytest.mark.asyncio
    async def test_ffmpeg_execution_failure_raises_execution_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify generic non-zero exit code raises FFmpegExecutionError with diagnostics."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/ffmpeg")
        video_path = tmp_path / "corrupt.mp4"
        video_path.touch()

        async def mock_failed_runner(
            _cmd: list[str], _timeout: float | None
        ) -> tuple[int, str, str]:
            return (137, "partial stdout", "Fatal: segmentation fault in libavformat")

        processor = FFmpegAudioProcessor(runner=mock_failed_runner)

        with pytest.raises(FFmpegExecutionError) as exc_info:
            await processor.process_lecture_audio(video_path, tmp_path / "out")

        err = exc_info.value
        assert err.returncode == 137
        assert "Fatal: segmentation fault" in err.stderr
        assert err.stdout == "partial stdout"

    @pytest.mark.asyncio
    async def test_timeout_raises_ffmpeg_timeout_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify timeout raises FFmpegTimeoutError."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/ffmpeg")
        video_path = tmp_path / "heavy_lecture.mp4"
        video_path.touch()

        async def mock_timeout_runner(
            _cmd: list[str], _timeout: float | None
        ) -> tuple[int, str, str]:
            raise TimeoutError("Command timed out")

        processor = FFmpegAudioProcessor(runner=mock_timeout_runner)

        with pytest.raises(FFmpegTimeoutError, match="exceeded execution timeout"):
            await processor.process_lecture_audio(video_path, tmp_path / "out", timeout=5.0)

    @pytest.mark.asyncio
    async def test_no_output_segments_produced_raises_processor_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify AudioProcessorError raised if FFmpeg exits with 0 but writes no files."""
        monkeypatch.setattr(shutil, "which", lambda _cmd: "/usr/bin/ffmpeg")
        video_path = tmp_path / "empty.mp4"
        video_path.touch()

        async def mock_empty_runner(
            _cmd: list[str], _timeout: float | None
        ) -> tuple[int, str, str]:
            return (0, "", "Done without writing")

        processor = FFmpegAudioProcessor(runner=mock_empty_runner)

        with pytest.raises(AudioProcessorError, match="produced no audio segment files"):
            await processor.process_lecture_audio(video_path, tmp_path / "out")


class TestDefaultAsyncRunner:
    """Verifies the default asyncio subprocess runner behavior."""

    @pytest.mark.asyncio
    async def test_missing_binary_in_default_runner(self) -> None:
        """Verify FileNotFoundError wraps into FFmpegNotFoundError in default runner."""
        with pytest.raises(FFmpegNotFoundError, match="Executable not found"):
            await _default_async_runner(["nonexistent_binary_xyz_123"])
