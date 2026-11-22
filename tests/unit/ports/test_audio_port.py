"""Unit tests for the audio processing port contracts and exceptions."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from nestor.domain.contracts.dtc import TimeRange
from nestor.ports.audio import (
    AudioExecutionError,
    AudioNotFoundError,
    AudioProcessingResult,
    AudioProcessor,
    AudioProcessorError,
    AudioSegment,
    AudioTimeoutError,
    AudioToolNotFoundError,
    NoAudioStreamError,
)


class TestAudioSegment:
    """Verifies AudioSegment contract invariants, defaults, and immutability."""

    def test_valid_audio_segment_instantiation(self, tmp_path: Path) -> None:
        """Verify successful instantiation with valid attributes."""
        segment_file = tmp_path / "segment_0000.wav"
        segment = AudioSegment(
            index=0,
            path=segment_file,
            time_range=TimeRange(start=0.0, end=300.0),
            sample_rate=16000,
            channels=1,
        )

        assert segment.index == 0
        assert segment.path == segment_file
        assert segment.time_range.start == 0.0
        assert segment.time_range.end == 300.0
        assert segment.duration == 300.0
        assert segment.sample_rate == 16000
        assert segment.channels == 1

    def test_default_channels_and_sample_rate(self, tmp_path: Path) -> None:
        """Verify default values adhere to 16 kHz Mono standard."""
        segment = AudioSegment(
            index=1,
            path=tmp_path / "segment_0001.wav",
            time_range=TimeRange(start=300.0, end=600.0),
        )

        assert segment.sample_rate == 16000
        assert segment.channels == 1
        assert segment.duration == 300.0

    def test_duration_computed_from_time_range(self, tmp_path: Path) -> None:
        """Verify duration dynamically reflects time_range.duration."""
        segment = AudioSegment(
            index=0,
            path=tmp_path / "test.wav",
            time_range=TimeRange(start=10.5, end=35.8),
        )
        assert segment.duration == 25.3

    def test_immutability_enforced(self, tmp_path: Path) -> None:
        """Verify AudioSegment is frozen and cannot be mutated."""
        segment = AudioSegment(
            index=0,
            path=tmp_path / "test.wav",
            time_range=TimeRange(start=0.0, end=10.0),
        )

        with pytest.raises(ValidationError):
            segment.index = 1  # type: ignore[misc]

    def test_extra_fields_forbidden(self, tmp_path: Path) -> None:
        """Verify unexpected attributes or explicit duration field are rejected."""
        with pytest.raises(ValidationError):
            AudioSegment(  # type: ignore[call-arg]
                index=0,
                path=tmp_path / "test.wav",
                time_range=TimeRange(start=0.0, end=10.0),
                unexpected_field="disallowed",
            )

        with pytest.raises(ValidationError):
            AudioSegment(  # type: ignore[call-arg]
                index=0,
                path=tmp_path / "test.wav",
                time_range=TimeRange(start=0.0, end=10.0),
                duration=10.0,
            )

    def test_negative_index_rejected(self, tmp_path: Path) -> None:
        """Verify segment index must be non-negative."""
        with pytest.raises(ValidationError):
            AudioSegment(
                index=-1,
                path=tmp_path / "test.wav",
                time_range=TimeRange(start=0.0, end=10.0),
            )


class TestAudioProcessingResult:
    """Verifies AudioProcessingResult container properties."""

    def test_valid_empty_result(self, tmp_path: Path) -> None:
        """Verify result with default empty segments list."""
        result = AudioProcessingResult(
            total_duration=0.0,
            output_dir=tmp_path,
        )
        assert result.segments == []
        assert result.total_duration == 0.0
        assert result.output_dir == tmp_path
        assert result.log == ""

    def test_result_immutability(self, tmp_path: Path) -> None:
        """Verify AudioProcessingResult is frozen."""
        result = AudioProcessingResult(
            total_duration=100.0,
            output_dir=tmp_path,
        )
        with pytest.raises(ValidationError):
            result.total_duration = 200.0  # type: ignore[misc]


class TestAudioExceptions:
    """Verifies audio exception hierarchy and metadata encapsulation."""

    def test_exception_inheritance(self) -> None:
        """Verify all domain audio exceptions inherit from AudioProcessorError."""
        assert issubclass(AudioNotFoundError, AudioProcessorError)
        assert issubclass(AudioToolNotFoundError, AudioProcessorError)
        assert issubclass(AudioExecutionError, AudioProcessorError)
        assert issubclass(AudioTimeoutError, AudioProcessorError)
        assert issubclass(NoAudioStreamError, AudioProcessorError)

    def test_audio_execution_error_properties(self) -> None:
        """Verify AudioExecutionError preserves diagnostics."""
        err = AudioExecutionError(
            "Process failed",
            returncode=1,
            cmd=["processor_bin", "-i", "input.mp4"],
            stderr="Decoder error",
            stdout="Progress info",
        )
        assert err.returncode == 1
        assert err.cmd == ["processor_bin", "-i", "input.mp4"]
        assert err.stderr == "Decoder error"
        assert err.stdout == "Progress info"
        assert str(err) == "Process failed"


class TestAudioProcessorProtocol:
    """Verifies runtime protocol checkability."""

    def test_protocol_runtime_checkable(self) -> None:
        """Verify AudioProcessor is runtime_checkable."""

        class DummyProcessor:
            async def process_lecture_audio(
                self,
                video_path: Path,
                output_dir: Path,
                *,
                segment_duration: float = 300.0,
                timeout: float | None = None,
            ) -> AudioProcessingResult:
                return AudioProcessingResult(total_duration=0.0, output_dir=output_dir)

        assert isinstance(DummyProcessor(), AudioProcessor)
