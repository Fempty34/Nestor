"""Data by Time Codes (DTC) domain contracts.

Defines the core data models for lecture content,
including time-coded blocks, visual and speech streams, and lecture metadata.
"""

from datetime import date
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TimeRange(BaseModel):
    """Represents a validated time interval within a media file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: Annotated[
        float,
        Field(ge=0.0, description="Start offset in seconds from the beginning of the recording."),
    ]
    end: Annotated[
        float,
        Field(gt=0.0, description="End offset in seconds from the beginning of the recording."),
    ]

    @model_validator(mode="after")
    def validate_chronology(self) -> Self:
        if self.start >= self.end:
            msg = f"start ({self.start}s) must be strictly less than end ({self.end}s)"
            raise ValueError(msg)
        return self

    @property
    def duration(self) -> float:
        """Calculate interval duration in seconds."""
        return round(self.end - self.start, 3)

    @property
    def formatted_range(self) -> str:
        """Return human-readable timecode formatted as '00:MM:SS - 00:MM:SS'."""
        return f"{self._format_timestamp(self.start)} - {self._format_timestamp(self.end)}"

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        hours = int(seconds) // 3600
        minutes = (int(seconds) % 3600) // 60
        secs = int(seconds) % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class DTCBlockMetadata(BaseModel):
    """Quality and source indicators for an individual time-coded block."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    confidence: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Aggregated extraction confidence score (0.0 to 1.0).",
        ),
    ]
    frame_snapshot_path: Annotated[
        Path | None,
        Field(
            description="Optional path to the keyframe image (e.g. captured on low confidence).",
        ),
    ] = None
    has_formula: Annotated[
        bool,
        Field(
            description="Flag indicating detection of mathematical formulas in visual stream.",
        ),
    ] = False


class DTCBlock(BaseModel):
    """Atomic multimodal information block synchronizing visual and audio streams."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: Annotated[
        int,
        Field(ge=0, description="0-based monotonically increasing block sequence number."),
    ]
    time_range: Annotated[
        TimeRange,
        Field(description="Exact temporal interval for this block."),
    ]
    board_content: Annotated[
        str,
        Field(
            description="Raw text extracted from the whiteboard.",
        ),
    ] = ""
    lector_speech: Annotated[
        str,
        Field(
            description="Speech transcript corresponding to this temporal window.",
        ),
    ] = ""
    metadata: Annotated[
        DTCBlockMetadata,
        Field(description="Extraction metrics."),
    ]

    @model_validator(mode="after")
    def validate_content_presence(self) -> Self:
        """Ensure the block is not empty."""
        if not self.board_content.strip() and not self.lector_speech.strip():
            msg = f"Block index {self.index} must contain either board content or speech."
            raise ValueError(msg)
        return self


class CourseMetadata(BaseModel):
    """Bureaucratic subject information."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    course_id: Annotated[str, Field(min_length=1, description="Unique course identifier.")]
    course_title: Annotated[str, Field(min_length=1, description="Human-readable course title.")]
    lecture_number: Annotated[
        int | None,
        Field(ge=1, description="Sequential lecture number in the course."),
    ] = None
    lecture_title: Annotated[
        str | None,
        Field(description="Specific topic or title of this lecture."),
    ] = None
    lecturer: Annotated[
        str | None,
        Field(description="Academic lecturer name."),
    ] = None
    recorded_date: Annotated[
        date | None,
        Field(description="Original recording date."),
    ] = None
    language: Annotated[
        str,
        Field(description="ISO 639-1 language code (e.g. 'ru', 'en')."),
    ] = "ru"


class LectureDTC(BaseModel):
    """Top-level document representing the canonical lecture DTC model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    course_metadata: Annotated[
        CourseMetadata,
        Field(description="Institutional context of the lecture."),
    ]
    blocks: Annotated[
        list[DTCBlock],
        Field(min_length=1, description="Ordered sequence of synchronized multimodal blocks."),
    ]

    @model_validator(mode="after")
    def validate_sequence_integrity(self) -> Self:
        """Enforce strict temporal monotonicity block indices."""
        for i in range(1, len(self.blocks)):
            prev_block = self.blocks[i - 1]
            curr_block = self.blocks[i]

            if curr_block.index != prev_block.index + 1:
                msg = (
                    f"Non-contiguous block indexing at position {i}: "
                    f"expected {prev_block.index + 1}, got {curr_block.index}."
                )
                raise ValueError(msg)

            if curr_block.time_range.start < prev_block.time_range.start:
                msg = (
                    f"Temporal regression at block {curr_block.index}: "
                    f"start time {curr_block.time_range.start}s precedes "
                    f"previous block start {prev_block.time_range.start}s."
                )
                raise ValueError(msg)

        return self


type DTCLecture = LectureDTC
type DTCSequence = LectureDTC
