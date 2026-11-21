"""Unit tests for Data by Time Codes (DTC) domain contracts."""

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from nestor.domain.contracts.dtc import (
    CourseMetadata,
    DTCBlock,
    DTCBlockMetadata,
    LectureDTC,
    TimeRange,
)


class TestTimeRange:
    """Validation and boundary tests for the TimeRange model."""

    def test_valid_time_range_and_properties(self) -> None:
        """Verify standard properties, duration calculation, and formatted ranges."""
        tr = TimeRange(start=10.25, end=75.5)
        assert tr.start == 10.25
        assert tr.end == 75.5
        assert tr.duration == 65.25
        assert tr.formatted_range == "00:00:10 - 00:01:15"

    def test_formatted_range_multi_hour(self) -> None:
        """Verify timecode formatting across hours, minutes, and seconds boundaries."""
        tr = TimeRange(start=3665.0, end=7325.0)
        assert tr.formatted_range == "01:01:05 - 02:02:05"

    @pytest.mark.parametrize(
        ("start", "end"),
        [
            (10.0, 10.0),  # equal timestamps
            (25.5, 10.0),  # start greater than end
            (100.0, 0.5),  # inversion with valid positive end
        ],
    )
    def test_invalid_chronology_raises_error(self, start: float, end: float) -> None:
        """Ensure chronological order (start < end) is strictly enforced."""
        with pytest.raises(ValidationError, match="must be strictly less than end"):
            TimeRange(start=start, end=end)

    def test_negative_start_raises_error(self) -> None:
        """Ensure start time cannot be negative."""
        with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
            TimeRange(start=-0.1, end=10.0)

    @pytest.mark.parametrize("end", [0.0, -5.0])
    def test_non_positive_end_raises_error(self, end: float) -> None:
        """Ensure end time must be strictly positive."""
        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            TimeRange(start=0.0, end=end)

    @pytest.mark.parametrize("invalid_val", ["not-a-number", None, []])
    def test_invalid_timestamp_types_raise_error(self, invalid_val: object) -> None:
        """Ensure non-numeric types fail validation during parsing."""
        with pytest.raises(ValidationError):
            TimeRange.model_validate({"start": invalid_val, "end": 10.0})

    def test_immutability(self) -> None:
        """Verify that TimeRange instances are frozen against mutation."""
        tr = TimeRange(start=0.0, end=10.0)
        with pytest.raises(ValidationError, match="Instance is frozen"):
            tr.start = 5.0  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Verify that extra unexpected attributes are rejected."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            TimeRange.model_validate({"start": 0.0, "end": 10.0, "unexpected": "value"})


class TestDTCBlockMetadata:
    """Validation and boundary tests for DTCBlockMetadata."""

    def test_valid_metadata_initialization(self) -> None:
        """Verify valid block metadata construction with optional frame snapshot path."""
        meta = DTCBlockMetadata(
            confidence=0.95,
            frame_snapshot_path=Path("frames/001.png"),
            has_formula=True,
        )
        assert meta.confidence == 0.95
        assert meta.frame_snapshot_path == Path("frames/001.png")
        assert meta.has_formula is True

    def test_snapshot_path_coercion_from_string(self) -> None:
        """Verify that string paths are automatically coerced to pathlib.Path."""
        meta = DTCBlockMetadata.model_validate(
            {"confidence": 0.8, "frame_snapshot_path": "frames/snapshot.jpg", "has_formula": False}
        )
        assert isinstance(meta.frame_snapshot_path, Path)
        assert meta.frame_snapshot_path == Path("frames/snapshot.jpg")

    @pytest.mark.parametrize("valid_conf", [0.0, 0.5, 1.0])
    def test_confidence_boundary_success(self, valid_conf: float) -> None:
        """Verify confidence scores on exact allowed boundaries."""
        meta = DTCBlockMetadata(confidence=valid_conf)
        assert meta.confidence == valid_conf

    @pytest.mark.parametrize("invalid_conf", [-0.01, 1.01, 100.0])
    def test_confidence_out_of_bounds_raises_error(self, invalid_conf: float) -> None:
        """Verify confidence scores outside [0.0, 1.0] are rejected."""
        with pytest.raises(ValidationError):
            DTCBlockMetadata(confidence=invalid_conf)

    def test_immutability_and_extra_fields(self) -> None:
        """Verify immutability and strict schema conformance."""
        meta = DTCBlockMetadata(confidence=0.5)
        with pytest.raises(ValidationError, match="Instance is frozen"):
            meta.confidence = 0.9  # type: ignore[misc]
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            DTCBlockMetadata.model_validate({"confidence": 0.5, "extra": "forbidden"})


class TestDTCBlock:
    """Validation tests for the atomic multimodal DTCBlock model."""

    @pytest.fixture
    def default_time_range(self) -> TimeRange:
        return TimeRange(start=0.0, end=10.0)

    @pytest.fixture
    def default_metadata(self) -> DTCBlockMetadata:
        return DTCBlockMetadata(confidence=1.0)

    def test_valid_board_only_content(
        self, default_time_range: TimeRange, default_metadata: DTCBlockMetadata
    ) -> None:
        """Verify a block containing only visual board content is valid."""
        block = DTCBlock(
            index=0,
            time_range=default_time_range,
            board_content="E = mc^2",
            lector_speech="",
            metadata=default_metadata,
        )
        assert block.board_content == "E = mc^2"
        assert block.lector_speech == ""

    def test_valid_speech_only_content(
        self, default_time_range: TimeRange, default_metadata: DTCBlockMetadata
    ) -> None:
        """Verify a block containing only speech transcript is valid."""
        block = DTCBlock(
            index=0,
            time_range=default_time_range,
            board_content="",
            lector_speech="Welcome to quantum electrodynamics.",
            metadata=default_metadata,
        )
        assert block.board_content == ""
        assert block.lector_speech == "Welcome to quantum electrodynamics."

    @pytest.mark.parametrize(
        ("board", "speech"),
        [
            ("", ""),
            ("   ", "   "),
            ("\n\t  ", "\r\n "),
        ],
    )
    def test_empty_content_validation_raises_error(
        self,
        board: str,
        speech: str,
        default_time_range: TimeRange,
        default_metadata: DTCBlockMetadata,
    ) -> None:
        """Ensure empty or whitespace-only blocks are rejected by validator."""
        with pytest.raises(ValidationError, match="must contain either board content or speech"):
            DTCBlock(
                index=0,
                time_range=default_time_range,
                board_content=board,
                lector_speech=speech,
                metadata=default_metadata,
            )

    def test_negative_index_raises_error(
        self, default_time_range: TimeRange, default_metadata: DTCBlockMetadata
    ) -> None:
        """Ensure block index cannot be negative."""
        with pytest.raises(ValidationError, match="Input should be greater than or equal to 0"):
            DTCBlock(
                index=-1,
                time_range=default_time_range,
                board_content="content",
                metadata=default_metadata,
            )

    def test_immutability(
        self, default_time_range: TimeRange, default_metadata: DTCBlockMetadata
    ) -> None:
        """Verify DTCBlock instances are frozen against mutation."""
        block = DTCBlock(
            index=0,
            time_range=default_time_range,
            board_content="Initial text",
            metadata=default_metadata,
        )
        with pytest.raises(ValidationError, match="Instance is frozen"):
            block.board_content = "Modified text"  # type: ignore[misc]


class TestCourseMetadata:
    """Validation tests for bureaucratic CourseMetadata."""

    def test_valid_course_metadata(self) -> None:
        """Verify standard CourseMetadata with defaults and optional fields."""
        cm = CourseMetadata(
            course_id="CS-501",
            course_title="Distributed Systems",
            lecture_number=5,
            lecturer="Leslie Lamport",
            recorded_date=date(2026, 10, 15),
        )
        assert cm.course_id == "CS-501"
        assert cm.course_title == "Distributed Systems"
        assert cm.lecture_number == 5
        assert cm.language == "ru"  # default value
        assert cm.lecture_title is None

    @pytest.mark.parametrize("empty_id", ["", "   "[:0]])
    def test_empty_course_id_raises_error(self, empty_id: str) -> None:
        """Ensure empty course_id fails min_length validation."""
        with pytest.raises(ValidationError):
            CourseMetadata(course_id=empty_id, course_title="Algorithms")

    @pytest.mark.parametrize("invalid_num", [0, -1, -10])
    def test_invalid_lecture_number_raises_error(self, invalid_num: int) -> None:
        """Ensure lecture_number must be >= 1 when provided."""
        with pytest.raises(ValidationError, match="Input should be greater than or equal to 1"):
            CourseMetadata(
                course_id="CS-101",
                course_title="Introduction",
                lecture_number=invalid_num,
            )

    def test_date_string_deserialization(self) -> None:
        """Ensure date strings in ISO format deserialize cleanly into date objects."""
        data = {
            "course_id": "PHYS-201",
            "course_title": "Thermodynamics",
            "recorded_date": "2026-11-21",
        }
        cm = CourseMetadata.model_validate(data)
        assert cm.recorded_date == date(2026, 11, 21)


class TestLectureDTC:
    """Integrity and sequence validation tests for the top-level LectureDTC document."""

    def test_synthetic_lecture_passes_validation(self, synthetic_lecture: LectureDTC) -> None:
        """Verify that the synthetic lecture fixture satisfies all domain constraints."""
        assert len(synthetic_lecture.blocks) == 3
        assert synthetic_lecture.course_metadata.course_id == "MIPT-MATH-AN-101"
        assert synthetic_lecture.blocks[0].index == 0
        assert synthetic_lecture.blocks[1].index == 1
        assert synthetic_lecture.blocks[2].index == 2

    def test_empty_blocks_list_raises_error(self, sample_course_metadata: CourseMetadata) -> None:
        """Ensure lecture without blocks is rejected (min_length=1)."""
        with pytest.raises(ValidationError):
            LectureDTC(course_metadata=sample_course_metadata, blocks=[])

    def test_non_contiguous_block_indices_raises_error(
        self, sample_course_metadata: CourseMetadata
    ) -> None:
        """Ensure block sequence numbers must strictly increment by 1."""
        b0 = DTCBlock(
            index=0,
            time_range=TimeRange(start=0.0, end=10.0),
            board_content="Content 0",
            metadata=DTCBlockMetadata(confidence=1.0),
        )
        b2 = DTCBlock(
            index=2,  # Skipped index 1
            time_range=TimeRange(start=10.0, end=20.0),
            board_content="Content 2",
            metadata=DTCBlockMetadata(confidence=1.0),
        )
        with pytest.raises(
            ValidationError, match="Non-contiguous block indexing at position 1: expected 1, got 2"
        ):
            LectureDTC(course_metadata=sample_course_metadata, blocks=[b0, b2])

    def test_temporal_regression_raises_error(self, sample_course_metadata: CourseMetadata) -> None:
        """Ensure successive blocks cannot start prior to preceding block starts."""
        b0 = DTCBlock(
            index=0,
            time_range=TimeRange(start=30.0, end=50.0),
            board_content="Content 0",
            metadata=DTCBlockMetadata(confidence=1.0),
        )
        b1 = DTCBlock(
            index=1,
            time_range=TimeRange(start=20.0, end=60.0),  # starts before b0
            board_content="Content 1",
            metadata=DTCBlockMetadata(confidence=1.0),
        )
        with pytest.raises(
            ValidationError,
            match=r"Temporal regression at block 1: start time 20\.0s precedes previous block start 30\.0s",
        ):
            LectureDTC(course_metadata=sample_course_metadata, blocks=[b0, b1])

    def test_concurrent_starts_allowed(self, sample_course_metadata: CourseMetadata) -> None:
        """Verify that concurrent blocks starting at the exact same timestamp are valid."""
        b0 = DTCBlock(
            index=0,
            time_range=TimeRange(start=10.0, end=20.0),
            board_content="Board explanation",
            metadata=DTCBlockMetadata(confidence=1.0),
        )
        b1 = DTCBlock(
            index=1,
            time_range=TimeRange(start=10.0, end=25.0),  # identical start is non-regressive
            lector_speech="Speech audio stream",
            metadata=DTCBlockMetadata(confidence=0.9),
        )
        lecture = LectureDTC(course_metadata=sample_course_metadata, blocks=[b0, b1])
        assert len(lecture.blocks) == 2


class TestDTCSerialization:
    """Serialization and deserialization round-trip tests."""

    def test_roundtrip_dict_serialization(self, synthetic_lecture: LectureDTC) -> None:
        """Verify lossless serialization and deserialization via Python dictionaries."""
        dumped = synthetic_lecture.model_dump()
        reconstructed = LectureDTC.model_validate(dumped)

        assert reconstructed == synthetic_lecture
        assert reconstructed.course_metadata.course_id == "MIPT-MATH-AN-101"
        assert len(reconstructed.blocks) == 3

    def test_roundtrip_json_serialization(self, synthetic_lecture: LectureDTC) -> None:
        """Verify lossless serialization and deserialization via JSON string."""
        json_data = synthetic_lecture.model_dump_json()
        reconstructed = LectureDTC.model_validate_json(json_data)

        assert reconstructed == synthetic_lecture
        assert reconstructed.blocks[0].metadata.frame_snapshot_path == Path(
            "snapshots/frame_000.png"
        )
        assert reconstructed.course_metadata.recorded_date == date(2026, 9, 1)

    def test_deserialization_from_raw_payload(self) -> None:
        """Verify deserialization of raw JSON dictionary payload mimicking pipeline inputs."""
        raw_payload = {
            "course_metadata": {
                "course_id": "MIPT-ALG-202",
                "course_title": "Algorithms & Data Structures",
                "lecture_number": 3,
                "lecture_title": "Red-Black Trees",
                "lecturer": "Dr. Turing",
                "recorded_date": "2026-11-20",
                "language": "ru",
            },
            "blocks": [
                {
                    "index": 0,
                    "time_range": {"start": 0.0, "end": 60.0},
                    "board_content": "Binary Search Tree properties",
                    "lector_speech": "Today we discuss balanced trees.",
                    "metadata": {
                        "confidence": 0.99,
                        "frame_snapshot_path": "snapshots/rb_tree.png",
                        "has_formula": False,
                    },
                },
                {
                    "index": 1,
                    "time_range": {"start": 60.0, "end": 180.0},
                    "board_content": "",
                    "lector_speech": "Notice tree rotations maintain invariant.",
                    "metadata": {
                        "confidence": 0.88,
                        "frame_snapshot_path": None,
                        "has_formula": False,
                    },
                },
            ],
        }

        lecture = LectureDTC.model_validate(raw_payload)
        assert lecture.course_metadata.course_id == "MIPT-ALG-202"
        assert lecture.course_metadata.recorded_date == date(2026, 11, 20)
        assert len(lecture.blocks) == 2
        assert lecture.blocks[0].metadata.frame_snapshot_path == Path("snapshots/rb_tree.png")
        assert lecture.blocks[0].time_range.duration == 60.0
        assert lecture.blocks[1].time_range.duration == 120.0

    def test_invalid_json_payload_raises_error(self) -> None:
        """Verify that malformed JSON payloads fail validation gracefully."""
        with pytest.raises(ValidationError):
            LectureDTC.model_validate_json('{"course_metadata": {}, "blocks": []}')
