"""Global test configuration and shared fixtures."""

from datetime import date
from pathlib import Path

import pytest

import nestor
from nestor.domain.contracts.dtc import (
    CourseMetadata,
    DTCBlock,
    DTCBlockMetadata,
    LectureDTC,
    TimeRange,
)


@pytest.fixture
def package_version() -> str:
    """Fixture returning the canonical package version string."""
    return nestor.__version__


@pytest.fixture
def sample_course_metadata() -> CourseMetadata:
    """Fixture providing valid sample CourseMetadata."""
    return CourseMetadata(
        course_id="MIPT-MATH-AN-101",
        course_title="Mathematical Analysis I",
        lecture_number=1,
        lecture_title="Limits and Continuous Functions",
        lecturer="Prof. Petrov",
        recorded_date=date(2026, 9, 1),
        language="ru",
    )


@pytest.fixture
def synthetic_lecture(sample_course_metadata: CourseMetadata) -> LectureDTC:
    """Fixture providing a canonical multimodal synthetic lecture."""
    blocks = [
        DTCBlock(
            index=0,
            time_range=TimeRange(start=0.0, end=45.0),
            board_content=r"\lim_{x \to 0} \frac{\sin x}{x} = 1",
            lector_speech="Сегодня мы докажем замечательный предел для синуса.",
            metadata=DTCBlockMetadata(
                confidence=0.98,
                frame_snapshot_path=Path("snapshots/frame_000.png"),
                has_formula=True,
            ),
        ),
        DTCBlock(
            index=1,
            time_range=TimeRange(start=45.0, end=120.5),
            board_content="",
            lector_speech="Рассмотрим единичную окружность и сравним площади геометрических секторов.",
            metadata=DTCBlockMetadata(
                confidence=0.92,
                frame_snapshot_path=None,
                has_formula=False,
            ),
        ),
        DTCBlock(
            index=2,
            time_range=TimeRange(start=120.5, end=210.0),
            board_content=r"\cos x < \frac{\sin x}{x} < 1",
            lector_speech="Используя теорему о зажатой функции, получаем искомый предел.",  # noqa: RUF001
            metadata=DTCBlockMetadata(
                confidence=0.95,
                frame_snapshot_path=Path("snapshots/frame_002.png"),
                has_formula=True,
            ),
        ),
    ]
    return LectureDTC(
        course_metadata=sample_course_metadata,
        blocks=blocks,
    )
