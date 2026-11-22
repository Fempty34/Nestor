"""Domain contracts package: immutable cross-boundary data transfer objects."""

from nestor.domain.contracts.dtc import (
    CourseMetadata,
    DTCBlock,
    DTCBlockMetadata,
    DTCLecture,
    DTCSequence,
    LectureDTC,
    TimeRange,
)

__all__ = [
    "CourseMetadata",
    "DTCBlock",
    "DTCBlockMetadata",
    "DTCLecture",
    "DTCSequence",
    "LectureDTC",
    "TimeRange",
]
