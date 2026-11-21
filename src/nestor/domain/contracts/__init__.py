"""Domain contracts package: immutable cross-boundary data transfer objects."""

from nestor.domain.contracts.dtc import (
    CourseMetadata,
    DTCBlock,
    DTCBlockMetadata,
    LectureDTC,
    TimeRange,
)

__all__ = [
    "CourseMetadata",
    "DTCBlock",
    "DTCBlockMetadata",
    "LectureDTC",
    "TimeRange",
]
