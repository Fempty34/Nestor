"""Typed state definitions and domain sub-models for the conspect contour."""

import operator
from enum import StrEnum
from typing import Annotated, Any, NotRequired, Required, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from nestor.domain.contracts.dtc import (
    CourseMetadata,
    DTCBlock,
    DTCBlockMetadata,
    LectureDTC,
    TimeRange,
)
from nestor.ports.compiler import CompilationResult, LatexError, LatexErrorKind


class ConspectStatus(StrEnum):
    """Lifecycle stage of a conspect generation run."""

    PENDING = "pending"
    DRAFTING = "drafting"
    CRITIQUING = "critiquing"
    COMPILING = "compiling"
    PATCHING = "patching"
    COMPLETED = "completed"
    FAILED = "failed"


class CritiqueSeverity(StrEnum):
    """Severity level for an issue flagged by the Critic agent."""

    ERROR = "error"
    WARNING = "warning"


class CritiqueIssue(BaseModel):
    """Individual mathematical, structural, or factual defect found in a draft."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    block_index: int | None = Field(
        default=None,
        ge=0,
        description="Optional 0-based DTC block index associated with this issue.",
    )
    severity: CritiqueSeverity = Field(
        default=CritiqueSeverity.ERROR,
        description="Whether this issue blocks acceptance or is advisory.",
    )
    category: str = Field(
        min_length=1,
        description="Issue category (e.g. 'formula', 'proof', 'terminology', 'omission').",
    )
    description: str = Field(
        min_length=1,
        description="Detailed explanation of the detected discrepancy or error.",
    )
    suggestion: str | None = Field(
        default=None,
        description="Actionable guidance for the Drafting agent to resolve the issue.",
    )


class CritiqueReport(BaseModel):
    """Structured acceptance decision produced by the Critic agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accepted: bool = Field(
        description="Whether the LaTeX draft passed mathematical and factual verification.",
    )
    summary: str = Field(
        default="",
        description="Concise overall assessment of the reviewed draft.",
    )
    issues: list[CritiqueIssue] = Field(
        default_factory=list,
        description="List of identified defects or improvement suggestions.",
    )


class ConspectState(TypedDict):
    """LangGraph state schema for the DTC-to-PDF conspect generation contour."""

    lecture: Required[LectureDTC]
    status: NotRequired[ConspectStatus]
    draft_tex: NotRequired[str]
    critique: NotRequired[CritiqueReport | None]
    critique_history: NotRequired[Annotated[list[CritiqueReport], operator.add]]
    compilation_result: NotRequired[CompilationResult | None]
    pdf_bytes: NotRequired[bytes | None]
    compiler_errors: NotRequired[list[LatexError]]
    draft_attempts: NotRequired[int]
    max_draft_attempts: NotRequired[int]
    patch_attempts: NotRequired[int]
    max_patch_attempts: NotRequired[int]
    error_message: NotRequired[str | None]
    execution_trace: NotRequired[Annotated[list[str], operator.add]]


# Custom domain and enum types serialized into LangGraph checkpoints.
CHECKPOINT_ALLOWED_TYPES: tuple[type[Any], ...] = (
    LectureDTC,
    CourseMetadata,
    DTCBlock,
    DTCBlockMetadata,
    TimeRange,
    CompilationResult,
    LatexError,
    LatexErrorKind,
    ConspectStatus,
    CritiqueSeverity,
    CritiqueIssue,
    CritiqueReport,
)


def create_initial_conspect_state(
    lecture: LectureDTC,
    *,
    max_draft_attempts: int = 3,
    max_patch_attempts: int = 3,
) -> ConspectState:
    """Create a normalized initial ConspectState for a lecture run.

    Args:
        lecture: Canonical DTC lecture produced by the preprocessing contour.
        max_draft_attempts: Maximum drafting passes permitted (>= 1).
        max_patch_attempts: Maximum compiler patching passes permitted (>= 0).

    Returns:
        Fully initialized ConspectState ready for graph invocation.
    """
    if max_draft_attempts < 1:
        msg = f"max_draft_attempts must be >= 1, got {max_draft_attempts}"
        raise ValueError(msg)
    if max_patch_attempts < 0:
        msg = f"max_patch_attempts must be >= 0, got {max_patch_attempts}"
        raise ValueError(msg)

    return ConspectState(
        lecture=lecture,
        status=ConspectStatus.PENDING,
        draft_tex="",
        critique=None,
        critique_history=[],
        compilation_result=None,
        pdf_bytes=None,
        compiler_errors=[],
        draft_attempts=0,
        max_draft_attempts=max_draft_attempts,
        patch_attempts=0,
        max_patch_attempts=max_patch_attempts,
        error_message=None,
        execution_trace=[],
    )
