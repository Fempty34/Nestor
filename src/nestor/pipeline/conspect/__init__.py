"""Conspect generation contour: transforms LectureDTC into verified LaTeX and PDF."""

from nestor.pipeline.conspect.graph import (
    DEFAULT_MAX_DRAFT_ATTEMPTS,
    DEFAULT_MAX_PATCH_ATTEMPTS,
    NODE_COMPILER,
    NODE_CRITIC,
    NODE_DRAFTING,
    NODE_PATCHING,
    ConspectNode,
    build_conspect_graph,
    create_memory_checkpointer,
    route_after_compiler,
    route_after_critic,
)
from nestor.pipeline.conspect.state import (
    CHECKPOINT_ALLOWED_TYPES,
    ConspectState,
    ConspectStatus,
    CritiqueIssue,
    CritiqueReport,
    CritiqueSeverity,
    create_initial_conspect_state,
)

__all__ = [
    "CHECKPOINT_ALLOWED_TYPES",
    "DEFAULT_MAX_DRAFT_ATTEMPTS",
    "DEFAULT_MAX_PATCH_ATTEMPTS",
    "NODE_COMPILER",
    "NODE_CRITIC",
    "NODE_DRAFTING",
    "NODE_PATCHING",
    "ConspectNode",
    "ConspectState",
    "ConspectStatus",
    "CritiqueIssue",
    "CritiqueReport",
    "CritiqueSeverity",
    "build_conspect_graph",
    "create_initial_conspect_state",
    "create_memory_checkpointer",
    "route_after_compiler",
    "route_after_critic",
]
