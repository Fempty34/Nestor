"""LangGraph topology and checkpointer wiring for the conspect contour."""

from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from nestor.pipeline.conspect.state import (
    CHECKPOINT_ALLOWED_TYPES,
    ConspectState,
    ConspectStatus,
    CritiqueIssue,
    CritiqueReport,
    CritiqueSeverity,
)
from nestor.ports.compiler import CompilationResult, TexCompiler

NODE_DRAFTING = "drafting"
NODE_CRITIC = "critic"
NODE_COMPILER = "compiler"
NODE_PATCHING = "patching"

DEFAULT_MAX_DRAFT_ATTEMPTS = 3
DEFAULT_MAX_PATCH_ATTEMPTS = 3

type ConspectNode = Callable[[ConspectState], Awaitable[dict[str, Any]] | dict[str, Any]]


def create_memory_checkpointer() -> MemorySaver:
    """Create an in-memory checkpointer for local development and testing.

    Registers Nestor domain models and enums in JsonPlusSerializer so checkpoint
    round-trips remain clean even when LANGGRAPH_STRICT_MSGPACK is enabled.
    """
    serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_ALLOWED_TYPES)
    return MemorySaver(serde=serde)


def route_after_critic(state: ConspectState) -> Literal["drafting", "compiler"]:
    """Route after the Critic node based on acceptance and remaining draft budget."""
    critique = state.get("critique")
    if critique is not None and critique.accepted:
        return NODE_COMPILER

    draft_attempts = state.get("draft_attempts", 0)
    max_draft_attempts = state.get("max_draft_attempts", DEFAULT_MAX_DRAFT_ATTEMPTS)
    if draft_attempts < max_draft_attempts:
        return NODE_DRAFTING

    # Best-effort fallback: proceed to compilation once the revision budget is exhausted.
    return NODE_COMPILER


def route_after_compiler(state: ConspectState) -> Literal["patching", "__end__"]:
    """Route after the Compiler node based on build status and remaining patch budget."""
    compilation_result = state.get("compilation_result")
    if compilation_result is not None and compilation_result.success:
        return END

    patch_attempts = state.get("patch_attempts", 0)
    max_patch_attempts = state.get("max_patch_attempts", DEFAULT_MAX_PATCH_ATTEMPTS)
    if patch_attempts < max_patch_attempts:
        return NODE_PATCHING

    return END


def _default_drafting_node(state: ConspectState) -> dict[str, Any]:
    """Placeholder Drafting node that produces a minimal LaTeX scaffold from DTC."""
    lecture = state["lecture"]
    title = lecture.course_metadata.lecture_title or lecture.course_metadata.course_title
    body_lines = [block.board_content or block.lector_speech for block in lecture.blocks]
    body = "\n\n".join(line for line in body_lines if line.strip())
    draft_tex = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        f"\\section*{{{title}}}\n"
        f"{body}\n"
        "\\end{document}\n"
    )
    return {
        "status": ConspectStatus.DRAFTING,
        "draft_tex": draft_tex,
        "draft_attempts": state.get("draft_attempts", 0) + 1,
        "execution_trace": [NODE_DRAFTING],
    }


def _default_critic_node(state: ConspectState) -> dict[str, Any]:
    """Placeholder Critic node that validates basic draft presence."""
    draft_tex = state.get("draft_tex", "")
    if not draft_tex.strip():
        report = CritiqueReport(
            accepted=False,
            summary="Draft LaTeX source is empty.",
            issues=[
                CritiqueIssue(
                    severity=CritiqueSeverity.ERROR,
                    category="omission",
                    description="Drafting node produced an empty LaTeX document.",
                    suggestion="Generate a non-empty LaTeX document from the DTC blocks.",
                )
            ],
        )
    else:
        report = CritiqueReport(
            accepted=True,
            summary="Placeholder critic accepted non-empty draft.",
            issues=[],
        )

    return {
        "status": ConspectStatus.CRITIQUING,
        "critique": report,
        "critique_history": [report],
        "execution_trace": [NODE_CRITIC],
    }


def _make_default_compiler_node(tex_compiler: TexCompiler | None) -> ConspectNode:
    """Build the default Compiler node backed by an optional TexCompiler port."""

    async def _default_compiler_node(state: ConspectState) -> dict[str, Any]:
        draft_tex = state.get("draft_tex", "")
        if tex_compiler is not None:
            result = await tex_compiler.compile(draft_tex)
        else:
            result = CompilationResult(
                success=bool(draft_tex.strip()),
                pdf_bytes=b"%PDF-1.4\n%%EOF" if draft_tex.strip() else None,
                errors=[],
                log="Stub compilation succeeded." if draft_tex.strip() else "Empty TeX source.",
            )

        patch_attempts = state.get("patch_attempts", 0)
        max_patch_attempts = state.get("max_patch_attempts", DEFAULT_MAX_PATCH_ATTEMPTS)
        if result.success:
            status = ConspectStatus.COMPLETED
            error_message = None
        elif patch_attempts >= max_patch_attempts:
            status = ConspectStatus.FAILED
            error_message = f"LaTeX compilation failed after {patch_attempts} patch attempt(s)."
        else:
            status = ConspectStatus.COMPILING
            error_message = None

        return {
            "status": status,
            "compilation_result": result,
            "pdf_bytes": result.pdf_bytes if result.success else None,
            "compiler_errors": list(result.errors),
            "error_message": error_message,
            "execution_trace": [NODE_COMPILER],
        }

    return _default_compiler_node


def _default_patching_node(state: ConspectState) -> dict[str, Any]:
    """Placeholder Patching node that increments patch attempts and preserves draft."""
    return {
        "status": ConspectStatus.PATCHING,
        "draft_tex": state.get("draft_tex", ""),
        "patch_attempts": state.get("patch_attempts", 0) + 1,
        "execution_trace": [NODE_PATCHING],
    }


def build_conspect_graph(
    *,
    drafting_node: ConspectNode | None = None,
    critic_node: ConspectNode | None = None,
    compiler_node: ConspectNode | None = None,
    patching_node: ConspectNode | None = None,
    tex_compiler: TexCompiler | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    use_memory_checkpointer: bool = False,
) -> CompiledStateGraph[ConspectState, Any, ConspectState, ConspectState]:
    """Construct and compile the conspect generation LangGraph topology.

    Topology:
        START -> drafting -> critic -> [drafting | compiler] -> [patching | END]
        with patching looping back into compiler.

    Args:
        drafting_node: Optional custom Drafting node implementation.
        critic_node: Optional custom Critic node implementation.
        compiler_node: Optional custom Compiler node implementation.
        patching_node: Optional custom Patching node implementation.
        tex_compiler: Optional TexCompiler port used by the default compiler node.
        checkpointer: Explicit LangGraph checkpointer instance (e.g. PostgresSaver).
        use_memory_checkpointer: When True and checkpointer is None, attaches a new
            MemorySaver configured for local development.

    Returns:
        Compiled LangGraph StateGraph ready for sync or async invocation.
    """
    if checkpointer is not None and use_memory_checkpointer:
        msg = "Pass either 'checkpointer' or 'use_memory_checkpointer=True', not both."
        raise ValueError(msg)

    resolved_checkpointer = (
        create_memory_checkpointer() if use_memory_checkpointer else checkpointer
    )
    resolved_compiler = compiler_node or _make_default_compiler_node(tex_compiler)

    builder: StateGraph[ConspectState, Any, ConspectState, ConspectState] = StateGraph(
        ConspectState
    )
    builder.add_node(NODE_DRAFTING, drafting_node or _default_drafting_node)
    builder.add_node(NODE_CRITIC, critic_node or _default_critic_node)
    builder.add_node(NODE_COMPILER, resolved_compiler)
    builder.add_node(NODE_PATCHING, patching_node or _default_patching_node)

    builder.add_edge(START, NODE_DRAFTING)
    builder.add_edge(NODE_DRAFTING, NODE_CRITIC)
    builder.add_conditional_edges(
        NODE_CRITIC,
        route_after_critic,
        {
            NODE_DRAFTING: NODE_DRAFTING,
            NODE_COMPILER: NODE_COMPILER,
        },
    )
    builder.add_conditional_edges(
        NODE_COMPILER,
        route_after_compiler,
        {
            NODE_PATCHING: NODE_PATCHING,
            END: END,
        },
    )
    builder.add_edge(NODE_PATCHING, NODE_COMPILER)

    return builder.compile(checkpointer=resolved_checkpointer)
