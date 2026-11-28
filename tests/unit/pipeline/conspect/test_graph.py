"""Unit tests for the conspect LangGraph topology and MemorySaver checkpointer."""

from typing import Any

import pytest
from langgraph.graph import END

from nestor.adapters.outbound.compiler.fake import MINIMAL_VALID_PDF_BYTES, FakeTexCompiler
from nestor.domain.contracts.dtc import LectureDTC
from nestor.pipeline.conspect.graph import (
    NODE_COMPILER,
    NODE_CRITIC,
    NODE_DRAFTING,
    NODE_PATCHING,
    build_conspect_graph,
    create_memory_checkpointer,
    route_after_compiler,
    route_after_critic,
)
from nestor.pipeline.conspect.state import (
    ConspectState,
    ConspectStatus,
    CritiqueIssue,
    CritiqueReport,
    create_initial_conspect_state,
)
from nestor.ports.compiler import CompilationResult


class TestConspectRouting:
    """Pure unit tests for conditional routing functions."""

    def test_route_after_critic_accepts_to_compiler(self, synthetic_lecture: LectureDTC) -> None:
        """Accepted critique routes directly to compiler."""
        state: ConspectState = {
            "lecture": synthetic_lecture,
            "critique": CritiqueReport(accepted=True, summary="OK"),
            "draft_attempts": 1,
            "max_draft_attempts": 3,
        }
        assert route_after_critic(state) == NODE_COMPILER

    def test_route_after_critic_rejects_within_budget(self, synthetic_lecture: LectureDTC) -> None:
        """Rejected critique with remaining budget routes back to drafting."""
        state: ConspectState = {
            "lecture": synthetic_lecture,
            "critique": CritiqueReport(accepted=False, summary="Fix sign"),
            "draft_attempts": 1,
            "max_draft_attempts": 3,
        }
        assert route_after_critic(state) == NODE_DRAFTING

    def test_route_after_critic_exhausted_budget_proceeds_to_compiler(
        self, synthetic_lecture: LectureDTC
    ) -> None:
        """Rejected critique at max_draft_attempts proceeds best-effort to compiler."""
        state: ConspectState = {
            "lecture": synthetic_lecture,
            "critique": CritiqueReport(accepted=False, summary="Still imperfect"),
            "draft_attempts": 3,
            "max_draft_attempts": 3,
        }
        assert route_after_critic(state) == NODE_COMPILER

    def test_route_after_compiler_success_ends(self, synthetic_lecture: LectureDTC) -> None:
        """Successful compilation terminates the graph."""
        state: ConspectState = {
            "lecture": synthetic_lecture,
            "compilation_result": CompilationResult(success=True, pdf_bytes=b"%PDF"),
            "patch_attempts": 0,
            "max_patch_attempts": 3,
        }
        assert route_after_compiler(state) == END

    def test_route_after_compiler_failure_within_budget_patches(
        self, synthetic_lecture: LectureDTC
    ) -> None:
        """Failed compilation with remaining patch budget routes to patching."""
        state: ConspectState = {
            "lecture": synthetic_lecture,
            "compilation_result": CompilationResult(success=False),
            "patch_attempts": 1,
            "max_patch_attempts": 3,
        }
        assert route_after_compiler(state) == NODE_PATCHING

    def test_route_after_compiler_exhausted_budget_ends(
        self, synthetic_lecture: LectureDTC
    ) -> None:
        """Failed compilation at max_patch_attempts terminates the graph."""
        state: ConspectState = {
            "lecture": synthetic_lecture,
            "compilation_result": CompilationResult(success=False),
            "patch_attempts": 3,
            "max_patch_attempts": 3,
        }
        assert route_after_compiler(state) == END


class TestConspectGraphTopology:
    """Tests verifying graph structure, loops, retry guards, and MemorySaver persistence."""

    def test_graph_structure_contains_expected_nodes_and_edges(self) -> None:
        """Verify the compiled graph exposes all expected nodes and edges."""
        graph = build_conspect_graph()
        drawable = graph.get_graph()

        expected_nodes = {
            "__start__",
            NODE_DRAFTING,
            NODE_CRITIC,
            NODE_COMPILER,
            NODE_PATCHING,
            "__end__",
        }
        assert expected_nodes.issubset(set(drawable.nodes.keys()))

        edge_pairs = {(edge.source, edge.target) for edge in drawable.edges}
        assert ("__start__", NODE_DRAFTING) in edge_pairs
        assert (NODE_DRAFTING, NODE_CRITIC) in edge_pairs
        assert (NODE_CRITIC, NODE_DRAFTING) in edge_pairs
        assert (NODE_CRITIC, NODE_COMPILER) in edge_pairs
        assert (NODE_COMPILER, NODE_PATCHING) in edge_pairs
        assert (NODE_COMPILER, "__end__") in edge_pairs
        assert (NODE_PATCHING, NODE_COMPILER) in edge_pairs

    def test_conflicting_checkpointer_arguments_raise(self) -> None:
        """Verify passing both checkpointer and use_memory_checkpointer=True fails fast."""
        with pytest.raises(ValueError, match="checkpointer"):
            build_conspect_graph(
                checkpointer=create_memory_checkpointer(),
                use_memory_checkpointer=True,
            )

    @pytest.mark.asyncio
    async def test_happy_path_with_memory_checkpointer(self, synthetic_lecture: LectureDTC) -> None:
        """Verify happy path execution and checkpoint persistence via MemorySaver."""
        fake_compiler = FakeTexCompiler()
        graph = build_conspect_graph(
            tex_compiler=fake_compiler,
            use_memory_checkpointer=True,
        )
        initial_state = create_initial_conspect_state(synthetic_lecture)
        config = {"configurable": {"thread_id": "lecture-happy-01"}}

        final_state = await graph.ainvoke(initial_state, config=config)

        assert final_state["status"] == ConspectStatus.COMPLETED
        assert final_state["execution_trace"] == [NODE_DRAFTING, NODE_CRITIC, NODE_COMPILER]
        assert final_state["draft_attempts"] == 1
        assert final_state["patch_attempts"] == 0
        assert final_state["pdf_bytes"] == MINIMAL_VALID_PDF_BYTES
        assert final_state["compiler_errors"] == []
        assert fake_compiler.call_count == 1

        snapshot = await graph.aget_state(config)
        assert snapshot.values["status"] == ConspectStatus.COMPLETED
        assert snapshot.values["lecture"] == synthetic_lecture
        assert snapshot.values["pdf_bytes"] == MINIMAL_VALID_PDF_BYTES

    @pytest.mark.asyncio
    async def test_critic_rejection_loops_back_to_drafting(
        self, synthetic_lecture: LectureDTC
    ) -> None:
        """Verify Critic rejection triggers Drafting revision and accumulates history."""
        critic_calls = 0

        async def revising_critic(state: ConspectState) -> dict[str, Any]:
            nonlocal critic_calls
            critic_calls += 1
            if critic_calls == 1:
                report = CritiqueReport(
                    accepted=False,
                    summary="Proof needs justification.",
                    issues=[
                        CritiqueIssue(
                            block_index=1,
                            category="proof",
                            description="Area comparison missing.",
                        )
                    ],
                )
            else:
                report = CritiqueReport(accepted=True, summary="Verified.")
            return {
                "status": ConspectStatus.CRITIQUING,
                "critique": report,
                "critique_history": [report],
                "execution_trace": [NODE_CRITIC],
            }

        fake_compiler = FakeTexCompiler()
        graph = build_conspect_graph(
            critic_node=revising_critic,
            tex_compiler=fake_compiler,
            use_memory_checkpointer=True,
        )
        initial_state = create_initial_conspect_state(synthetic_lecture, max_draft_attempts=3)
        config = {"configurable": {"thread_id": "lecture-revise-01"}}

        final_state = await graph.ainvoke(initial_state, config=config)

        assert final_state["execution_trace"] == [
            NODE_DRAFTING,
            NODE_CRITIC,
            NODE_DRAFTING,
            NODE_CRITIC,
            NODE_COMPILER,
        ]
        assert final_state["draft_attempts"] == 2
        assert len(final_state["critique_history"]) == 2
        assert final_state["critique_history"][0].accepted is False
        assert final_state["critique_history"][1].accepted is True
        assert final_state["status"] == ConspectStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_critic_respects_max_draft_attempts_guard(
        self, synthetic_lecture: LectureDTC
    ) -> None:
        """Verify persistent Critic rejections stop at max_draft_attempts."""

        def strict_critic(state: ConspectState) -> dict[str, Any]:
            report = CritiqueReport(accepted=False, summary="Never satisfied.")
            return {
                "status": ConspectStatus.CRITIQUING,
                "critique": report,
                "critique_history": [report],
                "execution_trace": [NODE_CRITIC],
            }

        graph = build_conspect_graph(
            critic_node=strict_critic,
            tex_compiler=FakeTexCompiler(),
        )
        initial_state = create_initial_conspect_state(synthetic_lecture, max_draft_attempts=2)

        final_state = await graph.ainvoke(initial_state)

        assert final_state["draft_attempts"] == 2
        assert final_state["execution_trace"] == [
            NODE_DRAFTING,
            NODE_CRITIC,
            NODE_DRAFTING,
            NODE_CRITIC,
            NODE_COMPILER,
        ]
        assert final_state["critique"] is not None
        assert final_state["critique"].accepted is False
        assert final_state["status"] == ConspectStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_compiler_failure_triggers_patching_loop(
        self, synthetic_lecture: LectureDTC
    ) -> None:
        """Verify compilation failure triggers Patching and retries compilation."""
        fake_compiler = FakeTexCompiler()
        fake_compiler.enqueue_syntax_error(message="Undefined control sequence.", line=4)
        fake_compiler.enqueue_result(
            CompilationResult(
                success=True,
                pdf_bytes=MINIMAL_VALID_PDF_BYTES,
                errors=[],
                log="Compiled on second attempt.",
            )
        )

        graph = build_conspect_graph(
            tex_compiler=fake_compiler,
            use_memory_checkpointer=True,
        )
        initial_state = create_initial_conspect_state(synthetic_lecture, max_patch_attempts=3)
        config = {"configurable": {"thread_id": "lecture-patch-01"}}

        final_state = await graph.ainvoke(initial_state, config=config)

        assert final_state["execution_trace"] == [
            NODE_DRAFTING,
            NODE_CRITIC,
            NODE_COMPILER,
            NODE_PATCHING,
            NODE_COMPILER,
        ]
        assert final_state["patch_attempts"] == 1
        assert final_state["status"] == ConspectStatus.COMPLETED
        assert final_state["pdf_bytes"] == MINIMAL_VALID_PDF_BYTES
        assert final_state["compiler_errors"] == []
        assert fake_compiler.call_count == 2

    @pytest.mark.asyncio
    async def test_compiler_failure_respects_max_patch_attempts_guard(
        self, synthetic_lecture: LectureDTC
    ) -> None:
        """Verify unfixable compilation failures terminate cleanly at max_patch_attempts."""
        fake_compiler = FakeTexCompiler()
        for _ in range(5):
            fake_compiler.enqueue_syntax_error(message="Missing $ inserted.", line=12)

        graph = build_conspect_graph(
            tex_compiler=fake_compiler,
            use_memory_checkpointer=True,
        )
        initial_state = create_initial_conspect_state(synthetic_lecture, max_patch_attempts=2)
        config = {"configurable": {"thread_id": "lecture-fail-01"}}

        final_state = await graph.ainvoke(initial_state, config=config)

        assert final_state["execution_trace"] == [
            NODE_DRAFTING,
            NODE_CRITIC,
            NODE_COMPILER,
            NODE_PATCHING,
            NODE_COMPILER,
            NODE_PATCHING,
            NODE_COMPILER,
        ]
        assert final_state["patch_attempts"] == 2
        assert final_state["status"] == ConspectStatus.FAILED
        assert final_state["pdf_bytes"] is None
        assert len(final_state["compiler_errors"]) == 1
        assert final_state["error_message"] is not None
        assert fake_compiler.call_count == 3

    @pytest.mark.asyncio
    async def test_zero_patch_attempts_disables_patching_loop(
        self, synthetic_lecture: LectureDTC
    ) -> None:
        """Verify max_patch_attempts=0 skips patching on compilation failure."""
        fake_compiler = FakeTexCompiler()
        fake_compiler.enqueue_syntax_error()

        graph = build_conspect_graph(tex_compiler=fake_compiler)
        initial_state = create_initial_conspect_state(synthetic_lecture, max_patch_attempts=0)

        final_state = await graph.ainvoke(initial_state)

        assert final_state["execution_trace"] == [NODE_DRAFTING, NODE_CRITIC, NODE_COMPILER]
        assert final_state["patch_attempts"] == 0
        assert final_state["status"] == ConspectStatus.FAILED
        assert fake_compiler.call_count == 1

    @pytest.mark.asyncio
    async def test_memory_checkpointer_isolates_threads(
        self, synthetic_lecture: LectureDTC
    ) -> None:
        """Verify MemorySaver isolates persisted state by thread_id."""
        failing_compiler = FakeTexCompiler()
        failing_compiler.enqueue_syntax_error()

        checkpointer = create_memory_checkpointer()
        graph = build_conspect_graph(
            tex_compiler=failing_compiler,
            checkpointer=checkpointer,
        )

        cfg_fail = {"configurable": {"thread_id": "thread-a"}}
        cfg_ok = {"configurable": {"thread_id": "thread-b"}}

        await graph.ainvoke(
            create_initial_conspect_state(synthetic_lecture, max_patch_attempts=0),
            config=cfg_fail,
        )
        await graph.ainvoke(
            create_initial_conspect_state(synthetic_lecture, max_patch_attempts=0),
            config=cfg_ok,
        )

        snap_fail = await graph.aget_state(cfg_fail)
        snap_ok = await graph.aget_state(cfg_ok)

        assert snap_fail.values["status"] == ConspectStatus.FAILED
        assert snap_ok.values["status"] == ConspectStatus.COMPLETED
