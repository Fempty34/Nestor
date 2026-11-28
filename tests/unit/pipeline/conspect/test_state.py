"""Unit tests for ConspectState and conspect domain sub-models."""

import pytest
from pydantic import ValidationError

from nestor.domain.contracts.dtc import LectureDTC
from nestor.pipeline.conspect.state import (
    CHECKPOINT_ALLOWED_TYPES,
    ConspectStatus,
    CritiqueIssue,
    CritiqueReport,
    CritiqueSeverity,
    create_initial_conspect_state,
)
from nestor.ports.compiler import CompilationResult


class TestCreateInitialConspectState:
    """Tests for ConspectState initialization helper."""

    def test_defaults(self, synthetic_lecture: LectureDTC) -> None:
        """Verify all initial fields and default budgets are populated."""
        state = create_initial_conspect_state(synthetic_lecture)

        assert state["lecture"] == synthetic_lecture
        assert state["status"] == ConspectStatus.PENDING
        assert state["draft_tex"] == ""
        assert state["critique"] is None
        assert state["critique_history"] == []
        assert state["compilation_result"] is None
        assert state["pdf_bytes"] is None
        assert state["compiler_errors"] == []
        assert state["draft_attempts"] == 0
        assert state["max_draft_attempts"] == 3
        assert state["patch_attempts"] == 0
        assert state["max_patch_attempts"] == 3
        assert state["error_message"] is None
        assert state["execution_trace"] == []

    def test_custom_budgets(self, synthetic_lecture: LectureDTC) -> None:
        """Verify custom retry budgets including zero patch attempts are accepted."""
        state = create_initial_conspect_state(
            synthetic_lecture,
            max_draft_attempts=5,
            max_patch_attempts=0,
        )

        assert state["max_draft_attempts"] == 5
        assert state["max_patch_attempts"] == 0

    @pytest.mark.parametrize("bad_draft_attempts", [0, -1])
    def test_invalid_max_draft_attempts_raises(
        self, synthetic_lecture: LectureDTC, bad_draft_attempts: int
    ) -> None:
        """Verify max_draft_attempts < 1 is rejected."""
        with pytest.raises(ValueError, match="max_draft_attempts"):
            create_initial_conspect_state(synthetic_lecture, max_draft_attempts=bad_draft_attempts)

    def test_invalid_max_patch_attempts_raises(self, synthetic_lecture: LectureDTC) -> None:
        """Verify negative max_patch_attempts is rejected."""
        with pytest.raises(ValueError, match="max_patch_attempts"):
            create_initial_conspect_state(synthetic_lecture, max_patch_attempts=-1)


class TestCritiqueModels:
    """Tests for CritiqueIssue and CritiqueReport validation."""

    def test_critique_report_valid_and_frozen(self) -> None:
        """Verify valid CritiqueReport construction and immutability."""
        issue = CritiqueIssue(
            block_index=0,
            severity=CritiqueSeverity.ERROR,
            category="formula",
            description="Missing limit condition in proof step.",
            suggestion="Add x -> 0 qualification.",
        )
        report = CritiqueReport(
            accepted=False,
            summary="Formula condition incomplete.",
            issues=[issue],
        )

        assert report.accepted is False
        assert len(report.issues) == 1
        assert report.issues[0].category == "formula"

        with pytest.raises(ValidationError):
            report.accepted = True  # type: ignore[misc]

    def test_critique_issue_forbids_extra_fields(self) -> None:
        """Verify extra fields are forbidden on CritiqueIssue."""
        with pytest.raises(ValidationError):
            CritiqueIssue(
                category="proof",
                description="Step skipped.",
                unknown_field="unexpected",  # type: ignore[call-arg]
            )

    def test_checkpoint_allowlist_contains_core_models(self) -> None:
        """Verify checkpoint serializer allowlist registers key domain contracts."""
        assert LectureDTC in CHECKPOINT_ALLOWED_TYPES
        assert CompilationResult in CHECKPOINT_ALLOWED_TYPES
        assert CritiqueReport in CHECKPOINT_ALLOWED_TYPES
        assert ConspectStatus in CHECKPOINT_ALLOWED_TYPES
