"""Unit tests for FakeTexCompiler and compiler port contracts."""

import pytest

from nestor.adapters.outbound.compiler.fake import MINIMAL_VALID_PDF_BYTES, FakeTexCompiler
from nestor.ports.compiler import (
    CompilationResult,
    LatexErrorKind,
    TexCompiler,
    TexCompilerError,
    TexNotFoundError,
    TexTimeoutError,
)


class TestTexCompilerPortContracts:
    """Verifies interface invariants, types, and exceptions of the TeX compiler port."""

    def test_protocol_conformance(self) -> None:
        """Verify FakeTexCompiler conforms to TexCompiler protocol via runtime check."""
        compiler = FakeTexCompiler()
        assert isinstance(compiler, TexCompiler)

    def test_arbitrary_object_does_not_conform_to_protocol(self) -> None:
        """Verify non-conforming objects fail runtime protocol check."""
        assert not isinstance(object(), TexCompiler)

    def test_exception_hierarchy(self) -> None:
        """Verify compiler domain exception inheritance relationships."""
        assert issubclass(TexNotFoundError, TexCompilerError)
        assert issubclass(TexTimeoutError, TexCompilerError)
        assert issubclass(TexCompilerError, Exception)

    def test_compilation_result_immutability(self) -> None:
        """Verify CompilationResult is immutable and frozen."""
        res = CompilationResult(
            success=True,
            pdf_bytes=b"dummy-pdf",
        )
        assert res.success is True
        assert res.pdf_bytes == b"dummy-pdf"
        with pytest.raises(Exception):  # noqa: B017
            setattr(res, "success", False)  # noqa: B010


class TestFakeTexCompilerCompilation:
    """Tests for in-memory FakeTexCompiler behavior."""

    @pytest.mark.asyncio
    async def test_compile_returns_default_pdf(self) -> None:
        """Verify default compilation succeeds and returns minimal valid PDF bytes."""
        compiler = FakeTexCompiler()
        result = await compiler.compile(
            r"\documentclass{article}\begin{document}Hello\end{document}"
        )

        assert result.success is True
        assert result.pdf_bytes == MINIMAL_VALID_PDF_BYTES
        assert result.errors == []
        assert compiler.call_count == 1
        assert compiler.last_call is not None
        assert "Hello" in compiler.last_call.source
        assert compiler.last_call.output_filename == "document.pdf"

    @pytest.mark.asyncio
    async def test_compile_with_custom_output_filename(self) -> None:
        """Verify custom output filename is recorded in call audit."""
        compiler = FakeTexCompiler()
        await compiler.compile(
            r"\documentclass{article}",
            output_filename="lecture_01.pdf",
        )
        assert compiler.last_call is not None
        assert compiler.last_call.output_filename == "lecture_01.pdf"

    @pytest.mark.asyncio
    async def test_compile_with_assets_and_timeout(self) -> None:
        """Verify call arguments including assets and timeout are recorded."""
        compiler = FakeTexCompiler()
        assets = {"img/graph.png": b"fake-png"}
        await compiler.compile(
            r"\input{img/graph}",
            assets=assets,
            timeout=15.0,
        )

        assert compiler.call_count == 1
        last = compiler.last_call
        assert last is not None
        assert last.assets == assets
        assert last.timeout == 15.0

    @pytest.mark.asyncio
    async def test_enqueue_syntax_error(self) -> None:
        """Verify helper enqueues realistic structured compilation failure."""
        compiler = FakeTexCompiler()
        compiler.enqueue_syntax_error(
            message="Undefined control sequence.",
            line=12,
            kind=LatexErrorKind.UNDEFINED_CONTROL_SEQUENCE,
            context=r"\nonexistent",
        )

        result = await compiler.compile(r"\nonexistent")
        assert result.success is False
        assert result.pdf_bytes is None
        assert len(result.errors) == 1
        assert result.errors[0].line == 12
        assert result.errors[0].kind == LatexErrorKind.UNDEFINED_CONTROL_SEQUENCE
        assert result.errors[0].context == r"\nonexistent"

    @pytest.mark.asyncio
    async def test_error_injection_raises_configured_exception(self) -> None:
        """Verify compiler raises configured error_to_raise exception."""
        compiler = FakeTexCompiler(error_to_raise=TexTimeoutError("Simulated timeout"))

        with pytest.raises(TexTimeoutError, match="Simulated timeout"):
            await compiler.compile(r"\documentclass{article}")

        assert compiler.call_count == 1

    @pytest.mark.asyncio
    async def test_reset_clears_queue_and_history(self) -> None:
        """Verify reset clears calls and queued results."""
        compiler = FakeTexCompiler()
        compiler.enqueue_syntax_error()
        await compiler.compile("first")

        assert compiler.call_count == 1
        compiler.reset()

        assert compiler.call_count == 0
        assert compiler.last_call is None
        assert compiler.calls == []

        res = await compiler.compile("second")
        assert res.success is True
