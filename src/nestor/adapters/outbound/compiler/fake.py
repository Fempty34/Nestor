"""Deterministic in-memory TeX compiler adapter for testing."""

from dataclasses import dataclass

from nestor.ports.compiler import (
    CompilationResult,
    LatexError,
    LatexErrorKind,
)

MINIMAL_VALID_PDF_BYTES = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<<>>\n%%EOF"


@dataclass(frozen=True)
class CompilerCallRecord:
    """Audit record capturing all arguments passed to a compile invocation."""

    source: str
    output_filename: str
    assets: dict[str, bytes] | None
    timeout: float | None


class FakeTexCompiler:
    """Deterministic mock TeX compiler for testing multi-agent pipelines."""

    def __init__(
        self,
        *,
        default_pdf: bytes = MINIMAL_VALID_PDF_BYTES,
        default_result: CompilationResult | None = None,
        results: list[CompilationResult] | None = None,
        error_to_raise: Exception | None = None,
    ) -> None:
        self.default_pdf = default_pdf
        self.default_result = default_result or CompilationResult(
            success=True,
            pdf_bytes=self.default_pdf,
            errors=[],
            log="Mocked Tectonic compilation successful",
        )
        self._result_queue: list[CompilationResult] = list(results) if results is not None else []
        self.error_to_raise = error_to_raise
        self._calls: list[CompilerCallRecord] = []

    @property
    def calls(self) -> list[CompilerCallRecord]:
        """Return shallow copy of recorded call history."""
        return list(self._calls)

    @property
    def call_count(self) -> int:
        """Return total number of compilation invocations."""
        return len(self._calls)

    @property
    def last_call(self) -> CompilerCallRecord | None:
        """Return most recent call record, or None if no calls were made."""
        return self._calls[-1] if self._calls else None

    def reset(self) -> None:
        """Clear recorded calls and reset result queue."""
        self._calls.clear()
        self._result_queue.clear()

    def enqueue_result(self, result: CompilationResult) -> None:
        """Append a canned result to the end of the FIFO queue."""
        self._result_queue.append(result)

    def enqueue_syntax_error(
        self,
        message: str = "Undefined control sequence.",
        line: int | None = 10,
        kind: LatexErrorKind = LatexErrorKind.UNDEFINED_CONTROL_SEQUENCE,
        context: str | None = "\\invalid",
        output_filename: str = "document.pdf",
    ) -> None:
        """Enqueue a realistic LaTeX compilation failure."""
        self.enqueue_result(
            CompilationResult(
                success=False,
                pdf_bytes=None,
                errors=[
                    LatexError(
                        message=message,
                        line=line,
                        kind=kind,
                        context=context,
                    )
                ],
                log=f"! {message}\nl.{line} {context}\n",
                output_filename=output_filename,
            )
        )

    async def compile(
        self,
        source: str,
        *,
        output_filename: str = "document.pdf",
        assets: dict[str, bytes] | None = None,
        timeout: float | None = None,
    ) -> CompilationResult:
        """Compile LaTeX source deterministically."""
        self._calls.append(
            CompilerCallRecord(
                source=source,
                output_filename=output_filename,
                assets=dict(assets) if assets is not None else None,
                timeout=timeout,
            )
        )

        if self.error_to_raise is not None:
            raise self.error_to_raise

        if self._result_queue:
            return self._result_queue.pop(0)

        return self.default_result
