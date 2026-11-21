"""TeX compilation port protocol, result types, and compiler exceptions.

Defines the contract for LaTeX to PDF rendering engines.
"""

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class LatexErrorKind(StrEnum):
    """Categorization of LaTeX compilation diagnostics."""

    SYNTAX = "syntax"
    UNDEFINED_CONTROL_SEQUENCE = "undefined_control_sequence"
    MISSING_DELIMITER = "missing_delimiter"
    PACKAGE_NOT_FOUND = "package_not_found"
    FATAL = "fatal"
    UNKNOWN = "unknown"


class LatexError(BaseModel):
    """Structured diagnostic error extracted from compiler log output."""

    model_config = ConfigDict(frozen=True)

    message: str = Field(description="Normalized error explanation or root cause message.")
    line: int | None = Field(default=None, description="Source line number if detected.")
    kind: LatexErrorKind = Field(
        default=LatexErrorKind.UNKNOWN, description="Classification of the compilation error."
    )
    context: str | None = Field(
        default=None, description="Nearby LaTeX code snippet or problematic token context."
    )


class CompilationResult(BaseModel):
    """Standardized response container for TeX compilation execution."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(description="Indicates whether the PDF artifact was generated.")
    pdf_bytes: bytes | None = Field(
        default=None, description="Binary content of the compiled PDF document."
    )
    errors: list[LatexError] = Field(
        default_factory=list, description="Parsed diagnostic compilation errors."
    )
    log: str = Field(default="", description="Complete raw stdout and stderr output from engine.")
    output_filename: str = Field(
        default="document.pdf", description="Filename of the targeted or generated PDF artifact."
    )


class TexCompilerError(Exception):
    """Base exception for all TeX compiler port and adapter operations."""


class TexNotFoundError(TexCompilerError):
    """Raised when the TeX compiler binary is missing from PATH or specified path."""


class TexTimeoutError(TexCompilerError):
    """Raised when TeX compilation exceeds the configured execution deadline."""


@runtime_checkable
class TexCompiler(Protocol):
    """Abstract port for compiling LaTeX source code into PDF documents."""

    async def compile(
        self,
        source: str,
        *,
        output_filename: str = "document.pdf",
        assets: dict[str, bytes] | None = None,
        timeout: float | None = None,
    ) -> CompilationResult:
        """Compile raw LaTeX string into a PDF document asynchronously."""
        ...
