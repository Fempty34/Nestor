"""Ports layer: formal typing.Protocol definitions for driving and driven adapters."""

from nestor.ports.compiler import (
    CompilationResult,
    LatexError,
    LatexErrorKind,
    TexCompiler,
    TexCompilerError,
    TexNotFoundError,
    TexTimeoutError,
)
from nestor.ports.llm import (
    LlmAuthenticationError,
    LlmClient,
    LlmError,
    LlmProviderError,
    LlmRateLimitError,
    LlmResponse,
    LlmSchemaValidationError,
    LlmTimeoutError,
)

__all__ = [
    "CompilationResult",
    "LatexError",
    "LatexErrorKind",
    "LlmAuthenticationError",
    "LlmClient",
    "LlmError",
    "LlmProviderError",
    "LlmRateLimitError",
    "LlmResponse",
    "LlmSchemaValidationError",
    "LlmTimeoutError",
    "TexCompiler",
    "TexCompilerError",
    "TexNotFoundError",
    "TexTimeoutError",
]
