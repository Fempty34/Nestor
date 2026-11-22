"""Ports layer: formal typing.Protocol definitions for driving and driven adapters."""

from nestor.ports.audio import (
    AudioExecutionError,
    AudioNotFoundError,
    AudioProcessingResult,
    AudioProcessor,
    AudioProcessorError,
    AudioSegment,
    AudioTimeoutError,
    AudioToolNotFoundError,
    NoAudioStreamError,
)
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
    "AudioExecutionError",
    "AudioNotFoundError",
    "AudioProcessingResult",
    "AudioProcessor",
    "AudioProcessorError",
    "AudioSegment",
    "AudioTimeoutError",
    "AudioToolNotFoundError",
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
    "NoAudioStreamError",
    "TexCompiler",
    "TexCompilerError",
    "TexNotFoundError",
    "TexTimeoutError",
]
