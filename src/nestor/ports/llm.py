"""LLM client port protocol and standardized response types.

Defines the contract for language model providers.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class LlmResponse[T](BaseModel):
    """Standardized response container across all LLM providers."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    content: str = Field(description="Raw text or JSON string representation of LLM completion.")
    parsed: T | None = Field(
        default=None, description="Parsed structured data if response_model was provided."
    )
    model: str = Field(
        default="unknown", description="Identifier of the model that generated the response."
    )
    finish_reason: str = Field(
        default="stop", description="Reason model stopped generating (e.g. stop, length)."
    )


class LlmError(Exception):
    """Base exception for all LLM port and adapter operations."""


class LlmSchemaValidationError(LlmError):
    """Raised when an LLM fails to output valid structured data matching the schema."""


class LlmProviderError(LlmError):
    """Raised when an upstream LLM provider fails."""


class LlmRateLimitError(LlmProviderError):
    """Raised when an LLM provider rejects a request due to rate limits or quota exhaustion."""


class LlmTimeoutError(LlmProviderError):
    """Raised when an LLM provider request times out."""


class LlmAuthenticationError(LlmProviderError):
    """Raised when authentication or authorization fails with the LLM provider."""


@runtime_checkable
class LlmClient(Protocol):
    """Abstract port for language model operations."""

    async def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LlmResponse[str]:
        """Generate unstructured text from a prompt."""
        ...

    async def generate_structured[T: BaseModel](
        self,
        prompt: str,
        response_model: type[T],
        *,
        system_instruction: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LlmResponse[T]:
        """Generate validated structured data parsed into a Pydantic schema."""
        ...
