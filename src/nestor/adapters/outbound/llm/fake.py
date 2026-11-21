"""Deterministic in-memory LLM client adapter for testing."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from nestor.ports.llm import (
    LlmResponse,
    LlmSchemaValidationError,
)


@dataclass(frozen=True)
class LlmCallRecord:
    """Audit record capturing all arguments passed to an LLM invocation."""

    prompt: str
    system_instruction: str | None
    model: str | None
    temperature: float
    max_tokens: int | None
    response_model: type[BaseModel] | None = None


class FakeLlmClient:
    """Deterministic mock LLM provider for fast unit tests."""

    def __init__(
        self,
        *,
        default_response: str = "Mocked LLM completion",
        responses: list[str | BaseModel | LlmResponse[Any]] | None = None,
        prompt_mappings: dict[str, str | BaseModel | LlmResponse[Any]] | None = None,
        error_to_raise: Exception | None = None,
    ) -> None:
        self.default_response = default_response
        self._response_queue: list[str | BaseModel | LlmResponse[Any]] = (
            list(responses) if responses is not None else []
        )
        self.prompt_mappings: dict[str, str | BaseModel | LlmResponse[Any]] = (
            dict(prompt_mappings) if prompt_mappings is not None else {}
        )
        self.error_to_raise = error_to_raise
        self._calls: list[LlmCallRecord] = []

    @property
    def calls(self) -> list[LlmCallRecord]:
        """Return shallow copy of recorded call history."""
        return list(self._calls)

    @property
    def call_count(self) -> int:
        """Return total number of LLM invocations."""
        return len(self._calls)

    @property
    def last_call(self) -> LlmCallRecord | None:
        """Return most recent call record, or None if no calls were made."""
        return self._calls[-1] if self._calls else None

    def reset(self) -> None:
        """Clear recorded calls and reset response queue."""
        self._calls.clear()
        self._response_queue.clear()

    def enqueue_response(self, response: str | BaseModel | LlmResponse[Any]) -> None:
        """Append a canned response to the end of the FIFO queue."""
        self._response_queue.append(response)

    async def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LlmResponse[str]:
        """Generate unstructured text from prompt."""
        self._record_call(
            prompt=prompt,
            system_instruction=system_instruction,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_model=None,
        )

        if self.error_to_raise is not None:
            raise self.error_to_raise

        item = self._resolve_next_response(prompt)
        resolved_model = model or "fake-llm-model"

        if isinstance(item, LlmResponse):
            return item

        content = item if isinstance(item, str) else item.model_dump_json()

        return LlmResponse[str](
            content=content,
            parsed=None,
            model=resolved_model,
            finish_reason="stop",
        )

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
        self._record_call(
            prompt=prompt,
            system_instruction=system_instruction,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_model=response_model,
        )

        if self.error_to_raise is not None:
            raise self.error_to_raise

        item = self._resolve_next_response(prompt)
        resolved_model = model or "fake-llm-model"

        if isinstance(item, LlmResponse):
            if isinstance(item.parsed, response_model):
                return item
            if isinstance(item.content, str):
                try:
                    parsed = response_model.model_validate_json(item.content)
                    return LlmResponse[T](
                        content=item.content,
                        parsed=parsed,
                        model=item.model,
                        finish_reason=item.finish_reason,
                    )
                except Exception as err:
                    msg = (
                        f"Failed to validate response JSON against schema "
                        f"{response_model.__name__}: {err}"
                    )
                    raise LlmSchemaValidationError(msg) from err

        if isinstance(item, response_model):
            content = item.model_dump_json()
            return LlmResponse[T](
                content=content,
                parsed=item,
                model=resolved_model,
                finish_reason="stop",
            )

        if isinstance(item, BaseModel):
            content = item.model_dump_json()
            try:
                parsed = response_model.model_validate_json(content)
                return LlmResponse[T](
                    content=content,
                    parsed=parsed,
                    model=resolved_model,
                    finish_reason="stop",
                )
            except Exception as err:
                msg = (
                    f"Incompatible model {type(item).__name__} for expected "
                    f"schema {response_model.__name__}: {err}"
                )
                raise LlmSchemaValidationError(msg) from err

        if isinstance(item, str):
            try:
                parsed = response_model.model_validate_json(item)
                return LlmResponse[T](
                    content=item,
                    parsed=parsed,
                    model=resolved_model,
                    finish_reason="stop",
                )
            except Exception as err:
                msg = f"Failed to parse string response into {response_model.__name__}: {err}"
                raise LlmSchemaValidationError(msg) from err

        msg = f"Unexpected fake response type {type(item)} for schema {response_model.__name__}"
        raise LlmSchemaValidationError(msg)

    def _resolve_next_response(self, prompt: str) -> str | BaseModel | LlmResponse[Any]:
        if self._response_queue:
            return self._response_queue.pop(0)

        for pattern, mapped_response in self.prompt_mappings.items():
            if pattern in prompt:
                return mapped_response

        return self.default_response

    def _record_call(
        self,
        *,
        prompt: str,
        system_instruction: str | None,
        model: str | None,
        temperature: float,
        max_tokens: int | None,
        response_model: type[BaseModel] | None,
    ) -> None:
        self._calls.append(
            LlmCallRecord(
                prompt=prompt,
                system_instruction=system_instruction,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_model=response_model,
            )
        )
