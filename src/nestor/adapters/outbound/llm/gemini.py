"""Google Gemini outbound LLM adapter using the official google-genai SDK."""

import os
from collections.abc import Callable, Coroutine
from typing import Any

from google import genai
from google.genai import errors, types
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from nestor.ports.llm import (
    LlmAuthenticationError,
    LlmError,
    LlmProviderError,
    LlmRateLimitError,
    LlmResponse,
    LlmSchemaValidationError,
    LlmTimeoutError,
)

DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"


class GeminiClient:
    """Production adapter for Google Gemini language models."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_model: str = DEFAULT_GEMINI_MODEL,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        client: genai.Client | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key and client is None:
            msg = (
                "Gemini API key not found. Provide api_key argument or set "
                "the GEMINI_API_KEY environment variable."
            )
            raise LlmAuthenticationError(msg)

        self._client = client or genai.Client(api_key=resolved_key)
        self.default_model = default_model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LlmResponse[str]:
        """Generate unstructured text from a prompt via Gemini API."""
        target_model = model or self.default_model
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        async def _call() -> types.GenerateContentResponse:
            try:
                return await self._client.aio.models.generate_content(
                    model=target_model,
                    contents=prompt,
                    config=config,
                )
            except Exception as err:
                raise self._translate_error(err) from err

        response = await self._execute_with_retry(_call)
        content = response.text or ""
        finish_reason = self._extract_finish_reason(response)

        return LlmResponse[str](
            content=content,
            parsed=None,
            model=target_model,
            finish_reason=finish_reason,
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
        """Generate schema-constrained structured output parsed into Pydantic."""
        target_model = model or self.default_model
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            response_schema=response_model,
        )

        async def _call() -> types.GenerateContentResponse:
            try:
                return await self._client.aio.models.generate_content(
                    model=target_model,
                    contents=prompt,
                    config=config,
                )
            except Exception as err:
                raise self._translate_error(err) from err

        response = await self._execute_with_retry(_call)

        finish_reason = self._extract_finish_reason(response)
        raw_text = response.text or ""
        parsed: T
        if isinstance(response.parsed, response_model):
            parsed = response.parsed
        else:
            try:
                parsed = response_model.model_validate_json(raw_text)
            except Exception as err:
                msg = f"Failed to parse Gemini output into schema {response_model.__name__}: {err}"
                raise LlmSchemaValidationError(msg) from err

        return LlmResponse[T](
            content=raw_text,
            parsed=parsed,
            model=target_model,
            finish_reason=finish_reason,
        )

    async def _execute_with_retry[R](
        self,
        operation: Callable[[], Coroutine[Any, Any, R]],
    ) -> R:
        """Execute async operation with exponential backoff on transient errors."""
        retrier = AsyncRetrying(
            retry=retry_if_exception_type((LlmRateLimitError, LlmTimeoutError)),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_random_exponential(min=self.base_delay, max=self.max_delay),
            reraise=True,
        )
        async for attempt in retrier:
            with attempt:
                return await operation()
        msg = "Retry loop exhausted without result"
        raise LlmProviderError(msg)

    @staticmethod
    def _translate_error(err: Exception) -> LlmError:
        """Translate upstream Gemini API errors into domain exceptions."""
        if isinstance(err, LlmError):
            return err

        if isinstance(err, errors.APIError):
            status_code = getattr(err, "code", None)
            err_msg = str(err)
            if status_code == 429:
                return LlmRateLimitError(f"Gemini rate limit exceeded: {err_msg}")
            if status_code in (401, 403):
                return LlmAuthenticationError(f"Gemini authentication failed: {err_msg}")
            if status_code in (408, 504):
                return LlmTimeoutError(f"Gemini request timed out: {err_msg}")
            if status_code in (500, 502, 503):
                return LlmRateLimitError(f"Gemini server unavailable ({status_code}): {err_msg}")
            return LlmProviderError(f"Gemini API error ({status_code}): {err_msg}")

        if isinstance(err, TimeoutError):
            return LlmTimeoutError(f"Gemini client timeout: {err}")

        return LlmProviderError(f"Unexpected Gemini error: {err}")

    @staticmethod
    def _extract_finish_reason(response: types.GenerateContentResponse) -> str:
        """Extract normalized finish reason string from candidates."""
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.finish_reason is not None:
                return candidate.finish_reason.name.lower()
        return "stop"
