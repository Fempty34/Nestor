"""OpenRouter outbound LLM adapter using OpenAI async client."""

from collections.abc import Callable, Coroutine
from typing import Any

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from nestor.core.config.llm import OpenRouterConfig
from nestor.ports.llm import (
    LlmAuthenticationError,
    LlmError,
    LlmProviderError,
    LlmRateLimitError,
    LlmResponse,
    LlmSchemaValidationError,
    LlmTimeoutError,
)


class OpenRouterClient:
    """Production adapter for OpenRouter API supporting arbitrary routed models."""

    def __init__(
        self,
        config: OpenRouterConfig | None = None,
        *,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.config = config or OpenRouterConfig()

        resolved_key = self.config.raw_api_key
        if not resolved_key and client is None:
            msg = (
                "OpenRouter API key not found. Provide api_key via OpenRouterConfig "
                "or set the OPENROUTER_API_KEY environment variable."
            )
            raise LlmAuthenticationError(msg)

        default_headers: dict[str, str] = {}
        if self.config.site_url:
            default_headers["HTTP-Referer"] = self.config.site_url
        if self.config.site_name:
            default_headers["X-Title"] = self.config.site_name

        self._client = client or AsyncOpenAI(
            api_key=resolved_key,
            base_url=self.config.base_url,
            default_headers=default_headers or None,
            timeout=self.config.timeout,
        )
        self.default_model = self.config.default_model
        self.max_retries = self.config.max_retries
        self.base_delay = self.config.base_delay
        self.max_delay = self.config.max_delay

    async def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LlmResponse[str]:
        """Generate unstructured text from prompt via OpenRouter API."""
        target_model = model or self.default_model
        messages = self._build_messages(prompt=prompt, system_instruction=system_instruction)

        async def _call() -> Any:
            try:
                return await self._client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as err:
                raise self._translate_error(err) from err

        response = await self._execute_with_retry(_call)
        choice = response.choices[0]
        content = choice.message.content or ""
        finish_reason = choice.finish_reason or "stop"

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
        messages = self._build_messages(prompt=prompt, system_instruction=system_instruction)

        async def _call() -> Any:
            try:
                return await self._client.beta.chat.completions.parse(
                    model=target_model,
                    messages=messages,
                    response_format=response_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as err:
                raise self._translate_error(err) from err

        response = await self._execute_with_retry(_call)
        choice = response.choices[0]
        raw_text = choice.message.content or ""
        finish_reason = choice.finish_reason or "stop"

        parsed: T
        if hasattr(choice.message, "parsed") and isinstance(choice.message.parsed, response_model):
            parsed = choice.message.parsed
        else:
            try:
                parsed = response_model.model_validate_json(raw_text)
            except Exception as err:
                msg = (
                    f"Failed to parse OpenRouter output into schema "
                    f"{response_model.__name__}: {err}"
                )
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
        """Translate upstream OpenAI/OpenRouter errors into domain exceptions."""
        if isinstance(err, LlmError):
            return err
        if isinstance(err, openai.RateLimitError):
            return LlmRateLimitError(f"OpenRouter rate limit exceeded: {err}")
        if isinstance(err, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return LlmAuthenticationError(f"OpenRouter authentication failed: {err}")
        if isinstance(err, (openai.APITimeoutError, openai.APIConnectionError)):
            return LlmTimeoutError(f"OpenRouter connection or timeout error: {err}")
        if isinstance(err, openai.InternalServerError):
            return LlmRateLimitError(f"OpenRouter server error: {err}")
        if isinstance(err, openai.APIError):
            return LlmProviderError(f"OpenRouter API error: {err}")
        if isinstance(err, TimeoutError):
            return LlmTimeoutError(f"OpenRouter client timeout: {err}")

        return LlmProviderError(f"Unexpected OpenRouter error: {err}")

    @staticmethod
    def _build_messages(
        prompt: str,
        system_instruction: str | None,
    ) -> list[ChatCompletionMessageParam]:
        """Construct standard chat message sequence for completion request."""
        messages: list[ChatCompletionMessageParam] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        return messages
