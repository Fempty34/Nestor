"""LLM provider adapters."""

from nestor.adapters.outbound.llm.fake import FakeLlmClient, LlmCallRecord

__all__ = [
    "FakeLlmClient",
    "LlmCallRecord",
]
