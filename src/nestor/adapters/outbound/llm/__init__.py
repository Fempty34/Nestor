"""LLM provider adapters."""

from nestor.adapters.outbound.llm.fake import FakeLlmClient, LlmCallRecord
from nestor.adapters.outbound.llm.fake_multimodal import (
    FakeMultimodalBatchExtractor,
    MultimodalCallRecord,
)
from nestor.adapters.outbound.llm.gemini import GeminiClient
from nestor.adapters.outbound.llm.gemini_preprocessor import (
    MULTIMODAL_PREPROCESS_SYSTEM_INSTRUCTION,
    DTCBatchExtractionResponse,
    GeminiMultimodalBatchExtractor,
)
from nestor.adapters.outbound.llm.openrouter import OpenRouterClient

__all__ = [
    "MULTIMODAL_PREPROCESS_SYSTEM_INSTRUCTION",
    "DTCBatchExtractionResponse",
    "FakeLlmClient",
    "FakeMultimodalBatchExtractor",
    "GeminiClient",
    "GeminiMultimodalBatchExtractor",
    "LlmCallRecord",
    "MultimodalCallRecord",
    "OpenRouterClient",
]
