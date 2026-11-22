"""Multimodal extraction port protocol, result types, and domain exceptions.

Defines the contract for ingesting synchronized audio and video frame batches
and extracting structured, time-coded DTC blocks using vision-language models.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from nestor.domain.contracts.dtc import DTCBlock, TimeRange


class MultimodalExtractionError(Exception):
    """Base exception for all multimodal batch extraction operations."""


class MultimodalUploadError(MultimodalExtractionError):
    """Raised when an asset fails to upload to the multimodal provider."""


class MultimodalProcessingTimeoutError(MultimodalExtractionError):
    """Raised when uploaded file processing exceeds the polling deadline."""


class MultimodalFileFailedError(MultimodalExtractionError):
    """Raised when the provider reports a failure while processing an uploaded asset."""


class MultimodalExtractionParseError(MultimodalExtractionError):
    """Raised when the multimodal model output cannot be parsed into validated DTC chunks."""


@runtime_checkable
class MultimodalBatchExtractor(Protocol):
    """Abstract port for extracting blackboard formulas and synchronized speech from multimodal batches."""

    async def extract_batch(
        self,
        audio_path: Path | None = None,
        frame_paths: Sequence[Path] = (),
        *,
        time_range: TimeRange | None = None,
        start_index: int = 0,
        batch_index: int = 0,
        model: str | None = None,
    ) -> list[DTCBlock]:
        """Upload audio and frames, poll state, extract formulas/speech, and return validated DTC blocks."""
        ...
