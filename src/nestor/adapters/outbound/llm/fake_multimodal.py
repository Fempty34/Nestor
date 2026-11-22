"""In-memory test double for multimodal batch extraction."""

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from nestor.domain.contracts.dtc import DTCBlock, DTCBlockMetadata, TimeRange
from nestor.ports.multimodal import MultimodalExtractionError


class MultimodalCallRecord(BaseModel):
    """Execution telemetry record captured by FakeMultimodalBatchExtractor."""

    model_config = ConfigDict(frozen=True)

    time_range: TimeRange
    audio_path: Path | None
    frame_paths: tuple[Path, ...]
    start_index: int
    batch_index: int
    model: str | None


class FakeMultimodalBatchExtractor:
    """Test double implementing MultimodalBatchExtractor for unit testing."""

    def __init__(
        self,
        *,
        default_blocks: Sequence[DTCBlock] | None = None,
        exception_to_raise: Exception | None = None,
    ) -> None:
        self.default_blocks = list(default_blocks) if default_blocks is not None else None
        self.exception_to_raise = exception_to_raise
        self.call_history: list[MultimodalCallRecord] = []

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
        """Record the call and return preconfigured blocks or synthetic defaults."""
        effective_range = time_range or TimeRange(
            start=float(batch_index * 300),
            end=float((batch_index + 1) * 300),
        )

        record = MultimodalCallRecord(
            time_range=effective_range,
            audio_path=audio_path,
            frame_paths=tuple(frame_paths),
            start_index=start_index,
            batch_index=batch_index,
            model=model,
        )
        self.call_history.append(record)

        if self.exception_to_raise is not None:
            raise self.exception_to_raise

        if audio_path is None and not frame_paths:
            msg = (
                "At least one multimodal input asset (audio_path or frame_paths) must be provided."
            )
            raise MultimodalExtractionError(msg)

        if self.default_blocks is not None:
            return self.default_blocks

        # Generate synthetic default DTCBlocks with contiguous indexing
        midpoint = round(effective_range.start + (effective_range.duration / 2), 3)
        return [
            DTCBlock(
                index=start_index,
                time_range=TimeRange(start=effective_range.start, end=midpoint),
                board_content=r"\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}",
                lector_speech="Рассмотрим интеграл Пуассона и сведем его к двойному интегралу.",  # noqa: RUF001
                metadata=DTCBlockMetadata(
                    confidence=0.98,
                    frame_snapshot_path=None,
                    has_formula=True,
                ),
            ),
            DTCBlock(
                index=start_index + 1,
                time_range=TimeRange(start=midpoint, end=effective_range.end),
                board_content=r"I^2 = \iint_{\mathbb{R}^2} e^{-(x^2+y^2)} dx dy",
                lector_speech="Переходя к полярным координатам, вычисляем якобиан преобразования.",
                metadata=DTCBlockMetadata(
                    confidence=0.95,
                    frame_snapshot_path=None,
                    has_formula=True,
                ),
            ),
        ]
