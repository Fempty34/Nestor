"""Unit tests for FakeMultimodalBatchExtractor test double."""

from pathlib import Path

import pytest

from nestor.adapters.outbound.llm.fake_multimodal import FakeMultimodalBatchExtractor
from nestor.domain.contracts.dtc import DTCBlock, DTCBlockMetadata, TimeRange
from nestor.ports.multimodal import MultimodalExtractionError


class TestFakeMultimodalBatchExtractor:
    """Verifies fake behavior, telemetry recording, and error injection."""

    @pytest.mark.asyncio
    async def test_extract_batch_records_telemetry_and_returns_defaults(self) -> None:
        """Verify extractor logs call parameters and generates valid synthetic blocks."""
        extractor = FakeMultimodalBatchExtractor()
        time_range = TimeRange(start=0.0, end=300.0)
        audio = Path("audio/segment_000.wav")
        frames = [Path("frames/f0.jpg"), Path("frames/f1.jpg")]

        blocks = await extractor.extract_batch(
            audio_path=audio,
            frame_paths=frames,
            time_range=time_range,
            start_index=0,
            batch_index=0,
            model="gemini-3.1-flash-lite",
        )

        assert len(blocks) == 2
        assert blocks[0].index == 0
        assert blocks[0].time_range.start == 0.0
        assert blocks[0].time_range.end == 150.0
        assert blocks[0].metadata.has_formula is True
        assert len(extractor.call_history) == 1

        rec = extractor.call_history[0]
        assert rec.time_range == time_range
        assert rec.audio_path == audio
        assert rec.frame_paths == tuple(frames)
        assert rec.model == "gemini-3.1-flash-lite"

    @pytest.mark.asyncio
    async def test_extract_batch_without_assets_raises_error(self) -> None:
        """Verify error is raised if neither audio nor frames are supplied."""
        extractor = FakeMultimodalBatchExtractor()
        time_range = TimeRange(start=0.0, end=300.0)

        with pytest.raises(MultimodalExtractionError, match="At least one multimodal input asset"):
            await extractor.extract_batch(time_range=time_range)

    @pytest.mark.asyncio
    async def test_extract_batch_raises_injected_exception(self) -> None:
        """Verify configured exception is raised upon extraction call."""
        injected = RuntimeError("Provider down")
        extractor = FakeMultimodalBatchExtractor(exception_to_raise=injected)
        time_range = TimeRange(start=0.0, end=300.0)

        with pytest.raises(RuntimeError, match="Provider down"):
            await extractor.extract_batch(audio_path=Path("audio.wav"), time_range=time_range)

    @pytest.mark.asyncio
    async def test_extract_batch_returns_custom_blocks(self) -> None:
        """Verify preconfigured blocks are returned instead of defaults."""
        custom_block = DTCBlock(
            index=10,
            time_range=TimeRange(start=10.0, end=20.0),
            board_content="E = mc^2",
            lector_speech="Energy-mass equivalence",
            metadata=DTCBlockMetadata(confidence=1.0, has_formula=True),
        )
        extractor = FakeMultimodalBatchExtractor(default_blocks=[custom_block])
        blocks = await extractor.extract_batch(
            audio_path=Path("audio.wav"),
            time_range=TimeRange(start=0.0, end=30.0),
        )
        assert blocks == [custom_block]
