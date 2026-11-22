"""Unit tests for multimodal batch extractor port contracts and exception hierarchy."""

from nestor.adapters.outbound.llm.fake_multimodal import FakeMultimodalBatchExtractor
from nestor.ports.multimodal import (
    MultimodalBatchExtractor,
    MultimodalExtractionError,
    MultimodalExtractionParseError,
    MultimodalFileFailedError,
    MultimodalProcessingTimeoutError,
    MultimodalUploadError,
)


class TestMultimodalPortContracts:
    """Verifies runtime protocol conformance and exception inheritance."""

    def test_protocol_conformance(self) -> None:
        """Verify FakeMultimodalBatchExtractor satisfies abstract MultimodalBatchExtractor protocol."""
        extractor = FakeMultimodalBatchExtractor()
        assert isinstance(extractor, MultimodalBatchExtractor)

    def test_arbitrary_object_does_not_conform(self) -> None:
        """Verify arbitrary object fails runtime protocol check."""
        assert not isinstance(object(), MultimodalBatchExtractor)

    def test_exception_hierarchy(self) -> None:
        """Verify domain exceptions correctly inherit from MultimodalExtractionError."""
        assert issubclass(MultimodalUploadError, MultimodalExtractionError)
        assert issubclass(MultimodalProcessingTimeoutError, MultimodalExtractionError)
        assert issubclass(MultimodalFileFailedError, MultimodalExtractionError)
        assert issubclass(MultimodalExtractionParseError, MultimodalExtractionError)
        assert issubclass(MultimodalExtractionError, Exception)
