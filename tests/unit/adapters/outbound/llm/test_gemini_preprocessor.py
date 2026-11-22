"""Unit tests for GeminiMultimodalBatchExtractor.

Verifies:
1. File upload via Files API and processing state polling (ACTIVE, PROCESSING -> ACTIVE, FAILED, Timeout).
2. Structured prompt construction instructing model on blackboard LaTeX formulas and speech alignment.
3. Tenacity retry loop on transient rate limits (429) and timeouts.
4. Automatic deletion of uploaded files on Gemini servers post-extraction (including partial upload failures).
5. Output parsing, schema validation, and conversion to strictly validated Pydantic DTC chunks.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import errors, types
from pydantic import SecretStr

from nestor.adapters.outbound.llm.gemini_preprocessor import (
    MULTIMODAL_PREPROCESS_SYSTEM_INSTRUCTION,
    DTCBatchExtractionResponse,
    GeminiMultimodalBatchExtractor,
)
from nestor.core.config.multimodal import GeminiPreprocessConfig
from nestor.domain.contracts.dtc import DTCBlock, DTCBlockMetadata, TimeRange
from nestor.ports.llm import LlmAuthenticationError, LlmProviderError
from nestor.ports.multimodal import (
    MultimodalBatchExtractor,
    MultimodalExtractionError,
    MultimodalExtractionParseError,
    MultimodalFileFailedError,
    MultimodalProcessingTimeoutError,
    MultimodalUploadError,
)


@pytest.fixture
def sample_audio_path(tmp_path: Path) -> Path:
    """Create a temporary synthetic audio file."""
    audio_file = tmp_path / "segment_000.wav"
    audio_file.write_bytes(b"RIFF dummy wav audio data")
    return audio_file


@pytest.fixture
def sample_frame_paths(tmp_path: Path) -> list[Path]:
    """Create temporary synthetic visual frame files."""
    f1 = tmp_path / "frame_000.jpg"
    f2 = tmp_path / "frame_001.png"
    f1.write_bytes(b"\xff\xd8\xff dummy jpeg")
    f2.write_bytes(b"\x89PNG dummy png")
    return [f1, f2]


@pytest.fixture
def mock_genai_client() -> MagicMock:
    """Create a fully mocked google-genai Client."""
    mock_client = MagicMock()
    mock_client.aio = MagicMock()
    mock_client.aio.files = MagicMock()
    mock_client.aio.models = MagicMock()
    return mock_client


class TestGeminiMultimodalBatchExtractorInitialization:
    """Verifies protocol conformance and API key configuration handling."""

    def test_protocol_conformance(self, mock_genai_client: MagicMock) -> None:
        """Verify adapter conforms to MultimodalBatchExtractor protocol."""
        config = GeminiPreprocessConfig(api_key=SecretStr("dummy-key"))
        extractor = GeminiMultimodalBatchExtractor(config=config, client=mock_genai_client)
        assert isinstance(extractor, MultimodalBatchExtractor)

    def test_missing_api_key_raises_auth_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify LlmAuthenticationError is raised when no API key is provided or found."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(LlmAuthenticationError, match="Gemini API key not found"):
            GeminiMultimodalBatchExtractor(config=GeminiPreprocessConfig(api_key=None))

    def test_default_model_is_gemini_3_1_flash_lite(self, mock_genai_client: MagicMock) -> None:
        """Verify default configured model matches Gemini 3.1 Flash Lite."""
        config = GeminiPreprocessConfig(api_key=SecretStr("key"))
        extractor = GeminiMultimodalBatchExtractor(config=config, client=mock_genai_client)
        assert extractor.default_model == "gemini-3.1-flash-lite"


class TestFilesApiUploadAndPolling:
    """Checklist #1: Implement file upload and processing state polling."""

    @pytest.mark.asyncio
    async def test_upload_assets_sets_correct_mime_types(
        self,
        mock_genai_client: MagicMock,
        sample_audio_path: Path,
        sample_frame_paths: list[Path],
    ) -> None:
        """Verify audio and image assets are uploaded with proper MIME types."""
        file_audio = types.File(
            name="files/audio1",
            uri="https://generativelanguage.googleapis.com/v1beta/files/audio1",
            mime_type="audio/wav",
            state=types.FileState.ACTIVE,
        )
        file_img1 = types.File(
            name="files/img1",
            uri="https://generativelanguage.googleapis.com/v1beta/files/img1",
            mime_type="image/jpeg",
            state=types.FileState.ACTIVE,
        )
        file_img2 = types.File(
            name="files/img2",
            uri="https://generativelanguage.googleapis.com/v1beta/files/img2",
            mime_type="image/png",
            state=types.FileState.ACTIVE,
        )

        mock_genai_client.aio.files.upload = AsyncMock(
            side_effect=[file_audio, file_img1, file_img2]
        )

        config = GeminiPreprocessConfig(api_key=SecretStr("key"))
        extractor = GeminiMultimodalBatchExtractor(config=config, client=mock_genai_client)

        uploaded: list[types.File] = []
        await extractor._upload_assets(
            audio_path=sample_audio_path,
            frame_paths=sample_frame_paths,
            uploaded_files=uploaded,
        )

        assert len(uploaded) == 3
        assert mock_genai_client.aio.files.upload.await_count == 3

        calls = mock_genai_client.aio.files.upload.await_args_list
        assert calls[0].kwargs["config"].mime_type == "audio/wav"
        assert calls[1].kwargs["config"].mime_type == "image/jpeg"
        assert calls[2].kwargs["config"].mime_type == "image/png"

    @pytest.mark.asyncio
    async def test_upload_missing_file_raises_file_not_found(
        self,
        mock_genai_client: MagicMock,
    ) -> None:
        """Verify FileNotFoundError is raised if a local asset path is missing."""
        config = GeminiPreprocessConfig(api_key=SecretStr("key"))
        extractor = GeminiMultimodalBatchExtractor(config=config, client=mock_genai_client)

        uploaded: list[types.File] = []
        with pytest.raises(FileNotFoundError, match="Audio asset file not found"):
            await extractor._upload_assets(
                audio_path=Path("nonexistent.wav"),
                frame_paths=[],
                uploaded_files=uploaded,
            )

    @pytest.mark.asyncio
    async def test_polling_waits_for_processing_state_to_become_active(
        self,
        mock_genai_client: MagicMock,
    ) -> None:
        """Verify polling queries files.get until file transitions from PROCESSING to ACTIVE."""
        file_processing = types.File(
            name="files/f1",
            uri="https://generativelanguage.googleapis.com/v1beta/files/f1",
            mime_type="audio/wav",
            state=types.FileState.PROCESSING,
        )
        file_active = types.File(
            name="files/f1",
            uri="https://generativelanguage.googleapis.com/v1beta/files/f1",
            mime_type="audio/wav",
            state=types.FileState.ACTIVE,
        )

        mock_genai_client.aio.files.get = AsyncMock(side_effect=[file_processing, file_active])

        config = GeminiPreprocessConfig(api_key=SecretStr("key"), poll_interval=0.01)
        extractor = GeminiMultimodalBatchExtractor(config=config, client=mock_genai_client)

        result = await extractor._wait_for_files([file_processing])

        assert len(result) == 1
        assert result[0].state == types.FileState.ACTIVE
        assert mock_genai_client.aio.files.get.await_count == 2

    @pytest.mark.asyncio
    async def test_polling_file_failed_raises_error(
        self,
        mock_genai_client: MagicMock,
    ) -> None:
        """Verify MultimodalFileFailedError is raised when file state transitions to FAILED."""
        file_processing = types.File(
            name="files/bad",
            uri="https://generativelanguage.googleapis.com/v1beta/files/bad",
            mime_type="audio/wav",
            state=types.FileState.PROCESSING,
        )
        file_failed = types.File(
            name="files/bad",
            uri="https://generativelanguage.googleapis.com/v1beta/files/bad",
            mime_type="audio/wav",
            state=types.FileState.FAILED,
            error=types.FileStatus(code=3, message="Corrupt audio file"),
        )

        mock_genai_client.aio.files.get = AsyncMock(return_value=file_failed)

        config = GeminiPreprocessConfig(api_key=SecretStr("key"), poll_interval=0.01)
        extractor = GeminiMultimodalBatchExtractor(config=config, client=mock_genai_client)

        with pytest.raises(MultimodalFileFailedError, match="failed processing on Gemini servers"):
            await extractor._wait_for_files([file_processing])

    @pytest.mark.asyncio
    async def test_polling_timeout_raises_timeout_error(
        self,
        mock_genai_client: MagicMock,
    ) -> None:
        """Verify MultimodalProcessingTimeoutError is raised when polling exceeds deadline."""
        file_processing = types.File(
            name="files/slow",
            uri="https://generativelanguage.googleapis.com/v1beta/files/slow",
            mime_type="audio/wav",
            state=types.FileState.PROCESSING,
        )
        mock_genai_client.aio.files.get = AsyncMock(return_value=file_processing)

        config = GeminiPreprocessConfig(
            api_key=SecretStr("key"),
            poll_interval=0.01,
            poll_timeout=0.03,
        )
        extractor = GeminiMultimodalBatchExtractor(config=config, client=mock_genai_client)

        with pytest.raises(MultimodalProcessingTimeoutError, match="processing timed out"):
            await extractor._wait_for_files([file_processing])


class TestStructuredPromptAndInstructions:
    """Checklist #2: Write structured prompt instructing model to extract LaTeX and speech."""

    def test_system_instruction_contains_stem_requirements(self) -> None:
        """Verify system instruction explicitly mandates LaTeX math mode and speech alignment."""
        instruction = MULTIMODAL_PREPROCESS_SYSTEM_INSTRUCTION
        assert "LaTeX" in instruction
        assert "blackboard" in instruction
        assert "synchronized" in instruction
        assert "has_formula" in instruction
        assert "monotonicity" in instruction

    def test_build_extraction_prompt_formats_batch_context(self) -> None:
        """Verify prompt builder embeds formatted range, duration, and asset counts."""
        time_range = TimeRange(start=300.0, end=600.0)
        prompt = GeminiMultimodalBatchExtractor._build_extraction_prompt(
            time_range=time_range,
            batch_index=1,
            batch_duration=300.0,
            has_audio=True,
            frame_count=150,
        )

        assert "batch #1" in prompt
        assert "00:05:00 - 00:10:00" in prompt
        assert "300.0 seconds" in prompt
        assert "Audio present: True" in prompt
        assert "Sampled visual frames: 150" in prompt


class TestAutomaticDeletionPostExtraction:
    """Checklist #4: Implement automatic deletion of uploaded files on Gemini servers post-extraction."""

    @pytest.mark.asyncio
    async def test_deletion_executed_on_successful_extraction(
        self,
        mock_genai_client: MagicMock,
        sample_audio_path: Path,
    ) -> None:
        """Verify files.delete is awaited for all uploaded files after successful generation."""
        mock_file = types.File(
            name="files/audio_success",
            uri="https://generativelanguage.googleapis.com/v1beta/files/audio_success",
            mime_type="audio/wav",
            state=types.FileState.ACTIVE,
        )
        mock_genai_client.aio.files.upload = AsyncMock(return_value=mock_file)
        mock_genai_client.aio.files.delete = AsyncMock()

        sample_block = DTCBlock(
            index=0,
            time_range=TimeRange(start=0.0, end=30.0),
            board_content="f(x) = x^2",
            lector_speech="Defines quadratic function",
            metadata=DTCBlockMetadata(confidence=0.98, has_formula=True),
        )
        sample_response = DTCBatchExtractionResponse(blocks=[sample_block])
        mock_genai_res = MagicMock(spec=types.GenerateContentResponse)
        mock_genai_res.parsed = sample_response
        mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_genai_res)

        config = GeminiPreprocessConfig(api_key=SecretStr("key"))
        extractor = GeminiMultimodalBatchExtractor(config=config, client=mock_genai_client)

        blocks = await extractor.extract_batch(
            audio_path=sample_audio_path,
            time_range=TimeRange(start=0.0, end=60.0),
        )

        assert len(blocks) == 1
        mock_genai_client.aio.files.delete.assert_awaited_once_with(name="files/audio_success")

    @pytest.mark.asyncio
    async def test_deletion_guaranteed_even_when_generation_fails(
        self,
        mock_genai_client: MagicMock,
        sample_audio_path: Path,
    ) -> None:
        """Verify files.delete is called even if Gemini model generation throws an exception."""
        mock_file = types.File(
            name="files/audio_err",
            uri="https://generativelanguage.googleapis.com/v1beta/files/audio_err",
            mime_type="audio/wav",
            state=types.FileState.ACTIVE,
        )
        mock_genai_client.aio.files.upload = AsyncMock(return_value=mock_file)
        mock_genai_client.aio.files.delete = AsyncMock()

        # Simulate catastrophic generation failure
        mock_genai_client.aio.models.generate_content = AsyncMock(
            side_effect=RuntimeError("Internal model error")
        )

        config = GeminiPreprocessConfig(api_key=SecretStr("key"), max_retries=1)
        extractor = GeminiMultimodalBatchExtractor(config=config, client=mock_genai_client)

        with pytest.raises(LlmProviderError, match="Internal model error"):
            await extractor.extract_batch(
                audio_path=sample_audio_path,
                time_range=TimeRange(start=0.0, end=60.0),
            )

        # Deletion must still have executed in finally block!
        mock_genai_client.aio.files.delete.assert_awaited_once_with(name="files/audio_err")

    @pytest.mark.asyncio
    async def test_deletion_cleans_up_partial_uploads_on_upload_failure(
        self,
        mock_genai_client: MagicMock,
        sample_audio_path: Path,
        sample_frame_paths: list[Path],
    ) -> None:
        """Verify any already-uploaded files are deleted if subsequent asset upload fails."""
        mock_file_audio = types.File(
            name="files/partial_audio",
            uri="https://generativelanguage.googleapis.com/v1beta/files/partial_audio",
            mime_type="audio/wav",
            state=types.FileState.ACTIVE,
        )

        # Audio succeeds, but next frame upload raises an error
        mock_genai_client.aio.files.upload = AsyncMock(
            side_effect=[mock_file_audio, RuntimeError("Network dropped")]
        )
        mock_genai_client.aio.files.delete = AsyncMock()

        config = GeminiPreprocessConfig(api_key=SecretStr("key"))
        extractor = GeminiMultimodalBatchExtractor(config=config, client=mock_genai_client)

        with pytest.raises(MultimodalUploadError):
            await extractor.extract_batch(
                audio_path=sample_audio_path,
                frame_paths=sample_frame_paths,
            )

        # First file must be deleted even though second upload failed!
        mock_genai_client.aio.files.delete.assert_awaited_once_with(name="files/partial_audio")


class TestRetryLoop:
    """Checklist #3: Add retry loop."""

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit_error(
        self,
        mock_genai_client: MagicMock,
        sample_audio_path: Path,
    ) -> None:
        """Verify transient rate limit errors (HTTP 429) trigger retry and succeed on subsequent attempt."""
        mock_file = types.File(
            name="files/retry",
            uri="https://generativelanguage.googleapis.com/v1beta/files/retry",
            mime_type="audio/wav",
            state=types.FileState.ACTIVE,
        )
        mock_genai_client.aio.files.upload = AsyncMock(return_value=mock_file)
        mock_genai_client.aio.files.delete = AsyncMock()

        rate_limit_err = errors.APIError(429, "Resource exhausted")
        sample_block = DTCBlock(
            index=0,
            time_range=TimeRange(start=0.0, end=45.0),
            board_content=r"\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}",
            lector_speech="Закон электромагнитной индукции Фарадея.",
            metadata=DTCBlockMetadata(confidence=0.97, has_formula=True),
        )
        successful_response = MagicMock(spec=types.GenerateContentResponse)
        successful_response.parsed = DTCBatchExtractionResponse(blocks=[sample_block])

        # First call fails with 429, second call succeeds
        mock_genai_client.aio.models.generate_content = AsyncMock(
            side_effect=[rate_limit_err, successful_response]
        )

        config = GeminiPreprocessConfig(
            api_key=SecretStr("key"),
            max_retries=3,
            base_delay=0.01,
            max_delay=0.05,
        )
        extractor = GeminiMultimodalBatchExtractor(config=config, client=mock_genai_client)

        blocks = await extractor.extract_batch(
            audio_path=sample_audio_path,
            time_range=TimeRange(start=0.0, end=300.0),
        )

        assert len(blocks) == 1
        assert blocks[0].board_content.startswith(r"\nabla \times")
        assert mock_genai_client.aio.models.generate_content.await_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_server_unavailable_timeout(
        self,
        mock_genai_client: MagicMock,
        sample_audio_path: Path,
    ) -> None:
        """Verify 503 Service Unavailable is translated to retryable error and retried."""
        mock_file = types.File(
            name="files/retry_503",
            uri="https://generativelanguage.googleapis.com/v1beta/files/retry_503",
            mime_type="audio/wav",
            state=types.FileState.ACTIVE,
        )
        mock_genai_client.aio.files.upload = AsyncMock(return_value=mock_file)
        mock_genai_client.aio.files.delete = AsyncMock()

        server_err = errors.APIError(503, "Backend unavailable")
        successful_response = MagicMock(spec=types.GenerateContentResponse)
        successful_response.parsed = DTCBatchExtractionResponse(blocks=[])

        mock_genai_client.aio.models.generate_content = AsyncMock(
            side_effect=[server_err, successful_response]
        )

        config = GeminiPreprocessConfig(
            api_key=SecretStr("key"),
            max_retries=2,
            base_delay=0.01,
            max_delay=0.05,
        )
        extractor = GeminiMultimodalBatchExtractor(config=config, client=mock_genai_client)

        blocks = await extractor.extract_batch(
            audio_path=sample_audio_path,
            time_range=TimeRange(start=0.0, end=300.0),
        )

        assert len(blocks) == 1
        assert mock_genai_client.aio.models.generate_content.await_count == 2


class TestOutputParsingAndSchemaValidation:
    """Checklist #5: Write unit test to verify output parsing and schema validation."""

    @pytest.mark.asyncio
    async def test_parses_json_response_and_computes_contiguous_indices(
        self,
        mock_genai_client: MagicMock,
        sample_audio_path: Path,
    ) -> None:
        """Verify text JSON output parses into DTCBatchExtractionResponse with contiguous indices."""
        mock_file = types.File(
            name="files/f_parse",
            uri="https://generativelanguage.googleapis.com/v1beta/files/f_parse",
            mime_type="audio/wav",
            state=types.FileState.ACTIVE,
        )
        mock_genai_client.aio.files.upload = AsyncMock(return_value=mock_file)
        mock_genai_client.aio.files.delete = AsyncMock()

        json_payload = """
        {
            "blocks": [
                {
                    "index": 0,
                    "time_range": {"start": 300.0, "end": 350.5},
                    "board_content": "\\\\lim_{n \\\\to \\\\infty} (1 + 1/n)^n = e",
                    "lector_speech": "Второй замечательный предел.",
                    "metadata": {"confidence": 0.99, "has_formula": true}
                },
                {
                    "index": 1,
                    "time_range": {"start": 350.5, "end": 420.0},
                    "board_content": "",
                    "lector_speech": "Переходим к доказательству через бином Ньютона.",
                    "metadata": {"confidence": 0.94, "has_formula": false}
                }
            ]
        }
        """
        mock_res = MagicMock(spec=types.GenerateContentResponse)
        mock_res.parsed = None
        mock_res.text = json_payload
        mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_res)

        config = GeminiPreprocessConfig(api_key=SecretStr("key"))
        extractor = GeminiMultimodalBatchExtractor(config=config, client=mock_genai_client)

        time_range = TimeRange(start=300.0, end=600.0)
        blocks = await extractor.extract_batch(
            audio_path=sample_audio_path,
            time_range=time_range,
            start_index=5,
        )

        assert len(blocks) == 2

        # Block 0: index must strictly follow start_index = 5
        b0 = blocks[0]
        assert b0.index == 5
        assert b0.time_range.start == 300.0
        assert b0.time_range.end == 350.5
        assert b0.metadata.has_formula is True
        assert b0.metadata.confidence == 0.99
        assert "lim" in b0.board_content

        # Block 1: index must strictly follow start_index + 1 = 6
        b1 = blocks[1]
        assert b1.index == 6
        assert b1.time_range.start == 350.5
        assert b1.time_range.end == 420.0
        assert b1.board_content == ""
        assert "бином Ньютона" in b1.lector_speech

    @pytest.mark.asyncio
    async def test_empty_json_text_raises_parse_error(
        self,
        mock_genai_client: MagicMock,
        sample_audio_path: Path,
    ) -> None:
        """Verify empty text response raises MultimodalExtractionParseError."""
        mock_file = types.File(
            name="files/f_empty",
            uri="https://generativelanguage.googleapis.com/v1beta/files/f_empty",
            mime_type="audio/wav",
            state=types.FileState.ACTIVE,
        )
        mock_genai_client.aio.files.upload = AsyncMock(return_value=mock_file)
        mock_genai_client.aio.files.delete = AsyncMock()

        mock_res = MagicMock(spec=types.GenerateContentResponse)
        mock_res.parsed = None
        mock_res.text = ""
        mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_res)

        config = GeminiPreprocessConfig(api_key=SecretStr("key"), max_retries=1)
        extractor = GeminiMultimodalBatchExtractor(config=config, client=mock_genai_client)

        with pytest.raises(MultimodalExtractionParseError, match="returned empty response text"):
            await extractor.extract_batch(
                audio_path=sample_audio_path,
                time_range=TimeRange(start=0.0, end=60.0),
            )

    @pytest.mark.asyncio
    async def test_empty_blocks_yields_fallback_dtc_block(
        self,
        mock_genai_client: MagicMock,
        sample_audio_path: Path,
    ) -> None:
        """Verify response with empty blocks list generates safe fallback block."""
        mock_file = types.File(
            name="files/f_fallback",
            uri="https://generativelanguage.googleapis.com/v1beta/files/f_fallback",
            mime_type="audio/wav",
            state=types.FileState.ACTIVE,
        )
        mock_genai_client.aio.files.upload = AsyncMock(return_value=mock_file)
        mock_genai_client.aio.files.delete = AsyncMock()

        mock_res = MagicMock(spec=types.GenerateContentResponse)
        mock_res.parsed = DTCBatchExtractionResponse(blocks=[])
        mock_genai_client.aio.models.generate_content = AsyncMock(return_value=mock_res)

        config = GeminiPreprocessConfig(api_key=SecretStr("key"))
        extractor = GeminiMultimodalBatchExtractor(config=config, client=mock_genai_client)

        time_range = TimeRange(start=60.0, end=120.0)
        blocks = await extractor.extract_batch(
            audio_path=sample_audio_path,
            time_range=time_range,
            start_index=10,
        )

        assert len(blocks) == 1
        assert blocks[0].index == 10
        assert blocks[0].time_range == time_range
        assert "No speech or blackboard" in blocks[0].lector_speech

    @pytest.mark.asyncio
    async def test_extract_batch_without_assets_raises_error(self) -> None:
        """Verify error is raised if neither audio nor frames are supplied."""
        config = GeminiPreprocessConfig(api_key=SecretStr("key"))
        extractor = GeminiMultimodalBatchExtractor(config=config, client=MagicMock())

        with pytest.raises(MultimodalExtractionError, match="At least one multimodal input asset"):
            await extractor.extract_batch()
