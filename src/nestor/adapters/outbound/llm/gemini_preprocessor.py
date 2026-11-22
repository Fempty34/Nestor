"""Google Gemini multimodal batch extraction adapter using the official google-genai SDK.

Handles uploading synchronized 5-minute audio segments and video frames to the Gemini Files API,
polling file processing states, extracting blackboard LaTeX formulas and speech transcript
via Gemini 3.1 Flash Lite with structured outputs, and automatically cleaning up server-side files.
"""

import asyncio
import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from nestor.core.config.multimodal import GeminiPreprocessConfig
from nestor.domain.contracts.dtc import DTCBlock, DTCBlockMetadata, TimeRange
from nestor.ports.llm import (
    LlmAuthenticationError,
    LlmError,
    LlmProviderError,
    LlmRateLimitError,
    LlmTimeoutError,
)
from nestor.ports.multimodal import (
    MultimodalExtractionError,
    MultimodalExtractionParseError,
    MultimodalFileFailedError,
    MultimodalProcessingTimeoutError,
    MultimodalUploadError,
)

logger = logging.getLogger(__name__)

MULTIMODAL_PREPROCESS_SYSTEM_INSTRUCTION = """\
You are an expert academic STEM multimodal transcriber and LaTeX typesetter.
Your task is to analyze synchronized audio and blackboard video frames from a 5-minute lecture segment.

CRITICAL INSTRUCTIONS:
1. Extract ALL mathematical formulas, chemical equations, diagrams, definitions, and theorems written on the blackboard into clean, standard LaTeX.
   - Use standard LaTeX math mode ($...$ or \\[...\\]) and environments (e.g. aligned, cases, bmatrix).
   - Ensure syntactically valid LaTeX without markdown backtick wrappers inside the field.
   - Do NOT invent or hallucinate formulas not visible on the blackboard.
2. Transcribe verbatim the lecturer's spoken explanation synchronized with the appearance and discussion of each blackboard element.
   - Omit verbal hesitation and filler sounds ("um", "uh", "э-э", "ну"), but strictly preserve technical terminology, variable names, and logical reasoning.
3. Divide the 5-minute batch into non-overlapping chronological chunks.
   - For each block, specify time_range with start and end in seconds (start < end).
   - Ensure temporal monotonicity: start time of block (i) must be >= start time of block (i-1).
4. Set metadata.has_formula = true whenever mathematical notation or formulas are written on the blackboard.
5. Provide a realistic confidence score between 0.0 and 1.0 in metadata.confidence reflecting handwriting clarity and transcription certainty.
6. Every block must contain either blackboard content or lecturer speech (or both).
"""


class DTCBatchExtractionResponse(BaseModel):
    """Structured response container containing validated DTC blocks extracted by Gemini."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    blocks: list[DTCBlock] = Field(
        default_factory=list,
        description="Chronologically ordered sequence of validated DTC lecture blocks.",
    )


class GeminiMultimodalBatchExtractor:
    """Production outbound adapter for extracting lecture DTC blocks via Gemini Files API."""

    def __init__(
        self,
        config: GeminiPreprocessConfig | None = None,
        *,
        client: genai.Client | None = None,
    ) -> None:
        self.config = config or GeminiPreprocessConfig()

        resolved_key = self.config.raw_api_key
        if not resolved_key and client is None:
            msg = (
                "Gemini API key not found. Provide api_key via GeminiPreprocessConfig "
                "or set the GEMINI_API_KEY environment variable."
            )
            raise LlmAuthenticationError(msg)

        self._client = client or genai.Client(api_key=resolved_key)
        self.default_model = self.config.default_model
        self.max_retries = self.config.max_retries
        self.base_delay = self.config.base_delay
        self.max_delay = self.config.max_delay
        self.poll_interval = self.config.poll_interval
        self.poll_timeout = self.config.poll_timeout

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
        """Upload assets via Files API, poll processing state, extract DTC blocks, and delete server files."""
        if audio_path is None and not frame_paths:
            msg = (
                "At least one multimodal input asset (audio_path or frame_paths) must be provided."
            )
            raise MultimodalExtractionError(msg)

        effective_time_range = time_range or TimeRange(
            start=float(batch_index * 300),
            end=float((batch_index + 1) * 300),
        )

        target_model = model or self.default_model
        uploaded_files: list[types.File] = []

        try:
            # 1. Upload assets via Files API (tracking files continuously for safe cleanup)
            await self._upload_assets(
                audio_path=audio_path,
                frame_paths=frame_paths,
                uploaded_files=uploaded_files,
            )

            # 2. Poll processing states until all files are ACTIVE
            active_files = await self._wait_for_files(uploaded_files)

            # 3. Construct structured generation call
            batch_duration = effective_time_range.duration
            prompt = self._build_extraction_prompt(
                time_range=effective_time_range,
                batch_index=batch_index,
                batch_duration=batch_duration,
                has_audio=audio_path is not None,
                frame_count=len(frame_paths),
            )

            # 4. Generate structured content with retry loop
            raw_response = await self._generate_with_retry(
                target_model=target_model,
                files=active_files,
                prompt=prompt,
            )

            # 5. Parse and validate response into DTC chunks
            return self._parse_and_validate_dtc_blocks(
                raw_response=raw_response,
                time_range=effective_time_range,
                start_index=start_index,
            )

        finally:
            # 6. Automatic deletion of uploaded files on Gemini servers
            await self._cleanup_files(uploaded_files)

    async def _upload_assets(
        self,
        audio_path: Path | None,
        frame_paths: Sequence[Path],
        uploaded_files: list[types.File],
    ) -> None:
        """Upload audio and image files to Gemini Files API, tracking each uploaded file immediately."""
        try:
            if audio_path is not None:
                if not audio_path.is_file():
                    msg = f"Audio asset file not found at '{audio_path}'"
                    raise FileNotFoundError(msg)

                audio_mime = "audio/wav"
                if audio_path.suffix.lower() == ".mp3":
                    audio_mime = "audio/mp3"
                elif audio_path.suffix.lower() == ".m4a":
                    audio_mime = "audio/m4a"

                audio_file = await self._client.aio.files.upload(
                    file=audio_path,
                    config=types.UploadFileConfig(
                        display_name=audio_path.name,
                        mime_type=audio_mime,
                    ),
                )
                uploaded_files.append(audio_file)

            for frame_path in frame_paths:
                if not frame_path.is_file():
                    msg = f"Visual frame asset not found at '{frame_path}'"
                    raise FileNotFoundError(msg)

                image_mime = "image/jpeg"
                if frame_path.suffix.lower() == ".png":
                    image_mime = "image/png"

                frame_file = await self._client.aio.files.upload(
                    file=frame_path,
                    config=types.UploadFileConfig(
                        display_name=frame_path.name,
                        mime_type=image_mime,
                    ),
                )
                uploaded_files.append(frame_file)

        except FileNotFoundError:
            raise
        except Exception as err:
            msg = f"Failed to upload multimodal asset to Gemini Files API: {err}"
            raise MultimodalUploadError(msg) from err

    async def _wait_for_files(self, files: list[types.File]) -> list[types.File]:
        """Poll file processing states until all uploaded files are ACTIVE."""
        active_files: list[types.File] = []
        loop = asyncio.get_running_loop()
        start_time = loop.time()

        for file in files:
            current_file = file
            file_name = current_file.name or ""

            while current_file.state in (
                types.FileState.PROCESSING,
                types.FileState.STATE_UNSPECIFIED,
                "PROCESSING",
                "STATE_UNSPECIFIED",
            ):
                elapsed = loop.time() - start_time
                if elapsed > self.poll_timeout:
                    msg = (
                        f"Files API processing timed out after {elapsed:.1f}s "
                        f"for file '{file_name}'"
                    )
                    raise MultimodalProcessingTimeoutError(msg)

                await asyncio.sleep(self.poll_interval)
                try:
                    current_file = await self._client.aio.files.get(name=file_name)
                except Exception as err:
                    msg = f"Failed to poll status for file '{file_name}': {err}"
                    raise MultimodalUploadError(msg) from err

            if current_file.state in (types.FileState.FAILED, "FAILED"):
                err_detail = "Internal processing failure"
                if current_file.error is not None:
                    err_detail = getattr(current_file.error, "message", None) or str(
                        current_file.error
                    )
                msg = f"File '{file_name}' failed processing on Gemini servers: {err_detail}"
                raise MultimodalFileFailedError(msg)

            if current_file.state not in (types.FileState.ACTIVE, "ACTIVE"):
                msg = f"File '{file_name}' ended in unexpected state: {current_file.state}"
                raise MultimodalFileFailedError(msg)

            active_files.append(current_file)

        return active_files

    async def _cleanup_files(self, files: list[types.File]) -> None:
        """Automatically delete uploaded files from Gemini servers post-extraction."""
        for file in files:
            file_name = getattr(file, "name", None)
            if not file_name:
                continue
            try:
                await self._client.aio.files.delete(name=file_name)
                logger.debug("Successfully deleted remote Gemini file: %s", file_name)
            except Exception as err:
                logger.warning(
                    "Failed to delete uploaded Gemini file '%s' post-extraction: %s",
                    file_name,
                    err,
                )

    async def _generate_with_retry(
        self,
        target_model: str,
        files: list[types.File],
        prompt: str,
    ) -> types.GenerateContentResponse:
        """Call Gemini generate_content with structured output under an exponential retry policy."""
        config = types.GenerateContentConfig(
            system_instruction=MULTIMODAL_PREPROCESS_SYSTEM_INSTRUCTION,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=DTCBatchExtractionResponse,
        )

        contents: list[Any] = [*files, prompt]

        async def _call() -> types.GenerateContentResponse:
            try:
                return await self._client.aio.models.generate_content(
                    model=target_model,
                    contents=contents,
                    config=config,
                )
            except Exception as err:
                raise self._translate_error(err) from err

        retrier = AsyncRetrying(
            retry=retry_if_exception_type((LlmRateLimitError, LlmTimeoutError)),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_random_exponential(min=self.base_delay, max=self.max_delay),
            reraise=True,
        )

        async for attempt in retrier:
            with attempt:
                return await _call()

        msg = "Retry loop exhausted without result from Gemini"
        raise LlmProviderError(msg)

    def _parse_and_validate_dtc_blocks(
        self,
        raw_response: types.GenerateContentResponse,
        time_range: TimeRange,
        start_index: int,
    ) -> list[DTCBlock]:
        """Transform raw Gemini output into strictly validated chronological DTCBlock objects."""
        raw_blocks = self._extract_raw_blocks(raw_response)

        if not raw_blocks:
            return [
                DTCBlock(
                    index=start_index,
                    time_range=time_range,
                    board_content="",
                    lector_speech="[No speech or blackboard notes detected in this interval]",
                    metadata=DTCBlockMetadata(
                        confidence=0.5,
                        frame_snapshot_path=None,
                        has_formula=False,
                    ),
                )
            ]

        dtc_blocks: list[DTCBlock] = []
        base_start = time_range.start
        batch_duration = time_range.duration
        prev_start = base_start

        for idx, item in enumerate(raw_blocks):
            # Extract start and end seconds
            start_val, end_val = self._extract_interval(item, base_start, batch_duration)

            # Guarantee strict monotonicity and minimum interval duration
            if start_val < prev_start:
                start_val = prev_start
            if end_val <= start_val:
                end_val = round(start_val + 1.0, 3)

            board_text = getattr(item, "board_content", "").strip()
            speech_text = getattr(item, "lector_speech", "").strip()
            if not board_text and not speech_text:
                speech_text = "[No audible speech or blackboard writing in window]"

            has_formula = bool(getattr(getattr(item, "metadata", None), "has_formula", False))
            if not has_formula:
                has_formula = bool(getattr(item, "has_formula", False))
            if not has_formula:
                has_formula = bool(
                    re.search(
                        r"[\$\\_{}\^]|\\(?:frac|lim|sum|int|sqrt|alpha|beta|gamma)", board_text
                    )
                )

            confidence = 0.95
            meta_obj = getattr(item, "metadata", None)
            if meta_obj is not None and hasattr(meta_obj, "confidence"):
                confidence = float(meta_obj.confidence)
            elif hasattr(item, "confidence"):
                confidence = float(item.confidence)

            block = DTCBlock(
                index=start_index + idx,
                time_range=TimeRange(start=start_val, end=end_val),
                board_content=board_text,
                lector_speech=speech_text,
                metadata=DTCBlockMetadata(
                    confidence=confidence,
                    frame_snapshot_path=None,
                    has_formula=has_formula,
                ),
            )
            dtc_blocks.append(block)
            prev_start = start_val

        return dtc_blocks

    @staticmethod
    def _extract_raw_blocks(response: types.GenerateContentResponse) -> list[Any]:
        """Extract structured block containers from GenerateContentResponse."""
        if isinstance(response.parsed, DTCBatchExtractionResponse):
            return list(response.parsed.blocks)

        if isinstance(response.parsed, list):
            return list(response.parsed)

        raw_text = response.text or ""
        if not raw_text.strip():
            msg = "Gemini returned empty response text during multimodal batch extraction."
            raise MultimodalExtractionParseError(msg)

        try:
            parsed_batch = DTCBatchExtractionResponse.model_validate_json(raw_text)
            return list(parsed_batch.blocks)
        except Exception:
            pass

        try:
            return list(TypeAdapter(list[DTCBlock]).validate_json(raw_text))
        except Exception:
            pass

        try:
            data = json.loads(raw_text)
            if isinstance(data, dict):
                if "blocks" in data and isinstance(data["blocks"], list):
                    return data["blocks"]
                if "chunks" in data and isinstance(data["chunks"], list):
                    return data["chunks"]
            elif isinstance(data, list):
                return data
        except Exception as err:
            msg = f"Failed to parse Gemini output into structured DTC blocks: {err}"
            raise MultimodalExtractionParseError(msg) from err

        msg = f"Unrecognized JSON structure from Gemini: {raw_text[:200]}"
        raise MultimodalExtractionParseError(msg)

    @staticmethod
    def _extract_interval(
        item: Any, base_start: float, batch_duration: float
    ) -> tuple[float, float]:
        """Safely extract and normalize start and end seconds for a candidate block."""
        if isinstance(item, DTCBlock):
            return item.time_range.start, item.time_range.end

        if hasattr(item, "time_range") and item.time_range is not None:
            tr = item.time_range
            start = getattr(tr, "start", 0.0) if hasattr(tr, "start") else tr.get("start", 0.0)
            end = getattr(tr, "end", 1.0) if hasattr(tr, "end") else tr.get("end", 1.0)
            return float(start), float(end)

        if isinstance(item, dict):
            if "time_range" in item and isinstance(item["time_range"], dict):
                return float(item["time_range"].get("start", 0.0)), float(
                    item["time_range"].get("end", 1.0)
                )

            start_off = float(item.get("start_offset", item.get("start", 0.0)))
            end_off = float(item.get("end_offset", item.get("end", 1.0)))

            if start_off >= base_start and base_start > 0:
                return round(start_off, 3), round(end_off, 3)

            return (
                round(base_start + min(max(start_off, 0.0), batch_duration), 3),
                round(base_start + min(max(end_off, 0.0), batch_duration), 3),
            )

        start_raw = getattr(item, "start_offset", getattr(item, "start", 0.0))
        end_raw = getattr(item, "end_offset", getattr(item, "end", 1.0))
        return (
            round(base_start + min(max(float(start_raw), 0.0), batch_duration), 3),
            round(base_start + min(max(float(end_raw), 0.0), batch_duration), 3),
        )

    @staticmethod
    def _build_extraction_prompt(
        time_range: TimeRange,
        batch_index: int,
        batch_duration: float,
        has_audio: bool,
        frame_count: int,
    ) -> str:
        """Construct the extraction prompt contextualized with batch timing and modalities."""
        return (
            f"Analyze the attached multimodal batch #{batch_index} corresponding to lecture interval "
            f"{time_range.formatted_range} (batch duration: {batch_duration:.1f} seconds).\n"
            f"Included modalities: Audio present: {has_audio}, Sampled visual frames: {frame_count}.\n\n"
            "Extract all blackboard LaTeX equations and synchronized spoken explanations into chronological blocks. "
            "For each block, provide valid index, time_range with start and end seconds, board_content, "
            "lector_speech, and metadata with confidence and has_formula flag."
        )

    @staticmethod
    def _translate_error(err: Exception) -> LlmError:
        """Translate upstream Gemini API errors into standard domain exceptions."""
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
                return LlmTimeoutError(f"Gemini server unavailable ({status_code}): {err_msg}")
            return LlmProviderError(f"Gemini API error ({status_code}): {err_msg}")

        if isinstance(err, TimeoutError):
            return LlmTimeoutError(f"Gemini client timeout: {err}")

        return LlmProviderError(f"Unexpected Gemini error: {err}")
