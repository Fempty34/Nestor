"""End-to-end lecture video preprocessing contour.

Orchestrates audio normalization, video frame sampling, sequential 5-minute
window multimodal extraction, chunk aggregation into canonical LectureDTC,
and guaranteed cleanup of ephemeral media artifacts.
"""

import asyncio
import logging
import shutil
import tempfile
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from nestor.domain.contracts.dtc import (
    CourseMetadata,
    DTCBlock,
    DTCBlockMetadata,
    DTCLecture,
    DTCSequence,
    LectureDTC,
    TimeRange,
)
from nestor.ports.audio import AudioProcessingResult, AudioProcessor
from nestor.ports.multimodal import MultimodalBatchExtractor
from nestor.ports.video import VideoProcessingResult, VideoProcessor

logger = logging.getLogger(__name__)

__all__ = [
    "DTCLecture",
    "DTCSequence",
    "LecturePreprocessor",
    "PreprocessingResult",
    "PreprocessingTelemetry",
    "process_lecture",
]


class PreprocessingTelemetry(BaseModel):
    """Structured performance metrics and execution telemetry for the preprocessing contour."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_duration_seconds: float = Field(
        ge=0.0, description="Total wall-clock duration of the preprocessing contour in seconds."
    )
    audio_stage_duration_seconds: float = Field(
        ge=0.0, description="Duration in seconds of audio extraction and normalization."
    )
    video_stage_duration_seconds: float = Field(
        ge=0.0, description="Duration in seconds of video frame sampling and batch segmentation."
    )
    extraction_stage_duration_seconds: float = Field(
        ge=0.0, description="Duration in seconds of sequential multimodal LLM batch extraction."
    )
    processed_frame_count: int = Field(
        ge=0, description="Total count of sampled video frames ingested."
    )
    audio_segment_count: int = Field(
        ge=0, description="Total count of extracted 5-minute audio segments."
    )
    batch_count: int = Field(
        ge=0, description="Total count of multimodal 5-minute processing windows dispatched."
    )
    extracted_block_count: int = Field(
        ge=0, description="Total number of validated DTC blocks assembled in the final lecture."
    )


class PreprocessingResult(BaseModel):
    """Container holding the assembled canonical LectureDTC and structured execution telemetry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lecture: LectureDTC = Field(description="Assembled and validated canonical lecture DTC model.")
    telemetry: PreprocessingTelemetry = Field(
        description="Structured execution telemetry and timing metrics."
    )


class LecturePreprocessor:
    """Orchestrates audio, video, and multimodal extraction stages into a unified contour."""

    def __init__(
        self,
        audio_processor: AudioProcessor | None = None,
        video_processor: VideoProcessor | None = None,
        multimodal_extractor: MultimodalBatchExtractor | None = None,
        *,
        fps: float = 0.5,
        batch_duration: float = 300.0,
        default_course_metadata: CourseMetadata | None = None,
    ) -> None:
        self._audio_processor = audio_processor
        self._video_processor = video_processor
        self._multimodal_extractor = multimodal_extractor
        self._fps = fps
        self._batch_duration = batch_duration
        self._default_course_metadata = default_course_metadata

    async def process(
        self,
        video_path: Path,
        *,
        course_metadata: CourseMetadata | None = None,
        output_workdir: Path | None = None,
    ) -> PreprocessingResult:
        """Execute end-to-end preprocessing contour on the provided lecture video.

        Args:
            video_path: Filesystem path to the input lecture video.
            course_metadata: Optional institutional metadata; auto-generated if omitted.
            output_workdir: Optional caller-managed working directory. If None, an isolated
                ephemeral temporary directory is created and guaranteed to be deleted.

        Returns:
            PreprocessingResult containing canonical LectureDTC and execution telemetry.
        """
        video_file = Path(video_path)
        if not video_file.is_file():
            msg = f"Input lecture video file not found at '{video_file}'"
            raise FileNotFoundError(msg)

        meta = (
            course_metadata
            or self._default_course_metadata
            or self._generate_default_metadata(video_file)
        )

        audio_proc = self._audio_processor or self._resolve_default_audio_processor()
        video_proc = self._video_processor or self._resolve_default_video_processor()
        extractor = self._multimodal_extractor or self._resolve_default_multimodal_extractor()

        overall_start = time.perf_counter()
        audio_duration = 0.0
        video_duration = 0.0

        is_ephemeral = output_workdir is None
        temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None

        if is_ephemeral:
            temp_dir_obj = tempfile.TemporaryDirectory(prefix="nestor_preprocess_")
            root_dir = Path(temp_dir_obj.name)
        else:
            root_dir = Path(output_workdir)  # type: ignore[arg-type]
            root_dir.mkdir(parents=True, exist_ok=True)

        try:
            audio_workdir = root_dir / "audio"
            video_workdir = root_dir / "video"
            audio_workdir.mkdir(parents=True, exist_ok=True)
            video_workdir.mkdir(parents=True, exist_ok=True)

            logger.info(
                "Starting preprocessing contour for '%s' in temporary workspace '%s'",
                video_file.name,
                root_dir,
            )

            async def _run_audio() -> AudioProcessingResult:
                nonlocal audio_duration
                t_start = time.perf_counter()
                res = await audio_proc.process_lecture_audio(
                    video_file, audio_workdir, segment_duration=self._batch_duration
                )
                audio_duration = round(time.perf_counter() - t_start, 3)
                logger.info(
                    "Audio processing stage finished in %.2fs (%d segments)",
                    audio_duration,
                    len(res.segments),
                )
                return res

            async def _run_video() -> VideoProcessingResult:
                nonlocal video_duration
                t_start = time.perf_counter()
                res = await video_proc.extract_frames(
                    video_file, video_workdir, fps=self._fps, batch_duration=self._batch_duration
                )
                video_duration = round(time.perf_counter() - t_start, 3)
                logger.info(
                    "Video processing stage finished in %.2fs (%d frames, %d batches)",
                    video_duration,
                    res.total_frames,
                    len(res.batches),
                )
                return res

            audio_result, video_result = await asyncio.gather(_run_audio(), _run_video())

            t_extract_start = time.perf_counter()
            num_windows = max(len(audio_result.segments), len(video_result.batches))
            if num_windows == 0:
                msg = f"No audio segments or video batches produced for media '{video_file}'."
                raise ValueError(msg)

            all_blocks: list[DTCBlock] = []
            current_block_index = 0
            last_block_start = 0.0

            for batch_idx in range(num_windows):
                audio_seg = (
                    audio_result.segments[batch_idx]
                    if batch_idx < len(audio_result.segments)
                    else None
                )
                video_batch = (
                    video_result.batches[batch_idx]
                    if batch_idx < len(video_result.batches)
                    else None
                )

                if video_batch is not None:
                    time_range = video_batch.time_range
                elif audio_seg is not None:
                    time_range = audio_seg.time_range
                else:
                    time_range = TimeRange(
                        start=float(batch_idx * self._batch_duration),
                        end=float((batch_idx + 1) * self._batch_duration),
                    )

                audio_file = audio_seg.path if audio_seg is not None else None
                frame_files = (
                    [f.path for f in video_batch.frames] if video_batch is not None else []
                )

                logger.debug(
                    "Dispatching multimodal batch #%d (%s, %d frames, audio=%s)",
                    batch_idx,
                    time_range.formatted_range,
                    len(frame_files),
                    audio_file is not None,
                )

                batch_blocks = await extractor.extract_batch(
                    audio_path=audio_file,
                    frame_paths=frame_files,
                    time_range=time_range,
                    start_index=current_block_index,
                    batch_index=batch_idx,
                )

                for b_offset, block in enumerate(batch_blocks):
                    expected_idx = current_block_index + b_offset
                    clamped_start = max(block.time_range.start, last_block_start)
                    clamped_end = max(block.time_range.end, round(clamped_start + 0.1, 3))

                    fixed_block = DTCBlock(
                        index=expected_idx,
                        time_range=TimeRange(start=clamped_start, end=clamped_end),
                        board_content=block.board_content,
                        lector_speech=block.lector_speech,
                        metadata=block.metadata,
                    )
                    all_blocks.append(fixed_block)
                    last_block_start = clamped_start

                current_block_index += len(batch_blocks)

            extract_duration = round(time.perf_counter() - t_extract_start, 3)

            if not all_blocks:
                max_duration = max(
                    audio_result.total_duration,
                    video_result.total_duration,
                    1.0,
                )
                all_blocks.append(
                    DTCBlock(
                        index=0,
                        time_range=TimeRange(start=0.0, end=round(max_duration, 3)),
                        board_content="",
                        lector_speech="[No spoken explanations or blackboard notes detected in lecture]",
                        metadata=DTCBlockMetadata(
                            confidence=0.5,
                            frame_snapshot_path=None,
                            has_formula=False,
                        ),
                    )
                )

            lecture = LectureDTC(
                course_metadata=meta,
                blocks=all_blocks,
            )

            total_duration = round(time.perf_counter() - overall_start, 3)
            telemetry = PreprocessingTelemetry(
                total_duration_seconds=total_duration,
                audio_stage_duration_seconds=audio_duration,
                video_stage_duration_seconds=video_duration,
                extraction_stage_duration_seconds=extract_duration,
                processed_frame_count=video_result.total_frames,
                audio_segment_count=len(audio_result.segments),
                batch_count=num_windows,
                extracted_block_count=len(all_blocks),
            )

            logger.info(
                "Preprocessing contour finished successfully for '%s' in %.2fs "
                "(frames: %d, audio segments: %d, DTC blocks: %d)",
                video_file.name,
                total_duration,
                telemetry.processed_frame_count,
                telemetry.audio_segment_count,
                telemetry.extracted_block_count,
            )

            return PreprocessingResult(
                lecture=lecture,
                telemetry=telemetry,
            )

        finally:
            if temp_dir_obj is not None:
                try:
                    temp_dir_obj.cleanup()
                    logger.debug("Successfully cleaned up ephemeral workspace at '%s'", root_dir)
                except Exception as err:
                    logger.warning("Failed to remove ephemeral workspace '%s': %s", root_dir, err)
                    shutil.rmtree(root_dir, ignore_errors=True)

    @staticmethod
    def _generate_default_metadata(video_path: Path) -> CourseMetadata:
        """Construct fallback CourseMetadata using video file name properties."""
        clean_title = video_path.stem.replace("_", " ").replace("-", " ").strip().title()
        return CourseMetadata(
            course_id="AUTO-PREPROCESSED",
            course_title=clean_title or "Untitled Lecture",
            lecture_title=clean_title or "Untitled Lecture",
            language="ru",
        )

    @staticmethod
    def _resolve_default_audio_processor() -> AudioProcessor:
        from nestor.adapters.outbound.ffmpeg.adapter import FFmpegAudioProcessor

        return FFmpegAudioProcessor()

    @staticmethod
    def _resolve_default_video_processor() -> VideoProcessor:
        from nestor.adapters.outbound.ffmpeg.video_adapter import FFmpegVideoProcessor

        return FFmpegVideoProcessor()

    @staticmethod
    def _resolve_default_multimodal_extractor() -> MultimodalBatchExtractor:
        from nestor.adapters.outbound.llm.gemini_preprocessor import GeminiMultimodalBatchExtractor

        return GeminiMultimodalBatchExtractor()


async def process_lecture(
    video_path: Path,
    *,
    course_metadata: CourseMetadata | None = None,
    audio_processor: AudioProcessor | None = None,
    video_processor: VideoProcessor | None = None,
    multimodal_extractor: MultimodalBatchExtractor | None = None,
    fps: float = 0.5,
    batch_duration: float = 300.0,
) -> DTCLecture:
    """Primary entrypoint: execute end-to-end preprocessing contour returning a canonical DTCLecture."""
    preprocessor = LecturePreprocessor(
        audio_processor=audio_processor,
        video_processor=video_processor,
        multimodal_extractor=multimodal_extractor,
        fps=fps,
        batch_duration=batch_duration,
    )
    result = await preprocessor.process(video_path, course_metadata=course_metadata)
    return result.lecture
