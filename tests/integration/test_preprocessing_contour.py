"""End-to-end integration tests for the lecture preprocessing contour."""

from datetime import date
from pathlib import Path

import pytest

from nestor.adapters.outbound.ffmpeg.fake import FakeAudioProcessor
from nestor.adapters.outbound.ffmpeg.fake_video import FakeVideoProcessor
from nestor.adapters.outbound.llm.fake_multimodal import FakeMultimodalBatchExtractor
from nestor.domain.contracts.dtc import (
    CourseMetadata,
    DTCBlock,
    DTCBlockMetadata,
    DTCLecture,
    LectureDTC,
    TimeRange,
)
from nestor.pipeline.preprocessing import (
    LecturePreprocessor,
    PreprocessingResult,
    process_lecture,
)
from nestor.ports.audio import AudioExecutionError
from nestor.ports.multimodal import MultimodalExtractionError


@pytest.fixture
def synthetic_video_file(tmp_path: Path) -> Path:
    """Create a minimal synthetic MP4 video container artifact for integration testing."""
    video_path = tmp_path / "math_analysis_lecture_01.mp4"
    # Minimal container header bytes
    video_path.write_bytes(
        b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00isomiso2mp41\x00\x00\x00\x08free"
    )
    return video_path


@pytest.fixture
def sample_metadata() -> CourseMetadata:
    """Fixture providing valid CourseMetadata."""
    return CourseMetadata(
        course_id="MIPT-CALC-201",
        course_title="Multivariable Calculus",
        lecture_number=1,
        lecture_title="Double and Triple Integrals",
        lecturer="Prof. Ivanov",
        recorded_date=date(2026, 9, 2),
        language="ru",
    )


class TestPreprocessingContourEndToEnd:
    """Verifies end-to-end orchestration across audio, video, and multimodal extraction."""

    @pytest.mark.asyncio
    async def test_full_pipeline_success(
        self, synthetic_video_file: Path, sample_metadata: CourseMetadata
    ) -> None:
        """Verify complete pipeline execution: parallel ingestion, sequential dispatch, DTC assembly."""
        audio_proc = FakeAudioProcessor(
            default_segment_count=2,
            default_segment_duration=300.0,
            write_disk_files=True,
        )
        video_proc = FakeVideoProcessor(
            default_batch_count=2,
            default_frames_per_batch=150,
            write_disk_files=True,
        )
        extractor = FakeMultimodalBatchExtractor()

        preprocessor = LecturePreprocessor(
            audio_processor=audio_proc,
            video_processor=video_proc,
            multimodal_extractor=extractor,
            fps=0.5,
            batch_duration=300.0,
        )

        result: PreprocessingResult = await preprocessor.process(
            synthetic_video_file,
            course_metadata=sample_metadata,
        )

        assert isinstance(result.lecture, LectureDTC)
        assert result.lecture.course_metadata == sample_metadata
        assert len(result.lecture.blocks) == 4

        for idx, block in enumerate(result.lecture.blocks):
            assert block.index == idx
            assert block.time_range.start < block.time_range.end
            if idx > 0:
                prev_block = result.lecture.blocks[idx - 1]
                assert block.time_range.start >= prev_block.time_range.start

        telemetry = result.telemetry
        assert telemetry.total_duration_seconds > 0.0
        assert telemetry.audio_stage_duration_seconds >= 0.0
        assert telemetry.video_stage_duration_seconds >= 0.0
        assert telemetry.extraction_stage_duration_seconds >= 0.0
        assert telemetry.processed_frame_count == 300  # 150 * 2
        assert telemetry.audio_segment_count == 2
        assert telemetry.batch_count == 2
        assert telemetry.extracted_block_count == 4

        assert audio_proc.call_count == 1
        assert video_proc.call_count == 1
        assert len(extractor.call_history) == 2
        assert extractor.call_history[0].batch_index == 0
        assert extractor.call_history[0].start_index == 0
        assert extractor.call_history[1].batch_index == 1
        assert extractor.call_history[1].start_index == 2

    @pytest.mark.asyncio
    async def test_process_lecture_facade_entrypoint(
        self, synthetic_video_file: Path, sample_metadata: CourseMetadata
    ) -> None:
        """Verify the main process_lecture entrypoint function returns valid DTCLecture."""
        audio_proc = FakeAudioProcessor(default_segment_count=1, default_segment_duration=120.0)
        video_proc = FakeVideoProcessor(default_batch_count=1, default_frames_per_batch=60)
        extractor = FakeMultimodalBatchExtractor()

        lecture: DTCLecture = await process_lecture(
            synthetic_video_file,
            course_metadata=sample_metadata,
            audio_processor=audio_proc,
            video_processor=video_proc,
            multimodal_extractor=extractor,
            fps=0.5,
            batch_duration=300.0,
        )

        assert isinstance(lecture, LectureDTC)
        assert lecture.course_metadata.course_title == "Multivariable Calculus"
        assert len(lecture.blocks) == 2
        assert lecture.blocks[0].index == 0
        assert lecture.blocks[1].index == 1

    @pytest.mark.asyncio
    async def test_auto_generated_metadata_when_omitted(self, synthetic_video_file: Path) -> None:
        """Verify auto-generation of CourseMetadata when caller provides none."""
        audio_proc = FakeAudioProcessor(default_segment_count=1)
        video_proc = FakeVideoProcessor(default_batch_count=1)
        extractor = FakeMultimodalBatchExtractor()

        lecture = await process_lecture(
            synthetic_video_file,
            audio_processor=audio_proc,
            video_processor=video_proc,
            multimodal_extractor=extractor,
        )

        assert lecture.course_metadata.course_id == "AUTO-PREPROCESSED"
        assert "Math Analysis Lecture 01" in lecture.course_metadata.course_title
        assert lecture.course_metadata.language == "ru"


class TestEphemeralWorkspaceCleanup:
    """Verifies that all local scratch audio/frame directories are deleted on completion or failure."""

    @pytest.mark.asyncio
    async def test_cleanup_upon_successful_completion(self, synthetic_video_file: Path) -> None:
        """Verify temporary workspace root is wiped from disk after successful run."""
        audio_proc = FakeAudioProcessor(default_segment_count=1, write_disk_files=True)
        video_proc = FakeVideoProcessor(default_batch_count=1, write_disk_files=True)
        extractor = FakeMultimodalBatchExtractor()

        preprocessor = LecturePreprocessor(
            audio_processor=audio_proc,
            video_processor=video_proc,
            multimodal_extractor=extractor,
        )

        # Intercept created workdir path
        await preprocessor.process(synthetic_video_file)

        # The audio output dir was captured in call record
        captured_audio_dir = audio_proc.calls[0].output_dir
        captured_root = captured_audio_dir.parent

        assert not captured_root.exists(), (
            f"Ephemeral directory '{captured_root}' was not cleaned up!"
        )
        assert not captured_audio_dir.exists()

    @pytest.mark.asyncio
    async def test_cleanup_upon_multimodal_extraction_failure(
        self, synthetic_video_file: Path
    ) -> None:
        """Verify temporary directories are cleaned up even when an exception is raised in LLM stage."""
        audio_proc = FakeAudioProcessor(default_segment_count=1, write_disk_files=True)
        video_proc = FakeVideoProcessor(default_batch_count=1, write_disk_files=True)
        extractor = FakeMultimodalBatchExtractor(
            exception_to_raise=MultimodalExtractionError("Gemini Files API connection reset")
        )

        preprocessor = LecturePreprocessor(
            audio_processor=audio_proc,
            video_processor=video_proc,
            multimodal_extractor=extractor,
        )

        with pytest.raises(MultimodalExtractionError, match="Gemini Files API connection reset"):
            await preprocessor.process(synthetic_video_file)

        captured_root = audio_proc.calls[0].output_dir.parent
        assert not captured_root.exists(), "Ephemeral directory persisted after multimodal failure!"

    @pytest.mark.asyncio
    async def test_cleanup_upon_audio_processing_failure(self, synthetic_video_file: Path) -> None:
        """Verify temporary directories are cleaned up when audio processing crashes."""
        audio_proc = FakeAudioProcessor(
            error_to_raise=AudioExecutionError("FFmpeg segmentation fault")
        )
        video_proc = FakeVideoProcessor()
        extractor = FakeMultimodalBatchExtractor()

        preprocessor = LecturePreprocessor(
            audio_processor=audio_proc,
            video_processor=video_proc,
            multimodal_extractor=extractor,
        )

        with pytest.raises(AudioExecutionError, match="FFmpeg segmentation fault"):
            await preprocessor.process(synthetic_video_file)

        captured_root = audio_proc.calls[0].output_dir.parent
        assert not captured_root.exists(), "Ephemeral directory persisted after audio failure!"


class TestPreprocessingEdgeCases:
    """Verifies boundary conditions, mismatched segment counts, and timestamp jitter corrections."""

    @pytest.mark.asyncio
    async def test_missing_video_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError is raised when target video does not exist."""
        preprocessor = LecturePreprocessor(
            audio_processor=FakeAudioProcessor(),
            video_processor=FakeVideoProcessor(),
            multimodal_extractor=FakeMultimodalBatchExtractor(),
        )

        with pytest.raises(FileNotFoundError, match="Input lecture video file not found"):
            await preprocessor.process(tmp_path / "nonexistent_lecture.mp4")

    @pytest.mark.asyncio
    async def test_mismatched_audio_video_counts_alignment(
        self, synthetic_video_file: Path
    ) -> None:
        """Verify alignment when audio produces 3 segments but video produces only 2 batches."""
        audio_proc = FakeAudioProcessor(default_segment_count=3, default_segment_duration=300.0)
        video_proc = FakeVideoProcessor(default_batch_count=2, default_frames_per_batch=50)
        extractor = FakeMultimodalBatchExtractor()

        preprocessor = LecturePreprocessor(
            audio_processor=audio_proc,
            video_processor=video_proc,
            multimodal_extractor=extractor,
        )

        result = await preprocessor.process(synthetic_video_file)

        assert result.telemetry.batch_count == 3
        assert len(extractor.call_history) == 3
        
        assert extractor.call_history[2].audio_path is not None
        assert len(extractor.call_history[2].frame_paths) == 0

    @pytest.mark.asyncio
    async def test_timestamp_jitter_and_monotonicity_correction(
        self, synthetic_video_file: Path
    ) -> None:
        """Verify that out-of-order or jittered timestamps from LLM are clamped to satisfy LectureDTC."""
        jittered_blocks = [
            DTCBlock(
                index=0,
                time_range=TimeRange(start=10.0, end=20.0),
                board_content=r"x^2 + y^2 = r^2",
                lector_speech="Уравнение окружности.",
                metadata=DTCBlockMetadata(confidence=0.9, has_formula=True),
            ),
            
            DTCBlock(
                index=1,
                time_range=TimeRange(start=5.0, end=15.0),
                board_content=r"\sin^2 x + \cos^2 x = 1",
                lector_speech="Основное тригонометрическое тождество.",
                metadata=DTCBlockMetadata(confidence=0.9, has_formula=True),
            ),
        ]

        extractor = FakeMultimodalBatchExtractor(default_blocks=jittered_blocks)
        preprocessor = LecturePreprocessor(
            audio_processor=FakeAudioProcessor(default_segment_count=1),
            video_processor=FakeVideoProcessor(default_batch_count=1),
            multimodal_extractor=extractor,
        )

        result = await preprocessor.process(synthetic_video_file)

        # Clamped so start >= previous start
        b0, b1 = result.lecture.blocks
        assert b0.time_range.start == 10.0
        assert b1.time_range.start >= b0.time_range.start
        assert b1.time_range.end > b1.time_range.start
