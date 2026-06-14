from pathlib import Path

from gist.core.modes import AudioScoringMode, VisualScoringMode
from gist.core.presets import CompressionPreset
from gist.core.schemas import Candidate
from gist.media.longform import ProcessingMode
from gist.media.models import (
    AudioWindow,
    ExtractedFrame,
    IngestedVideo,
    IngestionSettings,
    VideoMetadata,
)
from gist.pipeline import LocalCompressionPipeline, resolve_audio_scorer


class FakeIngestor:
    def __init__(self) -> None:
        self.calls = 0

    def ingest(
        self,
        video_path: Path,
        sample_count: int | None,
        audio_window_seconds: float | None,
        processing_mode: ProcessingMode = ProcessingMode.SHORT,
    ) -> IngestedVideo:
        self.calls += 1
        resolved_sample_count = sample_count or 2
        resolved_audio_window_seconds = audio_window_seconds or 10.0
        return IngestedVideo(
            video_id="video-1",
            source_path=video_path,
            metadata=VideoMetadata(duration_seconds=3.0, has_audio=True),
            frames=[
                ExtractedFrame(index=0, timestamp_seconds=0.0, path=Path("frame-0.jpg")),
                ExtractedFrame(index=1, timestamp_seconds=1.0, path=Path("frame-1.jpg")),
            ],
            audio_windows=[
                AudioWindow(
                    index=0,
                    start_seconds=0.0,
                    duration_seconds=resolved_audio_window_seconds,
                    path=Path("audio-0.wav"),
                )
            ],
            settings=IngestionSettings(
                processing_mode=processing_mode.value,
                sample_count=resolved_sample_count,
                audio_window_seconds=resolved_audio_window_seconds,
                audio_context_window_count=0
                if processing_mode == ProcessingMode.LONG
                else 1,
                max_audio_windows=1,
                reason="test settings",
            ),
        )


class CountingCandidateGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, ingested_video: IngestedVideo, query: str):
        from gist.candidates.baseline import BaselineCandidateGenerator

        self.calls += 1
        return BaselineCandidateGenerator().generate(ingested_video, query)


def test_local_pipeline_ingests_generates_candidates_and_compresses(tmp_path: Path) -> None:
    pipeline = LocalCompressionPipeline(output_root=tmp_path, ingestor=FakeIngestor())

    ingestion, compression = pipeline.run(
        video_path=tmp_path / "video.mp4",
        query="audio",
        preset=CompressionPreset.BALANCED,
        sample_count=2,
        audio_window_seconds=1.0,
        visual_scorer=VisualScoringMode.BASELINE,
        audio_scorer=AudioScoringMode.BASELINE,
        adaptive_budget=False,
        decompose_query=False,
    )

    assert ingestion.video_id == "video-1"
    assert compression.video_id == "video-1"
    assert compression.metrics.input_candidates == 3
    assert compression.metrics.raw_input_candidates == 3
    assert compression.metrics.selected_candidates == 3
    assert compression.audio_scorer_used == AudioScoringMode.BASELINE


def test_auto_audio_scorer_routes_long_speech_queries_to_whisper() -> None:
    assert (
        resolve_audio_scorer(
            requested=AudioScoringMode.AUTO,
            query="What does the speaker say about pricing?",
            duration_seconds=3600,
            whisper_available=True,
        )
        == AudioScoringMode.WHISPER
    )


def test_auto_audio_scorer_keeps_short_or_non_speech_queries_on_baseline() -> None:
    assert (
        resolve_audio_scorer(
            requested=AudioScoringMode.AUTO,
            query="What does the speaker say about pricing?",
            duration_seconds=120,
            whisper_available=True,
        )
        == AudioScoringMode.BASELINE
    )
    assert (
        resolve_audio_scorer(
            requested=AudioScoringMode.AUTO,
            query="Show the red robot hand on screen",
            duration_seconds=3600,
            whisper_available=True,
        )
        == AudioScoringMode.BASELINE
    )


def test_auto_audio_scorer_falls_back_when_whisper_is_unavailable() -> None:
    assert (
        resolve_audio_scorer(
            requested=AudioScoringMode.AUTO,
            query="What does the speaker say about pricing?",
            duration_seconds=3600,
            whisper_available=False,
        )
        == AudioScoringMode.BASELINE
    )


def test_explicit_audio_scorer_bypasses_auto_routing() -> None:
    assert (
        resolve_audio_scorer(
            requested=AudioScoringMode.CLAP,
            query="What does the speaker say about pricing?",
            duration_seconds=3600,
        )
        == AudioScoringMode.CLAP
    )


def test_local_pipeline_reuses_disk_cache_on_repeated_runs(tmp_path: Path) -> None:
    ingestor = FakeIngestor()
    candidate_generator = CountingCandidateGenerator()
    pipeline = LocalCompressionPipeline(
        output_root=tmp_path,
        ingestor=ingestor,
        candidate_generator=candidate_generator,
    )

    for _ in range(2):
        pipeline.run(
            video_path=tmp_path / "video.mp4",
            query="audio",
            preset=CompressionPreset.BALANCED,
            sample_count=2,
            audio_window_seconds=1.0,
            visual_scorer=VisualScoringMode.BASELINE,
            audio_scorer=AudioScoringMode.BASELINE,
            adaptive_budget=False,
            decompose_query=False,
        )

    assert ingestor.calls == 1
    assert candidate_generator.calls == 1


def test_local_pipeline_shortlists_longform_candidates_before_compression(tmp_path: Path) -> None:
    class LongIngestor(FakeIngestor):
        def ingest(
            self,
            video_path: Path,
            sample_count: int | None,
            audio_window_seconds: float | None,
            processing_mode: ProcessingMode = ProcessingMode.LONG,
        ) -> IngestedVideo:
            ingested = super().ingest(
                video_path=video_path,
                sample_count=sample_count,
                audio_window_seconds=audio_window_seconds,
                processing_mode=ProcessingMode.LONG,
            )
            return ingested.model_copy(
                update={
                    "metadata": VideoMetadata(duration_seconds=90 * 60, has_audio=True),
                    "frames": [
                        ExtractedFrame(
                            index=index,
                            timestamp_seconds=float(index * 250),
                            path=Path(f"frame-{index}.jpg"),
                        )
                        for index in range(20)
                    ],
                    "audio_windows": [
                        AudioWindow(
                            index=index,
                            start_seconds=float(index * 250),
                            duration_seconds=60,
                            path=Path(f"audio-{index}.wav"),
                        )
                        for index in range(20)
                    ],
                }
            )

    class LongCandidateGenerator:
        def generate(self, ingested_video: IngestedVideo, query: str):
            from gist.candidates.baseline import CandidateSet

            return CandidateSet(
                visual=[
                    Candidate(
                        id=f"v-{index}",
                        timestamp_seconds=frame.timestamp_seconds,
                        text="refund policy" if index == 10 else "unrelated",
                        asset_path=frame.path,
                    )
                    for index, frame in enumerate(ingested_video.frames)
                ],
                audio=[
                    Candidate(
                        id=f"a-{index}",
                        timestamp_seconds=window.start_seconds,
                        text="refund policy" if index == 10 else "unrelated",
                        asset_path=window.path,
                    )
                    for index, window in enumerate(ingested_video.audio_windows)
                ],
            )

    pipeline = LocalCompressionPipeline(
        output_root=tmp_path,
        ingestor=LongIngestor(),
        candidate_generator=LongCandidateGenerator(),
    )

    _ingestion, compression = pipeline.run(
        video_path=tmp_path / "long.mp4",
        query="refund policy",
        processing_mode=ProcessingMode.LONG,
        sample_count=None,
        audio_window_seconds=None,
    )

    assert compression.metrics.input_candidates < 40
    assert compression.metrics.raw_input_candidates == 40
    assert compression.metrics.fused_input_candidates == compression.metrics.input_candidates
    assert "a-10+v-10" in {item.id for item in compression.selected}
