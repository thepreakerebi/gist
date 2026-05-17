from pathlib import Path

from gist.core.modes import AudioScoringMode, VisualScoringMode
from gist.core.presets import CompressionPreset
from gist.media.models import AudioWindow, ExtractedFrame, IngestedVideo, VideoMetadata
from gist.pipeline import LocalCompressionPipeline


class FakeIngestor:
    def __init__(self) -> None:
        self.calls = 0

    def ingest(
        self,
        video_path: Path,
        sample_count: int,
        audio_window_seconds: float,
    ) -> IngestedVideo:
        self.calls += 1
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
                    duration_seconds=audio_window_seconds,
                    path=Path("audio-0.wav"),
                )
            ],
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
        decompose_query=False,
    )

    assert ingestion.video_id == "video-1"
    assert compression.video_id == "video-1"
    assert compression.metrics.input_candidates == 3
    assert compression.metrics.selected_candidates == 3


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
            decompose_query=False,
        )

    assert ingestor.calls == 1
    assert candidate_generator.calls == 1
