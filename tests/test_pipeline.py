from pathlib import Path

from gist.core.modes import AudioScoringMode, VisualScoringMode
from gist.core.presets import CompressionPreset
from gist.media.models import AudioWindow, ExtractedFrame, IngestedVideo, VideoMetadata
from gist.pipeline import LocalCompressionPipeline


class FakeIngestor:
    def ingest(
        self,
        video_path: Path,
        sample_count: int,
        audio_window_seconds: float,
    ) -> IngestedVideo:
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
    )

    assert ingestion.video_id == "video-1"
    assert compression.video_id == "video-1"
    assert compression.metrics.input_candidates == 3
    assert compression.metrics.selected_candidates == 3
