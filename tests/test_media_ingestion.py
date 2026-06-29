from pathlib import Path

from gist.media.ingestion import MediaIngestor, stable_video_id
from gist.media.longform import ProcessingMode
from gist.media.models import AudioWindow, ExtractedFrame, IngestedVideo, VideoMetadata


class FakeProcessor:
    def probe(self, video_path: Path) -> VideoMetadata:
        return VideoMetadata(duration_seconds=2.0, width=320, height=240, has_audio=True)

    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        sample_count: int,
    ) -> list[ExtractedFrame]:
        output_dir.mkdir(parents=True, exist_ok=True)
        return [
            ExtractedFrame(
                index=index,
                timestamp_seconds=float(index),
                path=output_dir / f"{index}.jpg",
            )
            for index in range(sample_count)
        ]

    def extract_audio_windows(
        self,
        video_path: Path,
        output_dir: Path,
        window_seconds: float,
    ) -> list[AudioWindow]:
        output_dir.mkdir(parents=True, exist_ok=True)
        return [
            AudioWindow(
                index=0,
                start_seconds=0.0,
                duration_seconds=window_seconds,
                path=output_dir / "0.wav",
            )
        ]


def test_stable_video_id_is_deterministic(tmp_path: Path) -> None:
    video_path = tmp_path / "demo.mp4"

    assert stable_video_id(video_path) == stable_video_id(video_path)


def test_media_ingestor_returns_structured_manifest(tmp_path: Path) -> None:
    video_path = tmp_path / "demo.mp4"
    ingestor = MediaIngestor(output_root=tmp_path / "ingested", processor=FakeProcessor())

    manifest = ingestor.ingest(video_path, sample_count=3, audio_window_seconds=0.5)

    assert isinstance(manifest, IngestedVideo)
    assert manifest.video_id == stable_video_id(video_path)
    assert len(manifest.frames) == 3
    assert len(manifest.audio_windows) == 1
    assert manifest.frames[0].path.parent.name == "frames"
    assert manifest.audio_windows[0].path.parent.name == "audio"
    assert manifest.settings is not None
    assert manifest.settings.sample_count == 3
    assert manifest.settings.audio_window_seconds == 0.5


def test_media_ingestor_separates_artifacts_by_sampling_plan(tmp_path: Path) -> None:
    video_path = tmp_path / "demo.mp4"
    ingestor = MediaIngestor(output_root=tmp_path / "ingested", processor=FakeProcessor())

    sparse = ingestor.ingest(video_path, sample_count=3, audio_window_seconds=0.5)
    dense = ingestor.ingest(video_path, sample_count=5, audio_window_seconds=0.5)

    assert sparse.frames[0].path != dense.frames[0].path
    assert sparse.frames[0].path.parent.name == "frames"
    assert dense.frames[0].path.parent.name == "frames"
    assert sparse.frames[0].path.parent.parent != dense.frames[0].path.parent.parent


def test_media_ingestor_auto_long_mode_bounds_audio_windows(tmp_path: Path) -> None:
    class LongFakeProcessor(FakeProcessor):
        def probe(self, video_path: Path) -> VideoMetadata:
            return VideoMetadata(duration_seconds=90 * 60, width=320, height=240, has_audio=True)

        def extract_audio_windows(
            self,
            video_path: Path,
            output_dir: Path,
            window_seconds: float,
        ) -> list[AudioWindow]:
            output_dir.mkdir(parents=True, exist_ok=True)
            return [
                AudioWindow(
                    index=index,
                    start_seconds=index * window_seconds,
                    duration_seconds=window_seconds,
                    path=output_dir / f"{index}.wav",
                )
                for index in range(int((90 * 60) / window_seconds))
            ]

    ingestor = MediaIngestor(output_root=tmp_path / "ingested", processor=LongFakeProcessor())

    manifest = ingestor.ingest(
        tmp_path / "long.mp4",
        sample_count=None,
        audio_window_seconds=None,
        processing_mode=ProcessingMode.AUTO,
    )

    assert manifest.settings is not None
    assert manifest.settings.processing_mode == "long"
    assert manifest.settings.sample_count == 512
    assert manifest.settings.audio_window_seconds >= 30.0
    assert len(manifest.audio_windows) <= 240
