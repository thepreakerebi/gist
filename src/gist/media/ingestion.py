import hashlib
from pathlib import Path

from gist.media.ffmpeg import FfmpegMediaProcessor
from gist.media.longform import ProcessingMode, plan_ingestion
from gist.media.models import IngestedVideo, IngestionSettings


class MediaIngestor:
    def __init__(
        self,
        output_root: Path,
        processor: FfmpegMediaProcessor | None = None,
    ) -> None:
        self.output_root = output_root
        self.processor = processor or FfmpegMediaProcessor()

    def ingest(
        self,
        video_path: Path,
        sample_count: int | None = 128,
        audio_window_seconds: float | None = 1.0,
        processing_mode: ProcessingMode = ProcessingMode.SHORT,
    ) -> IngestedVideo:
        video_id = stable_video_id(video_path)
        ingest_dir = self.output_root / video_id
        frames_dir = ingest_dir / "frames"
        audio_dir = ingest_dir / "audio"

        metadata = self.processor.probe(video_path)
        plan = plan_ingestion(
            duration_seconds=metadata.duration_seconds,
            mode=processing_mode,
            sample_count=sample_count,
            audio_window_seconds=audio_window_seconds,
        )
        frames = self.processor.extract_frames(
            video_path=video_path,
            output_dir=frames_dir,
            sample_count=plan.sample_count,
        )
        audio_windows = self.processor.extract_audio_windows(
            video_path=video_path,
            output_dir=audio_dir,
            window_seconds=plan.audio_window_seconds,
        )

        return IngestedVideo(
            video_id=video_id,
            source_path=video_path,
            metadata=metadata,
            frames=frames,
            audio_windows=audio_windows,
            settings=IngestionSettings(
                processing_mode=plan.mode.value,
                sample_count=plan.sample_count,
                audio_window_seconds=plan.audio_window_seconds,
                audio_context_window_count=plan.audio_context_window_count,
                max_audio_windows=plan.max_audio_windows,
                reason=plan.reason,
            ),
        )


def stable_video_id(video_path: Path) -> str:
    absolute_path = video_path.expanduser().resolve(strict=False)
    digest = hashlib.sha256(str(absolute_path).encode("utf-8")).hexdigest()
    return digest[:16]
