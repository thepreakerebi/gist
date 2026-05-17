from pathlib import Path

from gist.candidates.baseline import BaselineCandidateGenerator
from gist.media.models import AudioWindow, ExtractedFrame, IngestedVideo, VideoMetadata


def test_baseline_candidate_generator_maps_manifest_to_candidates() -> None:
    manifest = IngestedVideo(
        video_id="video-1",
        source_path=Path("video.mp4"),
        metadata=VideoMetadata(duration_seconds=2.0, has_audio=True),
        frames=[
            ExtractedFrame(index=0, timestamp_seconds=0.0, path=Path("frame.jpg")),
        ],
        audio_windows=[
            AudioWindow(
                index=0,
                start_seconds=0.0,
                duration_seconds=1.0,
                path=Path("audio.wav"),
            ),
        ],
    )

    candidates = BaselineCandidateGenerator().generate(manifest, query="anything")

    assert candidates.visual[0].id == "video-1:visual:0"
    assert candidates.visual[0].timestamp_seconds == 0.0
    assert candidates.audio[0].id == "video-1:audio:0"
    assert candidates.audio[0].timestamp_seconds == 0.0

