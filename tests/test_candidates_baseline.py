from pathlib import Path

from gist.candidates.baseline import BaselineCandidateGenerator
from gist.media.models import AudioWindow, ExtractedFrame, IngestedVideo, VideoMetadata


class FakeVisualScorer:
    def score_frames(self, frames: list[ExtractedFrame], query: str) -> dict[Path, float]:
        return {frame.path: 0.9 for frame in frames}


class FakeAudioTranscriber:
    def transcribe_windows(self, windows: list[AudioWindow]) -> dict[Path, str]:
        return {window.path: "speaker explains pricing" for window in windows}


class FakeAudioScorer:
    def score_windows(self, windows: list[AudioWindow], query: str) -> dict[Path, float]:
        return {window.path: 0.8 for window in windows}


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
                start_seconds=2.0,
                duration_seconds=4.0,
                path=Path("audio.wav"),
            ),
        ],
    )

    candidates = BaselineCandidateGenerator().generate(manifest, query="anything")

    assert candidates.visual[0].id == "video-1:visual:0"
    assert candidates.visual[0].timestamp_seconds == 0.0
    assert candidates.audio[0].id == "video-1:audio:0"
    assert candidates.audio[0].timestamp_seconds == 4.0


def test_baseline_candidate_generator_can_attach_visual_saliency_scores() -> None:
    manifest = IngestedVideo(
        video_id="video-1",
        source_path=Path("video.mp4"),
        metadata=VideoMetadata(duration_seconds=2.0, has_audio=False),
        frames=[
            ExtractedFrame(index=0, timestamp_seconds=0.0, path=Path("frame.jpg")),
        ],
        audio_windows=[],
    )

    candidates = BaselineCandidateGenerator(visual_scorer=FakeVisualScorer()).generate(
        manifest,
        query="speaker",
    )

    assert candidates.visual[0].saliency_score == 0.9


def test_baseline_candidate_generator_can_attach_audio_transcripts() -> None:
    manifest = IngestedVideo(
        video_id="video-1",
        source_path=Path("video.mp4"),
        metadata=VideoMetadata(duration_seconds=2.0, has_audio=True),
        frames=[],
        audio_windows=[
            AudioWindow(
                index=0,
                start_seconds=0.0,
                duration_seconds=1.0,
                path=Path("audio.wav"),
            ),
        ],
    )

    candidates = BaselineCandidateGenerator(audio_transcriber=FakeAudioTranscriber()).generate(
        manifest,
        query="pricing",
    )

    assert candidates.audio[0].text == "speaker explains pricing"


def test_baseline_candidate_generator_can_attach_audio_saliency_scores() -> None:
    manifest = IngestedVideo(
        video_id="video-1",
        source_path=Path("video.mp4"),
        metadata=VideoMetadata(duration_seconds=2.0, has_audio=True),
        frames=[],
        audio_windows=[
            AudioWindow(
                index=0,
                start_seconds=0.0,
                duration_seconds=1.0,
                path=Path("audio.wav"),
            ),
        ],
    )

    candidates = BaselineCandidateGenerator(audio_scorer=FakeAudioScorer()).generate(
        manifest,
        query="applause",
    )

    assert candidates.audio[0].saliency_score == 0.8
