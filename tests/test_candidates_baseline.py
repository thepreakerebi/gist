from pathlib import Path

from gist.candidates.baseline import BaselineCandidateGenerator
from gist.media.models import AudioWindow, ExtractedFrame, IngestedVideo, VideoMetadata
from gist.vision.scene import FrameEmbedding


class FakeVisualScorer:
    def score_frames(self, frames: list[ExtractedFrame], query: str) -> dict[Path, float]:
        return {frame.path: 0.9 for frame in frames}


class FakeAudioTranscriber:
    def transcribe_windows(self, windows: list[AudioWindow]) -> dict[Path, str]:
        return {window.path: "speaker explains pricing" for window in windows}


class FakeAudioScorer:
    def score_windows(self, windows: list[AudioWindow], query: str) -> dict[Path, float]:
        return {window.path: 0.8 for window in windows}


class FakeFrameOcr:
    def extract_text(self, frames: list[ExtractedFrame]) -> dict[Path, str]:
        return {frame.path: "Conductor ships code with AI" for frame in frames}


class FakeSceneAwareVisualScorer(FakeVisualScorer):
    def embed_frames(self, frames: list[ExtractedFrame]) -> list[FrameEmbedding]:
        vectors = {
            0: (1.0, 0.0),
            1: (0.95, 0.05),
            2: (0.0, 1.0),
        }
        return [
            FrameEmbedding(
                frame_index=frame.index,
                timestamp_seconds=frame.timestamp_seconds,
                vector=vectors[frame.index],
            )
            for frame in frames
        ]


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
    assert candidates.visual[0].asset_path == Path("frame.jpg")
    assert candidates.audio[0].id == "video-1:audio:0"
    assert candidates.audio[0].timestamp_seconds == 4.0
    assert candidates.audio[0].asset_path == Path("audio.wav")


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


def test_baseline_candidate_generator_uses_frame_ocr_as_visual_text() -> None:
    manifest = IngestedVideo(
        video_id="video-1",
        source_path=Path("video.mp4"),
        metadata=VideoMetadata(duration_seconds=2.0, has_audio=False),
        frames=[
            ExtractedFrame(index=0, timestamp_seconds=5.0, path=Path("frame.jpg")),
        ],
        audio_windows=[],
    )

    candidates = BaselineCandidateGenerator(frame_ocr=FakeFrameOcr()).generate(
        manifest,
        query="Conductor AI",
    )

    assert candidates.visual[0].text == (
        "on-screen text near 5.00 seconds: Conductor ships code with AI"
    )


def test_scene_aware_candidate_generator_attaches_scene_metadata() -> None:
    manifest = IngestedVideo(
        video_id="video-1",
        source_path=Path("video.mp4"),
        metadata=VideoMetadata(duration_seconds=3.0, has_audio=False),
        frames=[
            ExtractedFrame(index=0, timestamp_seconds=0.0, path=Path("frame-0.jpg")),
            ExtractedFrame(index=1, timestamp_seconds=1.0, path=Path("frame-1.jpg")),
            ExtractedFrame(index=2, timestamp_seconds=2.0, path=Path("frame-2.jpg")),
        ],
        audio_windows=[],
    )

    candidates = BaselineCandidateGenerator(
        visual_scorer=FakeSceneAwareVisualScorer(),
        scene_aware_visuals=True,
    ).generate(manifest, query="person")

    assert candidates.visual[0].segment_id == "scene-0"
    assert candidates.visual[1].segment_id == "scene-0"
    assert candidates.visual[2].segment_id == "scene-1"
    assert candidates.visual[2].scene_start_seconds == 2.0


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


def test_baseline_candidate_generator_stitches_neighboring_audio_transcripts() -> None:
    manifest = IngestedVideo(
        video_id="video-1",
        source_path=Path("video.mp4"),
        metadata=VideoMetadata(duration_seconds=6.0, has_audio=True),
        frames=[],
        audio_windows=[
            AudioWindow(
                index=0,
                start_seconds=0.0,
                duration_seconds=2.0,
                path=Path("audio-0.wav"),
            ),
            AudioWindow(
                index=1,
                start_seconds=2.0,
                duration_seconds=2.0,
                path=Path("audio-1.wav"),
            ),
            AudioWindow(
                index=2,
                start_seconds=4.0,
                duration_seconds=2.0,
                path=Path("audio-2.wav"),
            ),
        ],
    )

    class SequentialTranscriber:
        def transcribe_windows(self, windows: list[AudioWindow]) -> dict[Path, str]:
            return {
                windows[0].path: "the architecture",
                windows[1].path: "for these missions",
                windows[2].path: "is taking shape",
            }

    candidates = BaselineCandidateGenerator(
        audio_transcriber=SequentialTranscriber()
    ).generate(manifest, query="missions")

    assert candidates.audio[1].text == (
        "the architecture for these missions is taking shape"
    )


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
