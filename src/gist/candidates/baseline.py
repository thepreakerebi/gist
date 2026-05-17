from pathlib import Path

from pydantic import BaseModel, ConfigDict

from gist.audio.transcribers import AudioTranscriber
from gist.core.schemas import Candidate
from gist.media.models import AudioWindow, ExtractedFrame, IngestedVideo
from gist.vision.scorers import VisualFrameScorer


class CandidateSet(BaseModel):
    model_config = ConfigDict(frozen=True)

    visual: list[Candidate]
    audio: list[Candidate]


class BaselineCandidateGenerator:
    """Build deterministic timestamp candidates before model-based scoring is available."""

    def __init__(
        self,
        visual_scorer: VisualFrameScorer | None = None,
        audio_transcriber: AudioTranscriber | None = None,
    ) -> None:
        self.visual_scorer = visual_scorer
        self.audio_transcriber = audio_transcriber

    def generate(self, ingested_video: IngestedVideo, query: str) -> CandidateSet:
        visual_scores = self._score_visual_frames(ingested_video, query)
        audio_transcripts = self._transcribe_audio_windows(ingested_video)
        return CandidateSet(
            visual=[
                self._visual_candidate(
                    ingested_video.video_id,
                    frame,
                    visual_scores.get(frame.path),
                )
                for frame in ingested_video.frames
            ],
            audio=[
                self._audio_candidate(
                    ingested_video.video_id,
                    window,
                    audio_transcripts.get(window.path),
                )
                for window in ingested_video.audio_windows
            ],
        )

    def _score_visual_frames(
        self,
        ingested_video: IngestedVideo,
        query: str,
    ) -> dict[Path, float]:
        if self.visual_scorer is None:
            return {}
        return self.visual_scorer.score_frames(ingested_video.frames, query=query)

    def _transcribe_audio_windows(self, ingested_video: IngestedVideo) -> dict[Path, str]:
        if self.audio_transcriber is None:
            return {}
        return self.audio_transcriber.transcribe_windows(ingested_video.audio_windows)

    def _visual_candidate(
        self,
        video_id: str,
        frame: ExtractedFrame,
        saliency_score: float | None,
    ) -> Candidate:
        return Candidate(
            id=f"{video_id}:visual:{frame.index}",
            timestamp_seconds=frame.timestamp_seconds,
            text=f"visual frame sampled at {frame.timestamp_seconds:.2f} seconds",
            saliency_score=saliency_score,
        )

    def _audio_candidate(
        self,
        video_id: str,
        window: AudioWindow,
        transcript: str | None,
    ) -> Candidate:
        end_seconds = window.start_seconds + window.duration_seconds
        text = transcript or (
            "audio window "
            f"from {window.start_seconds:.2f} to {end_seconds:.2f} seconds"
        )
        return Candidate(
            id=f"{video_id}:audio:{window.index}",
            timestamp_seconds=window.start_seconds,
            text=text,
        )
