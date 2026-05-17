from pathlib import Path

from pydantic import BaseModel, ConfigDict

from gist.audio.scorers import AudioWindowScorer
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
        audio_scorer: AudioWindowScorer | None = None,
        audio_context_window_count: int = 1,
    ) -> None:
        if audio_context_window_count < 0:
            raise ValueError("audio_context_window_count must be non-negative")
        self.visual_scorer = visual_scorer
        self.audio_transcriber = audio_transcriber
        self.audio_scorer = audio_scorer
        self.audio_context_window_count = audio_context_window_count

    def generate(self, ingested_video: IngestedVideo, query: str) -> CandidateSet:
        visual_scores = self._score_visual_frames(ingested_video, query)
        audio_transcripts = self._transcribe_audio_windows(ingested_video)
        audio_scores = self._score_audio_windows(ingested_video, query)
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
                    self._audio_transcript_context(
                        ingested_video.audio_windows,
                        audio_transcripts,
                        window.index,
                    ),
                    audio_scores.get(window.path),
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

    def _score_audio_windows(
        self,
        ingested_video: IngestedVideo,
        query: str,
    ) -> dict[Path, float]:
        if self.audio_scorer is None:
            return {}
        return self.audio_scorer.score_windows(ingested_video.audio_windows, query=query)

    def _audio_transcript_context(
        self,
        windows: list[AudioWindow],
        transcripts: dict[Path, str],
        center_index: int,
    ) -> str | None:
        if not transcripts:
            return None

        start_index = max(center_index - self.audio_context_window_count, 0)
        end_index = min(center_index + self.audio_context_window_count + 1, len(windows))
        snippets: list[str] = []
        for window in windows[start_index:end_index]:
            transcript = transcripts.get(window.path, "").strip()
            if transcript:
                snippets.append(transcript)
        return " ".join(snippets).strip() or None

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
        saliency_score: float | None,
    ) -> Candidate:
        end_seconds = window.start_seconds + window.duration_seconds
        midpoint_seconds = window.start_seconds + (window.duration_seconds / 2)
        text = transcript or (
            "audio window "
            f"from {window.start_seconds:.2f} to {end_seconds:.2f} seconds"
        )
        return Candidate(
            id=f"{video_id}:audio:{window.index}",
            timestamp_seconds=midpoint_seconds,
            text=text,
            saliency_score=saliency_score,
        )
