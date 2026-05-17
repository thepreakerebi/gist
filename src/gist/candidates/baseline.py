from dataclasses import dataclass

from gist.core.schemas import Candidate
from gist.media.models import AudioWindow, ExtractedFrame, IngestedVideo


@dataclass(frozen=True, slots=True)
class CandidateSet:
    visual: list[Candidate]
    audio: list[Candidate]


class BaselineCandidateGenerator:
    """Build deterministic timestamp candidates before model-based scoring is available."""

    def generate(self, ingested_video: IngestedVideo, query: str) -> CandidateSet:
        return CandidateSet(
            visual=[
                self._visual_candidate(ingested_video.video_id, frame)
                for frame in ingested_video.frames
            ],
            audio=[
                self._audio_candidate(ingested_video.video_id, window)
                for window in ingested_video.audio_windows
            ],
        )

    def _visual_candidate(self, video_id: str, frame: ExtractedFrame) -> Candidate:
        return Candidate(
            id=f"{video_id}:visual:{frame.index}",
            timestamp_seconds=frame.timestamp_seconds,
            text=f"visual frame sampled at {frame.timestamp_seconds:.2f} seconds",
        )

    def _audio_candidate(self, video_id: str, window: AudioWindow) -> Candidate:
        end_seconds = window.start_seconds + window.duration_seconds
        return Candidate(
            id=f"{video_id}:audio:{window.index}",
            timestamp_seconds=window.start_seconds,
            text=(
                "audio window "
                f"from {window.start_seconds:.2f} to {end_seconds:.2f} seconds"
            ),
        )

