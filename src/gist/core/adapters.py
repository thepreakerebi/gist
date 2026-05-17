from typing import Protocol

from gist.core.schemas import Candidate
from gist.media.models import IngestedVideo


class VisualCandidateExtractor(Protocol):
    def extract_visual_candidates(self, ingested_video: IngestedVideo, query: str) -> list[Candidate]:
        """Return timestamped visual candidates for a video/query pair."""


class AudioCandidateExtractor(Protocol):
    def extract_audio_candidates(self, ingested_video: IngestedVideo, query: str) -> list[Candidate]:
        """Return timestamped audio or speech candidates for a video/query pair."""


class VideoLlmGateway(Protocol):
    def answer(self, compressed_context: str, query: str) -> str:
        """Forward compressed context to a downstream video or omni-LLM."""
