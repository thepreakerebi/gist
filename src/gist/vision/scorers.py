from pathlib import Path
from typing import Protocol

from gist.media.models import ExtractedFrame


class VisualFrameScorer(Protocol):
    def score_frames(self, frames: list[ExtractedFrame], query: str) -> dict[Path, float]:
        """Return query relevance scores keyed by frame image path."""

