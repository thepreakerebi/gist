from pathlib import Path
from typing import Protocol

from gist.media.models import AudioWindow


class AudioWindowScorer(Protocol):
    def score_windows(self, windows: list[AudioWindow], query: str) -> dict[Path, float]:
        """Return query relevance scores keyed by audio-window path."""

