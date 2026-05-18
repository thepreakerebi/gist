from pathlib import Path
from typing import Protocol

from gist.media.models import ExtractedFrame
from gist.vision.scene import FrameEmbedding


class VisualFrameScorer(Protocol):
    def score_frames(self, frames: list[ExtractedFrame], query: str) -> dict[Path, float]:
        """Return query relevance scores keyed by frame image path."""


class VisualFrameIndexer(Protocol):
    def embed_frames(self, frames: list[ExtractedFrame]) -> list[FrameEmbedding]:
        """Return frame embeddings for scene detection and visual indexing."""
