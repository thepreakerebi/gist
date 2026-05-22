from pathlib import Path
from typing import Protocol

from gist.media.models import ExtractedFrame


class FrameOcr(Protocol):
    def extract_text(self, frames: list[ExtractedFrame]) -> dict[Path, str]:
        """Return OCR text keyed by frame image path."""
