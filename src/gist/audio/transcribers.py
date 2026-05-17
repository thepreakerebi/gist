from pathlib import Path
from typing import Protocol

from gist.media.models import AudioWindow


class AudioTranscriber(Protocol):
    def transcribe_windows(self, windows: list[AudioWindow]) -> dict[Path, str]:
        """Return transcripts keyed by audio-window path."""

