from pathlib import Path
from typing import Any

from gist.audio.errors import AudioTranscriptionError
from gist.media.models import AudioWindow


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: Any | None = None

    def transcribe_windows(self, windows: list[AudioWindow]) -> dict[Path, str]:
        if not windows:
            return {}

        self._load()
        assert self._model is not None

        transcripts: dict[Path, str] = {}
        for window in windows:
            if not window.path.exists() or not window.path.is_file():
                raise AudioTranscriptionError(f"audio window does not exist: {window.path}")

            segments, _info = self._model.transcribe(
                str(window.path),
                vad_filter=True,
                beam_size=1,
            )
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
            transcripts[window.path] = transcript

        return transcripts

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise AudioTranscriptionError(
                "Whisper transcription requires optional audio dependencies. "
                "Install with: pip install -e '.[audio]'"
            ) from exc

        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
