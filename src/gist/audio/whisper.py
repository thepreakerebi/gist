import hashlib
import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Any

from gist.audio.errors import AudioTranscriptionError
from gist.media.models import AudioWindow

_TRANSCRIPT_CACHE: dict[tuple[str, str, str, str, str, int, int], str] = {}


class TranscriptQuality(StrEnum):
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"


class WhisperSettings:
    def __init__(
        self,
        model_size: str,
        device: str,
        compute_type: str,
        beam_size: int,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size

    @property
    def cache_fingerprint(self) -> str:
        return (
            f"model={self.model_size}|device={self.device}|"
            f"compute={self.compute_type}|beam={self.beam_size}"
        )


_QUALITY_DEFAULTS = {
    TranscriptQuality.FAST: ("tiny", "cpu", "int8", 1),
    TranscriptQuality.BALANCED: ("base", "cpu", "int8", 3),
    TranscriptQuality.ACCURATE: ("small", "cpu", "int8", 5),
}


def resolve_whisper_settings(
    quality: TranscriptQuality = TranscriptQuality.BALANCED,
    model_size: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    beam_size: int | None = None,
) -> WhisperSettings:
    default_model, default_device, default_compute, default_beam = _QUALITY_DEFAULTS[quality]
    return WhisperSettings(
        model_size=model_size or os.getenv("GIST_WHISPER_MODEL_SIZE", default_model),
        device=device or os.getenv("GIST_WHISPER_DEVICE", default_device),
        compute_type=compute_type or os.getenv("GIST_WHISPER_COMPUTE_TYPE", default_compute),
        beam_size=beam_size
        or int(os.getenv("GIST_WHISPER_BEAM_SIZE", str(default_beam))),
    )


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
        cache_dir: Path | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.cache_dir = cache_dir
        self._model: Any | None = None

    def transcribe_windows(self, windows: list[AudioWindow]) -> dict[Path, str]:
        if not windows:
            return {}

        transcripts: dict[Path, str] = {}
        for window in windows:
            if not window.path.exists() or not window.path.is_file():
                raise AudioTranscriptionError(f"audio window does not exist: {window.path}")

            cache_key = self._cache_key(window.path)
            transcript = _TRANSCRIPT_CACHE.get(cache_key)
            if transcript is None:
                transcript = self._read_disk_cache(cache_key)
            if transcript is None:
                self._load()
                assert self._model is not None
                segments, _info = self._model.transcribe(
                    str(window.path),
                    vad_filter=True,
                    beam_size=self.beam_size,
                )
                transcript = " ".join(segment.text.strip() for segment in segments).strip()
                _TRANSCRIPT_CACHE[cache_key] = transcript
                self._write_disk_cache(cache_key, transcript)
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

    def _cache_key(self, path: Path) -> tuple[str, str, str, str, str, int, int]:
        stat = path.stat()
        return (
            self.model_size,
            self.device,
            self.compute_type,
            str(self.beam_size),
            str(path.resolve()),
            int(stat.st_mtime_ns),
            int(stat.st_size),
        )

    def _cache_path(
        self,
        cache_key: tuple[str, str, str, str, str, int, int],
    ) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(json.dumps(cache_key).encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / f"{digest}.json"

    def _read_disk_cache(
        self,
        cache_key: tuple[str, str, str, str, str, int, int],
    ) -> str | None:
        path = self._cache_path(cache_key)
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("cache_key") != list(cache_key):
            return None
        transcript = payload.get("transcript")
        if not isinstance(transcript, str):
            return None
        _TRANSCRIPT_CACHE[cache_key] = transcript
        return transcript

    def _write_disk_cache(
        self,
        cache_key: tuple[str, str, str, str, str, int, int],
        transcript: str,
    ) -> None:
        path = self._cache_path(cache_key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "cache_key": list(cache_key),
                    "transcript": transcript,
                },
                indent=2,
            )
        )
