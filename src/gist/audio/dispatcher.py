"""Speech-vs-sound dispatcher (Gist Pillar 1, audio path).

For each audio window, decide whether it is speech-dominant or sound-dominant
using CLAP probes, then score it query-conditionally with the right encoder:

    speech = CLAP(window, "a voice speaking") - CLAP(window, "ambient sound")
    if speech > tau:   # speech-dominant
        r = cos(sent_embed(Whisper(window)), sent_embed(query))
    else:              # sound-dominant
        r = CLAP(window, "the sound of: {query}")

This is the training-free, query-conditional cross-modal saliency the capstone
plan specifies. It implements the ``AudioWindowScorer`` protocol so it drops
into the existing candidate generator as ``audio_scorer``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from gist.audio.clap import HuggingFaceClapAudioScorer
from gist.audio.errors import AudioTranscriptionError
from gist.audio.whisper import FasterWhisperTranscriber
from gist.media.models import AudioWindow

_SPEECH_PROBE = "a voice speaking"
_AMBIENT_PROBE = "ambient sound"


class SentenceEmbedder(Protocol):
    def encode(self, texts: list[str]) -> Any:
        """Return L2-normalized sentence embeddings (rows aligned with texts)."""


class SentenceTransformerEmbedder:
    """Lazy wrapper around a small sentence-transformers model (text-vs-text)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise AudioTranscriptionError(
                "The speech dispatcher needs sentence embeddings. "
                "Install with: pip install -e '.[audio]'"
            ) from exc
        self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> Any:
        self._load()
        assert self._model is not None
        return self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )


class SpeechSoundDispatcher:
    """AudioWindowScorer that routes each window to Whisper (speech) or CLAP (sound)."""

    def __init__(
        self,
        clap: HuggingFaceClapAudioScorer,
        transcriber: FasterWhisperTranscriber,
        embedder: SentenceEmbedder | None = None,
        speech_threshold: float = 0.0,
    ) -> None:
        self.clap = clap
        self.transcriber = transcriber
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.speech_threshold = speech_threshold
        # Populated per call so callers can inspect routing decisions.
        self.last_routing: dict[Path, str] = {}

    def score_windows(self, windows: list[AudioWindow], query: str) -> dict[Path, float]:
        if not windows:
            return {}
        if not query.strip():
            raise ValueError("query must not be blank")

        query = query.strip()
        probes = self.clap.score_windows_against(
            windows, [_SPEECH_PROBE, _AMBIENT_PROBE, f"the sound of: {query}"]
        )

        speech_windows: list[AudioWindow] = []
        scores: dict[Path, float] = {}
        routing: dict[Path, str] = {}
        for window in windows:
            row = probes.get(window.path)
            if row is None:
                continue
            speech_sim, ambient_sim, sound_query_sim = row
            if speech_sim - ambient_sim > self.speech_threshold:
                speech_windows.append(window)
                routing[window.path] = "speech"
                # provisional; overwritten by the transcript score below
                scores[window.path] = sound_query_sim
            else:
                scores[window.path] = sound_query_sim
                routing[window.path] = "sound"

        if speech_windows:
            self._score_speech_windows(speech_windows, query, scores)

        self.last_routing = routing
        return scores

    def _score_speech_windows(
        self, speech_windows: list[AudioWindow], query: str, scores: dict[Path, float]
    ) -> None:
        transcripts = self.transcriber.transcribe_windows(speech_windows)
        texts = [(transcripts.get(w.path) or "").strip() for w in speech_windows]
        # A speech window with no usable transcript keeps its CLAP fallback score.
        indexed = [(i, t) for i, t in enumerate(texts) if t]
        if not indexed:
            return
        embeddings = self.embedder.encode([query] + [t for _i, t in indexed])
        query_vec = embeddings[0]
        for offset, (i, _text) in enumerate(indexed, start=1):
            sim = float((embeddings[offset] * query_vec).sum())  # both are normalized
            scores[speech_windows[i].path] = sim
