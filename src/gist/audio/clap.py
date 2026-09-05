import wave
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from gist.audio.errors import AudioTranscriptionError
from gist.media.models import AudioWindow


# ClapFeatureExtractor for the *unfused* checkpoint defaults to
# truncation="rand_trunc": any window longer than 10 s is reduced to a RANDOM
# 10 s excerpt, drawn from numpy's global RNG. Two identical calls therefore
# return different features (observed embedding cosine 0.89-0.97 between
# consecutive runs on the same 30 s window), which makes CLAP scores
# irreproducible run to run. "fusion" is not an alternative here — it emits
# 4-channel features the unfused model cannot consume.
#
# Seeding around feature extraction restores exact reproducibility (cosine
# 1.0). The excerpt chosen is still arbitrary, but it is now the *same*
# arbitrary excerpt every time, which is what determinism requires.
CLAP_TRUNCATION_SEED = 0


@contextmanager
def _deterministic_truncation(numpy: Any, torch: Any) -> Iterator[None]:
    """Seed the RNGs CLAP's random truncation draws from, then restore them.

    State is saved and restored so seeding never leaks into a caller that is
    relying on its own RNG stream.
    """

    numpy_state = numpy.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        numpy.random.seed(CLAP_TRUNCATION_SEED)
        torch.manual_seed(CLAP_TRUNCATION_SEED)
        yield
    finally:
        numpy.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


class HuggingFaceClapAudioScorer:
    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        device: str | None = None,
        batch_size: int = 8,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._numpy: Any | None = None

    def score_windows(self, windows: list[AudioWindow], query: str) -> dict[Path, float]:
        if not windows:
            return {}
        if not query.strip():
            raise ValueError("query must not be blank")

        prompt = f"the sound of: {query.strip()}"
        scored = self.score_windows_against(windows, [prompt])
        return {path: values[0] for path, values in scored.items()}

    def score_windows_against(
        self, windows: list[AudioWindow], prompts: list[str]
    ) -> dict[Path, list[float]]:
        """Score each window against several raw text prompts in one pass.

        Returns per-window contrastive similarities aligned with ``prompts``.
        Used by the speech-vs-sound dispatcher to probe "a voice speaking" vs
        "ambient sound" alongside the query prompt without re-encoding audio.
        """
        if not windows:
            return {}
        if not prompts:
            raise ValueError("prompts must not be empty")

        self._load()
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None

        scores: dict[Path, list[float]] = {}
        for batch in _chunks(windows, self.batch_size):
            # CLAP (htsat) expects 48 kHz; windows are 16 kHz (for Whisper), so resample.
            audio_arrays = [self._resample_48k(self._read_wav(window.path)) for window in batch]
            with _deterministic_truncation(self._numpy, self._torch):
                inputs = self._processor(
                    text=list(prompts),
                    audio=audio_arrays,
                    sampling_rate=48000,
                    return_tensors="pt",
                    padding=True,
                )
            inputs = {key: value.to(self._model.device) for key, value in inputs.items()}

            with self._torch.no_grad():
                output = self._model(**inputs)
                audio_embeds = _normalize(output.audio_embeds, self._torch)
                text_embeds = _normalize(output.text_embeds, self._torch)
                similarities = audio_embeds @ text_embeds.T  # [batch, prompts]

            for window, row in zip(batch, similarities.tolist(), strict=True):
                scores[window.path] = [float(value) for value in row]

        return scores

    def embed_windows(self, windows: list[AudioWindow]) -> dict[Path, list[float]]:
        """Embed audio windows without a query — the query-independent tower.

        Splitting the audio tower out is what lets ingestion run once and be
        persisted: encoding a window is expensive and does not depend on what
        anyone will later ask, while comparing that encoding to a query is a
        dot product. Mirrors ``HuggingFaceClipFrameScorer.embed_frames``.
        """

        if not windows:
            return {}

        self._load()
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None

        embeddings: dict[Path, list[float]] = {}
        for batch in _chunks(windows, self.batch_size):
            # CLAP (htsat) expects 48 kHz; windows are 16 kHz (for Whisper), so resample.
            audio_arrays = [self._resample_48k(self._read_wav(window.path)) for window in batch]
            with _deterministic_truncation(self._numpy, self._torch):
                inputs = self._processor(
                    audio=audio_arrays,
                    sampling_rate=48000,
                    return_tensors="pt",
                    padding=True,
                )
            inputs = {key: value.to(self._model.device) for key, value in inputs.items()}

            with self._torch.no_grad():
                audio_embeds = _normalize(
                    _feature_tensor(self._model.get_audio_features(**inputs)),
                    self._torch,
                )

            for window, vector in zip(batch, audio_embeds.tolist(), strict=True):
                embeddings[window.path] = [float(value) for value in vector]

        return embeddings

    def embed_text(self, text: str) -> list[float]:
        """Embed a query into CLAP's shared space, for comparison with stored windows."""

        if not text.strip():
            raise ValueError("text must not be blank")

        self._load()
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None

        # Same prompt template score_windows uses, so a stored-embedding score is
        # numerically comparable with a live one.
        prompt = f"the sound of: {text.strip()}"
        inputs = self._processor(text=[prompt], return_tensors="pt", padding=True)
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}

        with self._torch.no_grad():
            text_embeds = _normalize(
                _feature_tensor(self._model.get_text_features(**inputs)),
                self._torch,
            )

        return [float(value) for value in text_embeds[0].tolist()]

    def _load(self) -> None:
        if self._model is not None and self._processor is not None and self._torch is not None:
            return

        try:
            import numpy
            import torch
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise AudioTranscriptionError(
                "CLAP audio scoring requires optional sound dependencies. "
                "Install with: pip install -e '.[sound]'"
            ) from exc

        selected_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._numpy = numpy
        self._torch = torch
        self._processor = ClapProcessor.from_pretrained(self.model_name)
        self._model = ClapModel.from_pretrained(self.model_name).to(selected_device)
        self._model.eval()

    def _resample_48k(self, audio: Any, orig_sr: int = 16000) -> Any:
        if orig_sr == 48000 or len(audio) == 0:
            return audio
        np = self._numpy
        n_out = int(round(len(audio) * 48000 / orig_sr))
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)

    def _read_wav(self, path: Path) -> Any:
        if self._numpy is None:
            raise AudioTranscriptionError("CLAP scorer is not initialized")
        if not path.exists() or not path.is_file():
            raise AudioTranscriptionError(f"audio window does not exist: {path}")

        with wave.open(str(path), "rb") as wav_file:
            if wav_file.getnchannels() != 1:
                raise AudioTranscriptionError("CLAP scorer expects mono WAV audio")
            if wav_file.getframerate() != 16000:
                raise AudioTranscriptionError("CLAP scorer expects 16 kHz WAV audio")
            sample_width = wav_file.getsampwidth()
            raw = wav_file.readframes(wav_file.getnframes())

        if sample_width == 2:
            audio = self._numpy.frombuffer(raw, dtype=self._numpy.int16).astype(self._numpy.float32)
            return audio / 32768.0
        if sample_width == 4:
            audio = self._numpy.frombuffer(raw, dtype=self._numpy.int32).astype(self._numpy.float32)
            return audio / 2147483648.0

        raise AudioTranscriptionError(f"unsupported WAV sample width: {sample_width}")


def _feature_tensor(output: Any) -> Any:
    """Unwrap a features call that may return a tensor or an output object.

    Recent transformers wraps projected features in ``BaseModelOutputWithPooling``
    rather than returning a bare tensor. Audio and text must be unwrapped the
    same way or their embeddings land in different spaces.
    """

    if hasattr(output, "norm"):
        return output
    for attribute in ("audio_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        tensor = getattr(output, attribute, None)
        if tensor is not None:
            return tensor[:, 0] if attribute == "last_hidden_state" else tensor
    if isinstance(output, tuple) and output:
        return output[0]
    raise AudioTranscriptionError("CLAP feature output did not contain a tensor")


def _normalize(tensor: Any, torch: Any) -> Any:
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp(min=torch.finfo(tensor.dtype).eps)


def _chunks(windows: list[AudioWindow], batch_size: int) -> list[list[AudioWindow]]:
    return [windows[index : index + batch_size] for index in range(0, len(windows), batch_size)]
