import wave
from pathlib import Path
from typing import Any

from gist.audio.errors import AudioTranscriptionError
from gist.media.models import AudioWindow


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
            audio_arrays = [self._read_wav(window.path) for window in batch]
            inputs = self._processor(
                text=list(prompts),
                audios=audio_arrays,
                sampling_rate=16000,
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


def _normalize(tensor: Any, torch: Any) -> Any:
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp(min=torch.finfo(tensor.dtype).eps)


def _chunks(windows: list[AudioWindow], batch_size: int) -> list[list[AudioWindow]]:
    return [windows[index : index + batch_size] for index in range(0, len(windows), batch_size)]
