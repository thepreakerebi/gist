from pathlib import Path
from typing import Any, TypeVar

from gist.media.models import ExtractedFrame
from gist.vision.errors import VisualScoringError
from gist.vision.scene import FrameEmbedding

T = TypeVar("T")


class HuggingFaceClipFrameScorer:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None

    def score_frames(self, frames: list[ExtractedFrame], query: str) -> dict[Path, float]:
        if not frames:
            return {}
        if not query.strip():
            raise ValueError("query must not be blank")

        self._load()
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None

        image_paths = [frame.path for frame in frames]
        scores: dict[Path, float] = {}
        text_prompt = f"a video frame showing: {query.strip()}"

        for batch_paths in _chunks(image_paths, self.batch_size):
            images = [self._load_image(path) for path in batch_paths]
            inputs = self._processor(
                text=[text_prompt],
                images=images,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(self._model.device) for key, value in inputs.items()}

            with self._torch.no_grad():
                output = self._model(**inputs)
                image_embeds = _normalize(output.image_embeds, self._torch)
                text_embeds = _normalize(output.text_embeds, self._torch)
                similarities = image_embeds @ text_embeds.T

            for path, score in zip(batch_paths, similarities.squeeze(dim=1).tolist(), strict=True):
                scores[path] = float(score)

        return scores

    def embed_frames(self, frames: list[ExtractedFrame]) -> list[FrameEmbedding]:
        if not frames:
            return []

        self._load()
        assert self._model is not None
        assert self._processor is not None
        assert self._torch is not None

        embeddings: list[FrameEmbedding] = []
        for batch_frames in _chunks(frames, self.batch_size):
            images = [self._load_image(frame.path) for frame in batch_frames]
            inputs = self._processor(
                images=images,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(self._model.device) for key, value in inputs.items()}

            with self._torch.no_grad():
                image_embeds = _normalize(
                    self._model.get_image_features(**inputs),
                    self._torch,
                )

            for frame, vector in zip(batch_frames, image_embeds.tolist(), strict=True):
                embeddings.append(
                    FrameEmbedding(
                        frame_index=frame.index,
                        timestamp_seconds=frame.timestamp_seconds,
                        vector=tuple(float(value) for value in vector),
                    )
                )

        return embeddings

    def _load(self) -> None:
        if self._model is not None and self._processor is not None and self._torch is not None:
            return

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise VisualScoringError(
                "CLIP scoring requires optional vision dependencies. "
                "Install with: pip install -e '.[vision]'"
            ) from exc

        selected_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name).to(selected_device)
        self._model.eval()

    def _load_image(self, path: Path) -> Any:
        try:
            from PIL import Image
        except ImportError as exc:
            raise VisualScoringError(
                "CLIP scoring requires Pillow. Install with: pip install -e '.[vision]'"
            ) from exc

        if not path.exists() or not path.is_file():
            raise VisualScoringError(f"frame image does not exist: {path}")

        with Image.open(path) as image:
            return image.convert("RGB")


def _normalize(tensor: Any, torch: Any) -> Any:
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp(min=torch.finfo(tensor.dtype).eps)


def _chunks(items: list[T], batch_size: int) -> list[list[T]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]
