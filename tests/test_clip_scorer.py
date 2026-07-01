from pathlib import Path

import pytest

from gist.media.models import ExtractedFrame
from gist.vision.clip import HuggingFaceClipFrameScorer, _ClipProcessorCompat
from gist.vision.errors import VisualScoringError


def test_clip_scorer_requires_non_blank_query() -> None:
    scorer = HuggingFaceClipFrameScorer()

    with pytest.raises(ValueError, match="query must not be blank"):
        scorer.score_frames([ExtractedFrame(index=0, timestamp_seconds=0, path=Path("x.jpg"))], " ")


def test_clip_scorer_reports_missing_optional_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = __import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise ImportError("missing torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(VisualScoringError, match="optional vision dependencies"):
        HuggingFaceClipFrameScorer().score_frames(
            [ExtractedFrame(index=0, timestamp_seconds=0, path=Path("x.jpg"))],
            "speaker",
        )


def test_clip_processor_compat_merges_text_and_image_inputs() -> None:
    class FakeTokenizer:
        def __call__(self, text, return_tensors, padding):
            return {
                "input_ids": [len(text)],
                "attention_mask": [1 if padding else 0],
                "text_tensors": return_tensors,
            }

    class FakeImageProcessor:
        def __call__(self, images, return_tensors):
            return {
                "pixel_values": [len(images)],
                "image_tensors": return_tensors,
            }

    processor = _ClipProcessorCompat(
        image_processor=FakeImageProcessor(),
        tokenizer=FakeTokenizer(),
    )

    inputs = processor(
        text=["a frame"],
        images=["image"],
        return_tensors="pt",
        padding=True,
    )

    assert inputs == {
        "input_ids": [1],
        "attention_mask": [1],
        "text_tensors": "pt",
        "pixel_values": [1],
        "image_tensors": "pt",
    }
