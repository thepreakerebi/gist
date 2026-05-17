from pathlib import Path

import pytest

from gist.media.models import ExtractedFrame
from gist.vision.clip import HuggingFaceClipFrameScorer
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
