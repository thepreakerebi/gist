"""The library's text encoders load once per process, not once per question.

Reloading CLIP and CLAP for every question cost seconds each time and collapsed
under concurrent load (results/load-test/). These tests pin the reuse with
stand-in scorers, so no model weights are needed.
"""

from __future__ import annotations

import threading

import pytest

from gist.library import query


class _CountingScorer:
    instances = 0
    loads = 0

    def __init__(self) -> None:
        type(self).instances += 1

    def _load(self) -> None:
        type(self).loads += 1

    def embed_text(self, text: str) -> list[float]:
        return [float(len(text))]


@pytest.fixture
def fake_scorers(monkeypatch: pytest.MonkeyPatch) -> type[_CountingScorer]:
    import gist.audio.clap as clap
    import gist.vision.clip as clip

    _CountingScorer.instances = 0
    _CountingScorer.loads = 0
    monkeypatch.setattr(clip, "HuggingFaceClipFrameScorer", _CountingScorer)
    monkeypatch.setattr(clap, "HuggingFaceClapAudioScorer", _CountingScorer)
    monkeypatch.setattr(query, "_SCORERS", {})
    return _CountingScorer


def test_encoders_load_once_across_questions(fake_scorers: type[_CountingScorer]) -> None:
    for text in ["first question", "second question", "third"]:
        embeddings = query.embed_query(text)
        assert embeddings.visual == [float(len(text))]
        assert embeddings.audio == [float(len(text))]

    # One visual and one audio scorer for the whole process.
    assert fake_scorers.instances == 2
    assert fake_scorers.loads == 2


def test_concurrent_first_questions_load_each_encoder_once(
    fake_scorers: type[_CountingScorer],
) -> None:
    start = threading.Barrier(16)

    def ask() -> None:
        start.wait()
        query.embed_query("a question asked at the same moment")

    threads = [threading.Thread(target=ask) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert fake_scorers.instances == 2
    assert fake_scorers.loads == 2
