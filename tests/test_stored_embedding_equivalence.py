"""The stored-embedding path must score identically to the live path.

The video library persists encoder output at ingestion and scores later queries
against those stored vectors instead of re-encoding. That is only sound if it
reproduces the live scorer *exactly* — otherwise the demo would silently select
different evidence than the offline pipeline every measured capstone number
comes from, and the two could no longer be compared.

The failure this pins down is real and was caught here: ``score_frames`` wraps
the query in "a video frame showing: {query}" before encoding it, and an
``embed_text`` that skipped the template produced a completely different
ranking (top-5 overlap 1/5) while looking perfectly healthy in isolation.
"""

from pathlib import Path

import pytest

from gist.audio.clap import HuggingFaceClapAudioScorer
from gist.media.models import ExtractedFrame
from gist.vision.clip import HuggingFaceClipFrameScorer
from slow_utils import require_slow_model_tests

# Tolerance for float32 accumulation differences between a fused matmul and a
# Python dot product. Observed difference on real frames is ~6e-08.
TOLERANCE = 1e-05


def _dot(left: list[float], right: list[float]) -> float:
    return sum(x * y for x, y in zip(left, right, strict=True))


def _frames(limit: int = 24) -> list[ExtractedFrame]:
    roots = sorted(Path(".gist").glob("**/frames"))
    for root in roots:
        paths = sorted(root.glob("frame_*.jpg"))[:limit]
        if len(paths) >= 8:
            return [
                ExtractedFrame(index=index, timestamp_seconds=float(index * 5), path=path)
                for index, path in enumerate(paths)
            ]
    pytest.skip("no ingested frame assets available under .gist/")


def test_clip_stored_embeddings_match_live_scoring() -> None:
    require_slow_model_tests("torch", "transformers", "PIL")

    frames = _frames()
    query = "a slide with text on it"
    scorer = HuggingFaceClipFrameScorer()

    live = scorer.score_frames(frames, query)
    vectors = {
        frame.path: embedding.vector
        for frame, embedding in zip(frames, scorer.embed_frames(frames), strict=True)
    }
    query_vector = scorer.embed_text(query)
    stored = {path: _dot(list(vector), query_vector) for path, vector in vectors.items()}

    ranked_live = [path for path, _ in sorted(live.items(), key=lambda kv: -kv[1])]
    ranked_stored = [path for path, _ in sorted(stored.items(), key=lambda kv: -kv[1])]

    assert ranked_live == ranked_stored, "stored embeddings must preserve live ranking"
    for path, score in live.items():
        assert stored[path] == pytest.approx(score, abs=TOLERANCE)


def test_clip_embed_text_applies_the_scoring_prompt_template() -> None:
    require_slow_model_tests("torch", "transformers")

    scorer = HuggingFaceClipFrameScorer()

    # The templated query and the bare query must not embed to the same vector;
    # if they do, the template has been dropped from one of the two paths.
    templated = scorer.embed_text("a slide with text on it")
    bare_as_query = scorer.embed_text("a video frame showing: a slide with text on it")

    assert _dot(templated, bare_as_query) < 0.999


def test_clip_embed_text_rejects_blank() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        HuggingFaceClipFrameScorer().embed_text("   ")


def test_clap_embed_text_rejects_blank() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        HuggingFaceClapAudioScorer().embed_text("   ")


def test_clap_stored_embeddings_match_live_scoring() -> None:
    require_slow_model_tests("torch", "transformers", "numpy")

    from gist.media.models import AudioWindow

    roots = sorted(Path(".gist").glob("**/audio"))
    windows: list[AudioWindow] = []
    for root in roots:
        paths = sorted(root.glob("*.wav"))[:8]
        if len(paths) >= 4:
            windows = [
                AudioWindow(
                    index=index,
                    start_seconds=float(index * 30),
                    end_seconds=float(index * 30 + 30),
                    duration_seconds=30.0,
                    path=path,
                )
                for index, path in enumerate(paths)
            ]
            break
    if not windows:
        pytest.skip("no ingested audio windows available under .gist/")

    query = "someone explaining a concept"
    scorer = HuggingFaceClapAudioScorer()

    live = scorer.score_windows(windows, query)
    vectors = scorer.embed_windows(windows)
    query_vector = scorer.embed_text(query)
    stored = {path: _dot(vector, query_vector) for path, vector in vectors.items()}

    ranked_live = [path for path, _ in sorted(live.items(), key=lambda kv: -kv[1])]
    ranked_stored = [path for path, _ in sorted(stored.items(), key=lambda kv: -kv[1])]

    assert ranked_live == ranked_stored
    for path, score in live.items():
        assert stored[path] == pytest.approx(score, abs=TOLERANCE)


def test_clap_scoring_is_reproducible_across_calls() -> None:
    """CLAP must return identical scores for identical inputs.

    The unfused checkpoint's feature extractor defaults to random 10 s
    truncation of any longer window, so before seeding, two consecutive calls
    on the same 30 s audio disagreed (embedding cosine 0.89-0.97). Any CLAP
    number in the evaluation harness was therefore irreproducible.
    """

    require_slow_model_tests("torch", "transformers", "numpy")

    from gist.media.models import AudioWindow

    windows: list[AudioWindow] = []
    for root in sorted(Path(".gist").glob("**/audio")):
        paths = sorted(root.glob("*.wav"))[:4]
        if len(paths) >= 4:
            windows = [
                AudioWindow(
                    index=index,
                    start_seconds=float(index * 30),
                    end_seconds=float(index * 30 + 30),
                    duration_seconds=30.0,
                    path=path,
                )
                for index, path in enumerate(paths)
            ]
            break
    if not windows:
        pytest.skip("no ingested audio windows available under .gist/")

    scorer = HuggingFaceClapAudioScorer()
    query = "someone explaining a concept"

    first = scorer.score_windows(windows, query)
    second = scorer.score_windows(windows, query)

    for path, score in first.items():
        assert second[path] == pytest.approx(score, abs=1e-09)


def test_clap_seeding_does_not_leak_into_caller_rng() -> None:
    """Seeding is scoped: a caller's own RNG stream must survive a CLAP call."""

    require_slow_model_tests("torch", "numpy")

    import numpy

    from gist.audio.clap import _deterministic_truncation

    numpy.random.seed(1234)
    expected = numpy.random.rand(3).tolist()

    numpy.random.seed(1234)
    import torch

    with _deterministic_truncation(numpy, torch):
        numpy.random.rand(10)
    assert numpy.random.rand(3).tolist() == expected
