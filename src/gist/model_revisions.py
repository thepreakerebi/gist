"""Pinned Hugging Face commits for every model Gist downloads.

Without a revision, ``from_pretrained`` resolves a repository's ``main`` branch at
download time. If a model owner pushes new weights, a fresh clone would then score
with a different model than the one behind the committed results, and nothing in
the output would show it. Pinning makes the model as fixed as the code.

Each hash below is the ``main`` commit on 2026-11-06, read from the Hugging Face
API, and is identical to the snapshot in the local cache that produced the
committed results. Every commit predates the runs in ``results/``. To move to new
weights, change the hash here and re-run the slow model tests
(``GIST_RUN_SLOW_MODEL_TESTS=1 uv run pytest -m slow``).

A model name not listed here loads at ``main``, unpinned, so passing a different
checkpoint stays possible without editing this file.
"""

from __future__ import annotations

MODEL_REVISIONS: dict[str, str] = {
    # Visual scorer, gist.vision.clip
    "openai/clip-vit-base-patch32": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    # Audio-event scorer, gist.audio.clap
    "laion/clap-htsat-unfused": "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a",
    # Transcript embeddings, gist.audio.dispatcher and gist.core.semantic
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    # Speech transcription, gist.audio.whisper (faster-whisper resolves a size
    # name such as "base" to the Systran repository)
    "Systran/faster-whisper-tiny": "d90ca5fe260221311c53c58e660288d3deb8d356",
    "Systran/faster-whisper-base": "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66",
    "Systran/faster-whisper-small": "536b0662742c02347bc0e980a01041f333bce120",
    # Answering model for every measured result. Recorded rather than enforced:
    # the run scripts under scripts/ and results/ are the record of what ran and
    # load it by name. Its main branch last moved on 2025-04-30, before any run,
    # so every run used this commit.
    "Qwen/Qwen2.5-Omni-7B": "ae9e1690543ffd5c0221dc27f79834d0294cba00",
}


def pinned_revision(model_name: str) -> str | None:
    """Return the pinned commit for ``model_name``, or None to load ``main``."""
    return MODEL_REVISIONS.get(model_name)


def whisper_repo(model_size: str) -> str:
    """The repository faster-whisper downloads for a size name like ``"base"``."""
    return f"Systran/faster-whisper-{model_size}"
