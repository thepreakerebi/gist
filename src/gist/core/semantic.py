"""Semantic (embedding-based) relevance for text/transcript span selection.

Token-overlap relevance (``scoring.lexical_relevance``) picks spans that share
words with the query, so on indirectly-worded questions it can rank a
keyword-overlapping-but-irrelevant span above the true answer span. This scorer
ranks spans by sentence-embedding cosine similarity to the query instead, so
paraphrased answers are found even when they share few words with the question.

It mirrors the CLIP/CLAP scorer pattern: score candidates, attach the score as
``saliency_score`` so it flows through the existing MMR selection unchanged.
"""

from __future__ import annotations

from typing import Any, Protocol

from gist.core.schemas import Candidate


class SentenceEmbedder(Protocol):
    def encode(self, texts: list[str]) -> Any:
        """Return L2-normalized sentence embeddings (rows aligned with texts)."""


class SentenceTransformerEmbedder:
    """Lazy wrapper around a small local sentence-transformers model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Semantic span selection needs sentence embeddings. "
                "Install with: pip install -e '.[audio]'"
            ) from exc
        self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> Any:
        self._load()
        assert self._model is not None
        return self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )


class SemanticTextScorer:
    """Scores text candidates by embedding cosine similarity to the query."""

    def __init__(self, embedder: SentenceEmbedder | None = None) -> None:
        self.embedder = embedder or SentenceTransformerEmbedder()

    def score_texts(self, query: str, texts: list[str]) -> list[float]:
        """Cosine similarity of each text to the query (aligned with ``texts``)."""
        if not query.strip() or not texts:
            return [0.0 for _ in texts]
        # Empty/whitespace texts get 0; encode the rest.
        idx = [i for i, t in enumerate(texts) if t and t.strip()]
        scores = [0.0 for _ in texts]
        if not idx:
            return scores
        embeddings = self.embedder.encode([query] + [texts[i] for i in idx])
        query_vec = embeddings[0]
        for pos, i in enumerate(idx, start=1):
            # embeddings are L2-normalized, so dot product == cosine similarity.
            scores[i] = float((embeddings[pos] * query_vec).sum())
        return scores

    def score_candidates(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        """Return copies of ``candidates`` with ``saliency_score`` set to the
        semantic relevance of their text, so downstream MMR uses it directly."""
        scores = self.score_texts(query, [c.text or "" for c in candidates])
        return [
            candidate.model_copy(update={"saliency_score": score})
            for candidate, score in zip(candidates, scores, strict=True)
        ]
