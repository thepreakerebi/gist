"""Scoring primitives the selector is built from.

Provenance of each piece, so a reader can tell what is borrowed from what:

* ``z_scores`` — ordinary standardisation, (x - mean) / sd. No source to cite;
  it is textbook statistics. What is specific to this project is *why* it is
  needed: CLIP and CLAP emit similarities on different scales (roughly 0.25-0.32
  against 0.4-0.8), so comparing them raw would hand every evidence slot to
  whichever encoder happens to output larger numbers. Standardising per modality
  is what makes a joint competition between them meaningful.

* ``temporal_similarity`` — a Gaussian (radial basis) kernel,
  exp(-(dt)^2 / sigma^2). Again a standard form rather than a contribution; the
  choice here is to express "too close together in time" as a smooth decay
  instead of a hard window, so near-duplicate evidence is penalised in
  proportion to how near it actually is. Consumed by the MMR redundancy term in
  ``core.compressor`` and by the merge gate in ``core.tail_merging``.

* ``text_similarity`` — Jaccard index over token sets (Jaccard, 1912). Used only
  for merge decisions, never for ranking.

* ``lexical_relevance`` — **own work, and deliberately crude.** It is the
  fallback that runs when no model supplied ``saliency_score``; see the note on
  that field in ``core.schemas``. The 0.75/0.25 split between overlap and
  coverage was chosen by hand and has never been ablated, because in every
  measured result CLIP and CLAP were supplying scores and this path was not
  deciding anything.

* ``STOPWORDS`` — **hand-written for this project, not taken from NLTK, spaCy or
  any published stoplist**, and NLTK is not a dependency. It is deliberately
  small: enough to stop function words and the two commonest question openers
  from inflating overlap, with no extra dependency and no language model. Known
  wart, recorded rather than hidden: it contains "how" and "what" but not "why",
  "when", "where", "who" or "which", so in a "why did..." question the token
  "why" survives and counts toward overlap. ``core.query_intent`` meanwhile
  treats "why" as a meaningful signal. The two disagree about that word. It is
  confined to the fallback path, so no measured result went through it.
"""
import math
import re
from collections.abc import Iterable

from gist.core.schemas import Candidate

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "use",
    "what",
    "with",
}


def lexical_relevance(query: str, candidate: Candidate) -> float:
    if candidate.saliency_score is not None:
        return candidate.saliency_score

    query_terms = set(_tokens(query))
    candidate_terms = set(_tokens(candidate.text))
    if not query_terms or not candidate_terms:
        return 0.0

    overlap = len(query_terms & candidate_terms) / len(query_terms)
    coverage = len(query_terms & candidate_terms) / len(candidate_terms)
    return (0.75 * overlap) + (0.25 * coverage)


def z_scores(scores: Iterable[float]) -> list[float]:
    values = list(scores)
    if not values:
        return []

    mean = sum(values) / len(values)
    variance = sum((score - mean) ** 2 for score in values) / len(values)
    std = math.sqrt(variance)
    if std == 0:
        return [0.0 for _ in values]

    return [(score - mean) / std for score in values]


def temporal_similarity(left_seconds: float, right_seconds: float, sigma_seconds: float) -> float:
    if sigma_seconds <= 0:
        return 0.0

    delta = left_seconds - right_seconds
    return math.exp(-((delta * delta) / (sigma_seconds * sigma_seconds)))


def text_similarity(left: str, right: str) -> float:
    left_terms = set(_tokens(left))
    right_terms = set(_tokens(right))
    if not left_terms or not right_terms:
        return 0.0

    return len(left_terms & right_terms) / len(left_terms | right_terms)


def unique_token_count(value: str) -> int:
    return len(set(_tokens(value)))


def _tokens(value: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall(value.lower())
        if token not in STOPWORDS
    ]
