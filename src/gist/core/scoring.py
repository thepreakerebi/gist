import math
import re
from collections.abc import Iterable

from gist.core.schemas import Candidate

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


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
    return TOKEN_PATTERN.findall(value.lower())
