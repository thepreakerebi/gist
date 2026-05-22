import re

from gist.core.schemas import CompressionResponse, SelectedCandidate
from gist.core.scoring import text_similarity


WHY_ANSWER_TERMS = {
    "afraid",
    "because",
    "chased",
    "chasing",
    "fear",
    "freaked",
    "nightmare",
    "nightmares",
    "reason",
    "scared",
}


def answer_from_evidence(compression: CompressionResponse) -> str | None:
    if not compression.selected:
        return None

    best = max(
        compression.selected,
        key=lambda item: _answer_score(compression.query, item)
        + _temporal_target_score(compression.query, item, compression.selected),
    )
    sentence = _best_sentence(compression.query, best.text)
    if not sentence:
        return None

    if compression.query.lower().strip().startswith("why"):
        return _why_answer(sentence)
    return sentence


def _answer_score(query: str, item: SelectedCandidate) -> float:
    score = text_similarity(query, item.text)
    text = item.text.lower()
    if query.lower().strip().startswith("why"):
        score += sum(0.25 for term in WHY_ANSWER_TERMS if term in text)
    return score


def _temporal_target_score(
    query: str,
    item: SelectedCandidate,
    selected: list[SelectedCandidate],
) -> float:
    query_lower = query.lower()
    if " after " not in f" {query_lower} " and " before " not in f" {query_lower} ":
        return 0.0

    direction = "after" if " after " in f" {query_lower} " else "before"
    anchor_terms = _anchor_terms(query_lower, direction)
    if not anchor_terms:
        return 0.0

    anchor_items = [
        candidate
        for candidate in selected
        if anchor_terms & set(re.findall(r"[a-z0-9]+", candidate.text.lower()))
    ]
    if not anchor_items:
        return 0.0

    anchor_time = (
        max(candidate.timestamp_seconds for candidate in anchor_items)
        if direction == "after"
        else min(candidate.timestamp_seconds for candidate in anchor_items)
    )
    item_terms = set(re.findall(r"[a-z0-9]+", item.text.lower()))
    if anchor_terms & item_terms:
        return -0.4
    if direction == "after" and item.timestamp_seconds > anchor_time:
        return 0.6
    if direction == "before" and item.timestamp_seconds < anchor_time:
        return 0.6
    return 0.0


def _anchor_terms(query_lower: str, direction: str) -> set[str]:
    marker = f" {direction} "
    if marker not in f" {query_lower} ":
        return set()
    fragment = query_lower.split(marker, maxsplit=1)[1]
    return {
        token
        for token in re.findall(r"[a-z0-9]+", fragment)
        if token not in {"a", "an", "and", "the", "this", "that", "what", "which"}
    }


def _best_sentence(query: str, text: str) -> str:
    sentences = [
        sentence.strip(" ,")
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip(" ,")
    ]
    if not sentences:
        return text.strip()
    return max(sentences, key=lambda sentence: _sentence_score(query, sentence))


def _sentence_score(query: str, sentence: str) -> float:
    score = text_similarity(query, sentence)
    lower = sentence.lower()
    if query.lower().strip().startswith("why"):
        score += sum(0.35 for term in WHY_ANSWER_TERMS if term in lower)
    return score


def _why_answer(sentence: str) -> str:
    cleaned = sentence.strip()
    if cleaned.lower().startswith(("because ", "he ", "she ", "they ", "the ")):
        return cleaned
    return f"The evidence suggests: {cleaned}"
