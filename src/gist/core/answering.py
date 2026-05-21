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
        key=lambda item: _answer_score(compression.query, item),
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
