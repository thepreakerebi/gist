import re

from gist.core.schemas import CompressionResponse, SelectedCandidate
from gist.core.scoring import text_similarity, unique_token_count


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
MIN_CLAIM_SUPPORT_SCORE = 0.12
MIN_CLAIM_CONTENT_TOKENS = 4


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


def verify_answer_claims(
    answer: str | None,
    compression: CompressionResponse,
    min_support_score: float = MIN_CLAIM_SUPPORT_SCORE,
) -> str | None:
    """Remove unsupported answer claims using selected evidence text."""

    if answer is None:
        return None
    cleaned_answer = answer.strip()
    if not cleaned_answer or not compression.selected:
        return cleaned_answer or None
    if min_support_score < 0:
        raise ValueError("min_support_score must be non-negative")

    answer_body, evidence_section = _split_evidence_section(cleaned_answer)
    claims = _answer_claims(answer_body)
    if len(claims) <= 1:
        return cleaned_answer

    supported = [
        claim
        for claim in claims
        if _is_non_claim_line(claim)
        or _claim_support_score(claim, compression.selected) >= min_support_score
    ]
    if not supported:
        return cleaned_answer
    verified_body = " ".join(supported).strip()
    if not evidence_section:
        return verified_body
    if not verified_body:
        return evidence_section
    return f"{verified_body}\n\n{evidence_section}"


def _answer_score(query: str, item: SelectedCandidate) -> float:
    score = text_similarity(query, item.text)
    text = item.text.lower()
    if query.lower().strip().startswith("why"):
        score += sum(0.25 for term in WHY_ANSWER_TERMS if term in text)
    return score


def _answer_claims(answer: str) -> list[str]:
    claims: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^\d+[.)]\s+", stripped):
            claims.append(stripped)
            continue
        parts = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", stripped)
            if part.strip()
        ]
        claims.extend(parts or [stripped])
    return claims


def _split_evidence_section(answer: str) -> tuple[str, str]:
    match = re.search(
        r"^\s*evidence\s*:\s*$|\bevidence\s*:",
        answer,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        return answer, ""
    body = answer[: match.start()].strip()
    evidence = answer[match.start() :].strip()
    return body, evidence


def _is_non_claim_line(claim: str) -> bool:
    normalized = claim.strip().lower()
    return normalized in {"evidence:", "evidence"} or re.match(r"^\d+[.)]\s*", claim) is not None


def _claim_support_score(claim: str, selected: list[SelectedCandidate]) -> float:
    if unique_token_count(claim) < MIN_CLAIM_CONTENT_TOKENS:
        return 1.0
    return max(text_similarity(claim, item.text) for item in selected)


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
