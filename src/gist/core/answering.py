import re

from gist.core.query_intent import QueryIntent
from gist.core.schemas import CompressionResponse, SelectedCandidate
from gist.core.scoring import text_similarity, unique_token_count
from gist.core.temporal_query import parse_temporal_query, rank_temporal_pairs

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
_VISUAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "in",
    "is",
    "me",
    "of",
    "on",
    "screen",
    "show",
    "shown",
    "the",
    "to",
    "video",
    "what",
    "when",
    "where",
}


def answer_from_evidence(compression: CompressionResponse) -> str | None:
    if not compression.selected:
        return None

    visual_answer = _visual_object_answer(compression)
    if visual_answer is not None:
        return visual_answer

    entity_answer = _entity_evidence_answer(compression)
    if entity_answer is not None:
        return entity_answer

    answer_candidates = _answer_candidates(compression)
    if not answer_candidates:
        return None

    best = max(
        answer_candidates,
        key=lambda item: _answer_score(compression.query, item)
        + _temporal_target_score(compression.query, item, compression.selected),
    )
    sentence = _best_sentence(compression.query, best.text)
    if not sentence:
        return None

    if compression.query.lower().strip().startswith("why"):
        return _why_answer(sentence)
    return sentence


def _visual_object_answer(compression: CompressionResponse) -> str | None:
    query = compression.query.lower().strip()
    if compression.query_intent == QueryIntent.MIXED_AV:
        return None
    if _is_text_query(query):
        return None
    if not _is_visual_object_query(query):
        return None
    visual_items = [item for item in compression.selected if item.modality.value == "visual"]
    if not visual_items:
        return None
    best = max(
        visual_items,
        key=lambda item: (
            item.visual_support_score or 0.0,
            item.relevance_score,
            item.normalized_score,
        ),
    )
    target = _visual_target(compression.query)
    time_span = _time_span(best)
    return f"Visual evidence shows {target}{time_span}."


def _entity_evidence_answer(compression: CompressionResponse) -> str | None:
    query = compression.query.lower().strip()
    if _is_question_query(query) or _is_text_query(query):
        return None

    target = _visual_target(compression.query)
    target_terms = set(target.split())
    if not target_terms or len(target_terms) > 5:
        return None

    visual_item = _best_modality_item(compression.selected, "visual")
    audio_item = _best_text_item(compression.query, compression.selected)
    if visual_item is None and audio_item is None:
        return None

    parts: list[str] = []
    if visual_item is not None:
        parts.append(f"visual evidence shows {target}{_time_span(visual_item)}")
    if audio_item is not None:
        sentence = _best_sentence(compression.query, audio_item.text)
        if sentence:
            audio_span = _time_span(audio_item)
            parts.append(f"transcript evidence mentions {target}{audio_span}: {sentence}")

    if not parts:
        return None
    return _ensure_terminal_punctuation(f"Selected evidence indicates that {'; '.join(parts)}")


def _ensure_terminal_punctuation(value: str) -> str:
    return value if value.endswith((".", "?", "!")) else f"{value}."


def _is_question_query(query: str) -> bool:
    return query.startswith(("why ", "what ", "how ", "when ", "where ", "who ", "which "))


def _best_modality_item(
    selected: list[SelectedCandidate],
    modality: str,
) -> SelectedCandidate | None:
    items = [item for item in selected if item.modality.value == modality]
    if not items:
        return None
    return max(
        items,
        key=lambda item: (
            item.visual_support_score or 0.0,
            item.relevance_score,
            item.normalized_score,
            -item.selection_rank,
        ),
    )


def _best_text_item(query: str, selected: list[SelectedCandidate]) -> SelectedCandidate | None:
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower())) - _VISUAL_STOPWORDS
    if not query_terms:
        return None
    text_items = [
        item
        for item in selected
        if item.modality.value == "audio"
        and query_terms & set(re.findall(r"[a-z0-9]+", item.text.lower()))
    ]
    if not text_items:
        return None
    return max(
        text_items,
        key=lambda item: (
            text_similarity(query, item.text),
            item.relevance_score,
            item.normalized_score,
            -item.selection_rank,
        ),
    )


def _answer_candidates(compression: CompressionResponse) -> list[SelectedCandidate]:
    query = compression.query.lower().strip()
    if _is_text_query(query):
        return compression.selected
    if _is_question_query(query):
        transcript_items = [
            item
            for item in compression.selected
            if item.modality.value == "audio" and not _is_visual_placeholder(item.text)
        ]
        if transcript_items:
            return transcript_items
    return [
        item for item in compression.selected if not _is_visual_only_answer_text(item.text)
    ]


def _is_visual_only_answer_text(text: str) -> bool:
    return _is_visual_placeholder(text) or text.lower().startswith("on-screen text near")


def _is_visual_placeholder(text: str) -> bool:
    return text.lower().startswith("visual frame sampled at")


def _is_text_query(query: str) -> bool:
    return any(
        term in f" {query} "
        for term in [
            " text ",
            " words ",
            " caption ",
            " written ",
            " title ",
            " logo ",
            " label ",
            " slide ",
        ]
    )


def _is_visual_object_query(query: str) -> bool:
    visual_markers = {
        "appear",
        "appears",
        "display",
        "displayed",
        "look",
        "screen",
        "see",
        "show",
        "shown",
        "visible",
        "watch",
    }
    query_terms = set(re.findall(r"[a-z0-9]+", query))
    if query_terms & visual_markers:
        return True
    return query.startswith(("where ", "when ")) and bool(query_terms - _VISUAL_STOPWORDS)


def _visual_target(query: str) -> str:
    terms = [
        token
        for token in re.findall(r"[a-z0-9]+", query.lower())
        if token not in _VISUAL_STOPWORDS
    ]
    return " ".join(terms) if terms else "the requested visual target"


def _time_span(item: SelectedCandidate) -> str:
    start = (
        item.clip_start_seconds
        if item.clip_start_seconds is not None
        else item.scene_start_seconds
    )
    end = item.clip_end_seconds if item.clip_end_seconds is not None else item.scene_end_seconds
    if start is None or end is None:
        return f" near {item.timestamp_seconds:.2f} seconds"
    if start == end:
        return f" near {start:.2f} seconds"
    return f" from {min(start, end):.2f}s to {max(start, end):.2f}s"


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
    claim_variants = {claim, _strip_attribution_prefix(claim)}
    return max(
        max(text_similarity(variant, item.text), _token_overlap_support(variant, item.text))
        for variant in claim_variants
        for item in selected
    )


def _strip_attribution_prefix(claim: str) -> str:
    return re.sub(
        r"^\s*(the\s+)?(presenter|speaker|narrator)\s+(says?|explains?|describes?)\s+",
        "",
        claim,
        flags=re.IGNORECASE,
    )


def _token_overlap_support(claim: str, evidence: str) -> float:
    claim_terms = _claim_terms(claim)
    if not claim_terms:
        return 0.0
    evidence_terms = _claim_terms(evidence)
    overlap = len(claim_terms & evidence_terms) / len(claim_terms)
    return overlap * 0.2


def _claim_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _VISUAL_STOPWORDS and len(token) > 2
    }


def _temporal_target_score(
    query: str,
    item: SelectedCandidate,
    selected: list[SelectedCandidate],
) -> float:
    model_score = _model_temporal_target_score(query, item, selected)
    if model_score is not None:
        return model_score

    query_lower = query.lower()
    has_direction = (
        " after " in f" {query_lower} " or " before " in f" {query_lower} "
    )
    if not has_direction and any(
        marker in f" {query_lower} "
        for marker in [" opening ", " beginning ", " first ", " start "]
    ):
        earliest_timestamp = min(candidate.timestamp_seconds for candidate in selected)
        distance_seconds = max(item.timestamp_seconds - earliest_timestamp, 0.0)
        return max(1.5 - distance_seconds / 60.0, -0.5)
    if not has_direction:
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


def _model_temporal_target_score(
    query: str,
    item: SelectedCandidate,
    selected: list[SelectedCandidate],
) -> float | None:
    temporal_items = [
        candidate
        for candidate in selected
        if candidate.temporal_anchor_score is not None
        and candidate.temporal_target_score is not None
        and candidate.temporal_direction in {"after", "before"}
    ]
    if not temporal_items:
        return None

    temporal_query = parse_temporal_query(query)
    if temporal_query is None:
        return None
    pairs = rank_temporal_pairs(
        temporal_items,
        direction=temporal_query.direction,
        target_query=temporal_query.target,
    )
    if not pairs:
        return None
    _, anchor, target = pairs[0]
    if item.id == anchor.id:
        return -1.0
    if item.id != target.id:
        return -0.75
    return 3.0


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
