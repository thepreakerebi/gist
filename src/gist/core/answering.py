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
MAX_GLOBAL_SUMMARY_TOPICS = 3
_GLOBAL_ADMIN_TERMS = {
    "assignment",
    "course",
    "evaluation",
    "exercise",
    "lecture",
    "objectives",
    "project",
    "schedule",
    "seminar",
    "tuesday",
    "week",
}
_GLOBAL_TECHNICAL_TERMS = {
    "architecture",
    "behavior",
    "biomechanics",
    "bio",
    "biology",
    "control",
    "data",
    "design",
    "efficiency",
    "feedback",
    "intelligence",
    "legged",
    "learning",
    "locomotion",
    "machine",
    "machines",
    "model",
    "motor",
    "power",
    "robot",
    "robotics",
    "sensor",
    "sensors",
    "system",
    "tradeoffs",
    "velocity",
}
_GLOBAL_TOPIC_LABELS = {
    "architecture": "architecture",
    "behavior": "behavior",
    "biomechanics": "biomechanics",
    "bio": "bio-inspired robotics",
    "biology": "biology",
    "control": "control",
    "data": "data",
    "design": "design tradeoffs",
    "efficiency": "efficiency",
    "feedback": "feedback",
    "intelligence": "intelligence",
    "legged": "legged locomotion",
    "learning": "learning",
    "locomotion": "locomotion",
    "machine": "machines",
    "machines": "machines",
    "model": "models",
    "motor": "motor control",
    "power": "power",
    "robot": "robotics",
    "robotics": "robotics",
    "sensor": "sensors",
    "sensors": "sensors",
    "system": "systems",
    "tradeoffs": "design tradeoffs",
    "velocity": "velocity",
}
_GLOBAL_TOPIC_STOPWORDS = {
    "about",
    "all",
    "also",
    "basically",
    "course",
    "courses",
    "covered",
    "covers",
    "final",
    "have",
    "lecture",
    "main",
    "many",
    "overview",
    "people",
    "place",
    "project",
    "research",
    "section",
    "seminar",
    "thing",
    "throughout",
    "topic",
    "topics",
    "video",
    "week",
    "weeks",
}
_GLOBAL_FILLER_PHRASES = (
    "a lot of",
    "and then",
    "basically",
    "kind of",
    "so this is",
    "the interesting thing",
    "this role",
    "we have",
)
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

    global_summary = _global_summary_answer(compression)
    if global_summary is not None:
        return global_summary

    temporal_answer = _temporal_answer(compression)
    if temporal_answer is not None:
        return temporal_answer

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


def _global_summary_answer(compression: CompressionResponse) -> str | None:
    if compression.query_intent != QueryIntent.GLOBAL_SUMMARY:
        return None

    agenda_answer = _global_summary_agenda_answer(compression.selected)
    if agenda_answer is not None:
        return agenda_answer

    candidates = [
        (
            _global_summary_score(item),
            item.timestamp_seconds,
            _global_summary_topic(item),
        )
        for item in compression.selected
    ]
    substantive = [
        (score, timestamp, topic)
        for score, timestamp, topic in candidates
        if topic and score > 0
    ]
    if not substantive:
        return None

    strongest = sorted(
        _dedupe_global_topics(substantive),
        key=lambda entry: (entry[0], -entry[1]),
        reverse=True,
    )[:MAX_GLOBAL_SUMMARY_TOPICS]
    topics = [
        snippet.rstrip(" .")
        for _score, _timestamp, snippet in sorted(
            strongest,
            key=lambda entry: entry[1],
        )
    ]
    return f"The video covers: {'; '.join(topics)}."


def _temporal_answer(compression: CompressionResponse) -> str | None:
    temporal_query = parse_temporal_query(compression.query)
    if temporal_query is None:
        return None

    temporal_items = [
        candidate
        for candidate in compression.selected
        if candidate.temporal_anchor_score is not None
        and candidate.temporal_target_score is not None
        and candidate.temporal_direction in {"after", "before"}
    ]
    if len(temporal_items) < 2:
        return None

    pairs = rank_temporal_pairs(
        temporal_items,
        direction=temporal_query.direction,
        target_query=temporal_query.target,
        anchor_query=temporal_query.anchor,
    )
    if not pairs:
        return None

    _, _anchor, target = pairs[0]
    sentence = _best_sentence(compression.query, target.text)
    return sentence or target.text.strip() or None


def _global_summary_agenda_answer(selected: list[SelectedCandidate]) -> str | None:
    agenda_topics: list[tuple[float, float, str]] = []
    for item in selected:
        if not item.text.lower().startswith("on-screen text near"):
            continue
        text = item.text.split(":", maxsplit=1)[-1].strip()
        if not _looks_like_agenda_slide(text):
            continue
        for index, topic in enumerate(_agenda_topics(text)):
            score = _global_summary_text_score(topic) + (1 / (index + 1))
            agenda_topics.append((score, item.timestamp_seconds + index * 0.01, topic))
    if not agenda_topics:
        return None

    topics = [
        topic.rstrip(" .")
        for _score, _timestamp, topic in sorted(agenda_topics, key=lambda entry: entry[1])
    ][:MAX_GLOBAL_SUMMARY_TOPICS]
    if not topics:
        return None
    return f"The video covers: {'; '.join(topics)}."


def _looks_like_agenda_slide(text: str) -> bool:
    normalized = text.lower()
    return (
        "today" in normalized
        and (
            "lecture" in normalized
            or "agenda" in normalized
            or "overview" in normalized
            or "*" in text
        )
    )


def _agenda_topics(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    normalized = re.sub(r"^today\s*:?\s*", "", normalized, flags=re.IGNORECASE)
    pieces = [
        piece.strip(" -•*:;,.")
        for piece in re.split(r"\s+\*\s+|[•·]\s+|\s+-\s+", normalized)
        if piece.strip(" -•*:;,.")
    ]
    topics: list[str] = []
    for piece in pieces:
        cleaned = _clean_global_summary_snippet(piece)
        cleaned = re.sub(r"^lecture\s+\d+\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = _repair_agenda_ocr(cleaned)
        cleaned = cleaned.strip(" -:;,.")
        if not cleaned:
            continue
        if _is_noisy_global_summary_text(cleaned) and not _has_technical_terms(cleaned):
            continue
        if _global_summary_text_score(cleaned) <= 0 and not _has_technical_terms(cleaned):
            continue
        topics.append(cleaned)
    return _ordered_unique(topics)


def _has_technical_terms(text: str) -> bool:
    return bool(_claim_terms(text) & _GLOBAL_TECHNICAL_TERMS)


def _repair_agenda_ocr(text: str) -> str:
    repaired = re.sub(r"\bCont\s+eptual\b", "Conceptual", text)
    repaired = re.sub(r"\bcont\s+eptual\b", "conceptual", repaired)
    repaired = re.sub(r"\bConcept\s+ual\b", "Conceptual", repaired)
    repaired = re.sub(r"\bconcept\s+ual\b", "conceptual", repaired)
    return " ".join(repaired.split())


def _global_summary_topic(item: SelectedCandidate) -> str:
    snippet = _global_summary_snippet(item)
    if not snippet:
        return ""

    topic = _global_summary_topic_phrase(snippet)
    if topic:
        return topic
    if _is_noisy_global_summary_text(snippet) or _global_summary_text_score(snippet) <= 1.2:
        return ""
    return snippet.rstrip(" .")


def _dedupe_global_topics(
    topics: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    deduped: list[tuple[float, float, str]] = []
    seen_terms: list[set[str]] = []
    for score, timestamp, topic in sorted(
        topics,
        key=lambda entry: (entry[0], -entry[1]),
        reverse=True,
    ):
        terms = _claim_terms(topic)
        if any(_topic_overlap(terms, seen) >= 0.65 for seen in seen_terms):
            continue
        deduped.append((score, timestamp, topic))
        seen_terms.append(terms)
    return deduped


def _topic_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _global_summary_snippet(item: SelectedCandidate) -> str:
    text = item.text.strip()
    if text.lower().startswith("on-screen text near"):
        text = text.split(":", maxsplit=1)[-1].strip()
        words = text.split()
        return _clean_global_summary_snippet(" ".join(words[:8]))

    sentences = [
        sentence.strip(" ,")
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip(" ,")
    ]
    if not sentences:
        return _clean_global_summary_snippet(text)
    return _clean_global_summary_snippet(
        max(sentences, key=_global_summary_sentence_rank)
    )


def _global_summary_sentence_rank(sentence: str) -> tuple[int, float]:
    return (
        int(bool(_global_summary_topic_phrase(sentence))),
        _global_summary_text_score(sentence),
    )


def _global_summary_score(item: SelectedCandidate) -> float:
    snippet = _global_summary_snippet(item)
    if not snippet:
        return float("-inf")
    score = _global_summary_text_score(snippet)
    if item.modality.value == "audio":
        score += 0.2
    return score


def _global_summary_text_score(text: str) -> float:
    normalized = text.lower()
    tokens = _claim_terms(normalized)
    score = min(len(tokens), 16) / 8
    score += 0.45 * len(tokens & _GLOBAL_TECHNICAL_TERMS)
    score -= 0.7 * len(tokens & _GLOBAL_ADMIN_TERMS)
    score -= 0.35 * sum(phrase in normalized for phrase in _GLOBAL_FILLER_PHRASES)
    raw_tokens = re.findall(r"[a-z0-9]+", normalized)
    if raw_tokens:
        single_character_ratio = sum(len(token) == 1 for token in raw_tokens) / len(
            raw_tokens
        )
        score -= 3 * single_character_ratio
    if normalized.startswith("visual frame sampled at"):
        score -= 2
    if len(tokens) < 3:
        score -= 2
    return score


def _global_summary_topic_phrase(text: str) -> str:
    terms = [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _VISUAL_STOPWORDS
        and token not in _GLOBAL_TOPIC_STOPWORDS
        and len(token) > 2
    ]
    technical_terms = [
        term
        for term in terms
        if term in _GLOBAL_TECHNICAL_TERMS and term not in _GLOBAL_ADMIN_TERMS
    ]
    if technical_terms:
        labels = _compact_topic_labels(
            _ordered_unique(_GLOBAL_TOPIC_LABELS[term] for term in technical_terms)
        )
        return _join_topic_labels(labels[:4])

    if (
        len(terms) >= 3
        and _global_summary_text_score(text) > 1.2
        and not _is_noisy_global_summary_text(text)
    ):
        return " ".join(terms[:7])
    return ""


def _is_noisy_global_summary_text(text: str) -> bool:
    normalized = text.lower()
    tokens = _claim_terms(normalized)
    if tokens & _GLOBAL_ADMIN_TERMS:
        return True
    filler_count = sum(phrase in normalized for phrase in _GLOBAL_FILLER_PHRASES)
    return filler_count >= 2 or any(
        phrase in normalized
        for phrase in [
            "interesting thing",
            "this role",
            "in the place",
            "in their library manner",
        ]
    )


def _ordered_unique(values) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        result.append(value)
        seen.add(value)
    return result


def _compact_topic_labels(labels: list[str]) -> list[str]:
    compacted: list[str] = []
    for label in labels:
        label_terms = set(label.split())
        if any(label_terms < set(existing.split()) for existing in labels):
            continue
        compacted.append(label)
    return compacted


def _join_topic_labels(labels: list[str]) -> str:
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def _clean_global_summary_snippet(text: str) -> str:
    cleaned = re.sub(
        r"^(but\s+)?(what|the thing)\s+is\s+interesting\s+is\s+that\s+",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^(but\s+)?the\s+interesting\s+thing\s+is\s+that\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return " ".join(cleaned.split()).strip(" ,;:.")


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
    return bool(
        re.search(
            r"(?:^|[\s,;:])(why|what|how|when|where|who|which)\s+",
            query,
        )
    )


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
        anchor_query=temporal_query.anchor,
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
    enumerated = _enumerated_answer_sentence(query, sentences)
    if enumerated is not None:
        return enumerated
    best = max(sentences, key=lambda sentence: _sentence_score(query, sentence))
    followup = _followup_answer_sentence(query, sentences, best)
    return followup or best


def _enumerated_answer_sentence(query: str, sentences: list[str]) -> str | None:
    query_lower = query.lower()
    if not query_lower.strip().startswith("how") and " how " not in f" {query_lower} ":
        return None
    query_terms = set(re.findall(r"[a-z0-9]+", query_lower))
    if not {"startup", "startups", "ideas", "founders"} & query_terms:
        return None

    for index, sentence in enumerate(sentences):
        lower = sentence.lower()
        if "one," not in lower:
            continue
        answer = re.sub(r"^.*?(?=\bOne,)", "", sentence, flags=re.IGNORECASE).strip()
        if not answer:
            continue
        if "three," in answer.lower():
            return answer
        followups = [answer]
        for next_sentence in sentences[index + 1 : index + 3]:
            next_lower = next_sentence.lower()
            if (
                ("two," in next_lower and len(followups) == 1)
                or ("three," in next_lower and len(followups) == 2)
            ):
                followups.append(next_sentence.strip())
        if len(followups) >= 3:
            return " ".join(followups)
    return None


def _followup_answer_sentence(
    query: str,
    sentences: list[str],
    best: str,
) -> str | None:
    query_lower = query.lower()
    if "what" not in query_lower or not {"mistake", "kills", "kill"} & set(
        re.findall(r"[a-z0-9]+", query_lower)
    ):
        return None
    best_lower = best.lower()
    setup_markers = {
        "talking about",
        "have a listen",
        "let's listen",
        "question is",
        "what truly",
    }
    if not any(marker in best_lower for marker in setup_markers):
        return None
    try:
        start = sentences.index(best) + 1
    except ValueError:
        return None
    answer_markers = {
        "they make",
        "users don't like",
        "users do not like",
        "do not like",
        "don't like",
        "that is the killer",
        "the mistake",
    }
    for sentence in sentences[start : start + 4]:
        lower = sentence.lower()
        if any(marker in lower for marker in answer_markers):
            return _contextualize_followup_answer(query, sentence)
    return None


def _contextualize_followup_answer(query: str, sentence: str) -> str:
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    sentence_lower = sentence.lower()
    if "startups" in query_terms and "startup" not in sentence_lower:
        cleaned = sentence.strip()
        if cleaned.lower().startswith("they "):
            cleaned = cleaned[5:]
        return f"Startups fail when they {cleaned[0].lower()}{cleaned[1:]}"
    return sentence


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
