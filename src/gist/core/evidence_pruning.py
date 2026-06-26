import re
from dataclasses import dataclass

from gist.core.schemas import CompressionMetrics, CompressionResponse, Modality, SelectedCandidate
from gist.core.scoring import text_similarity
from gist.core.token_estimation import TOKEN_ESTIMATE_PROFILES

DEFAULT_MAX_PRUNED_EVIDENCE = 4
DEFAULT_MIN_PRUNED_EVIDENCE = 3
ANSWER_SUPPORT_RELATIVE_THRESHOLD = 0.55
MIN_CITED_SUPPORT_SCORE = 0.035
MIN_CITED_SUPPORT_RELATIVE_THRESHOLD = 0.35
STRONG_SUPPORT_SCORE = 0.12
MEDIUM_SUPPORT_SCORE = 0.05
REDUNDANT_EVIDENCE_SIMILARITY_THRESHOLD = 0.6
DIRECT_GROUNDING_SCORE = MEDIUM_SUPPORT_SCORE
CONTEXTUAL_GROUNDING_SCORE = 0.03


@dataclass(frozen=True, slots=True)
class EvidenceSupport:
    item: SelectedCandidate
    score: float
    answer_score: float
    query_score: float
    audio_score: float
    ocr_score: float
    visual_score: float
    cross_modal_score: float
    grounding_label: str
    grounding_reason: str


def annotate_evidence_support(compression: CompressionResponse) -> CompressionResponse:
    """Attach answer/query support metadata to every selected evidence item."""

    if not compression.selected:
        return compression
    selected = [
        _with_support_metadata(_support_score(compression, item))
        for item in compression.selected
    ]
    return compression.model_copy(update={"selected": selected})


def prune_evidence_to_answer(
    compression: CompressionResponse,
    max_items: int = DEFAULT_MAX_PRUNED_EVIDENCE,
    min_items: int = DEFAULT_MIN_PRUNED_EVIDENCE,
) -> CompressionResponse:
    """Keep final evidence that supports the generated answer and query."""

    if max_items <= 0:
        raise ValueError("max_items must be greater than zero")
    if min_items < 0:
        raise ValueError("min_items must be non-negative")
    if min_items > max_items:
        raise ValueError("min_items must not exceed max_items")
    if _is_visual_text_query(compression):
        max_items = min(max_items, 2)
        min_items = min(min_items, 1)
    if _is_mixed_speech_answer_query(compression):
        max_items = min(max_items, 2)
        min_items = min(min_items, 1)
    if not compression.answer or len(compression.selected) <= max_items:
        return compression

    ranked = sorted(
        (_support_score(compression, item) for item in compression.selected),
        key=lambda support: (
            support.score,
            support.item.relevance_score,
            -support.item.timestamp_seconds,
        ),
        reverse=True,
    )
    best_score = ranked[0].score if ranked else 0.0
    threshold = max(0.05, best_score * ANSWER_SUPPORT_RELATIVE_THRESHOLD)
    retained = [support for support in ranked if support.score >= threshold][:max_items]
    if len(retained) < min_items:
        retained = ranked[: min(min_items, len(ranked))]
    retained = _ensure_mixed_av_audio_retained(
        compression=compression,
        retained=retained,
        ranked=ranked,
        max_items=max_items,
    )
    retained = _ensure_global_summary_audio_coverage(
        compression=compression,
        retained=retained,
        ranked=ranked,
        max_items=max_items,
    )

    selected = [
        _with_pruning_reason(index, support)
        for index, support in enumerate(
            sorted(retained, key=lambda support: support.item.timestamp_seconds),
            start=1,
        )
    ]
    return compression.model_copy(
        update={
            "selected": selected,
            "metrics": _metrics_for_pruned_selection(compression.metrics, selected),
        }
    )


def prune_evidence_to_answer_citations(
    compression: CompressionResponse,
    min_items: int = 1,
) -> CompressionResponse:
    """Drop uncited evidence when the generated answer explicitly cites evidence ranks."""

    if min_items < 0:
        raise ValueError("min_items must be non-negative")
    if not compression.answer or len(compression.selected) <= min_items:
        return compression

    cited_ranks = _cited_evidence_ranks(compression.answer, len(compression.selected))
    if len(cited_ranks) < min_items:
        return compression

    ranked = [_support_score(compression, item) for item in compression.selected]
    cited = [
        support
        for rank, support in enumerate(ranked, start=1)
        if rank in cited_ranks
    ]
    retained = _support_filtered_citations(
        cited=cited,
        ranked=ranked,
        min_items=min_items,
    )
    retained = _ensure_mixed_av_audio_retained(
        compression=compression,
        retained=retained,
        ranked=ranked,
        max_items=len(compression.selected),
    )
    retained = _ensure_global_summary_audio_coverage(
        compression=compression,
        retained=retained,
        ranked=ranked,
        max_items=len(compression.selected),
    )

    selected = [
        _with_citation_reason(index, support)
        for index, support in enumerate(
            sorted(retained, key=lambda support: support.item.timestamp_seconds),
            start=1,
        )
    ]
    return compression.model_copy(
        update={
            "selected": selected,
            "metrics": _metrics_for_pruned_selection(compression.metrics, selected),
        }
    )


def prune_weakly_grounded_evidence(
    compression: CompressionResponse,
    min_items: int = 1,
) -> CompressionResponse:
    """Drop weakly grounded evidence when direct/contextual evidence remains."""

    if min_items < 0:
        raise ValueError("min_items must be non-negative")
    if len(compression.selected) <= min_items:
        return compression

    supports = [_support_score(compression, item) for item in compression.selected]
    retained = [support for support in supports if support.grounding_label != "weak"]
    if len(retained) < min_items or len(retained) == len(supports):
        return annotate_evidence_support(compression)
    retained = _ensure_mixed_av_audio_retained(
        compression=compression,
        retained=retained,
        ranked=supports,
        max_items=len(compression.selected),
    )
    retained = _ensure_global_summary_audio_coverage(
        compression=compression,
        retained=retained,
        ranked=supports,
        max_items=len(compression.selected),
    )

    selected = [
        _with_grounding_filter_reason(index, support)
        for index, support in enumerate(
            sorted(retained, key=lambda support: support.item.timestamp_seconds),
            start=1,
        )
    ]
    return compression.model_copy(
        update={
            "selected": selected,
            "metrics": _metrics_for_pruned_selection(compression.metrics, selected),
        }
    )


def consolidate_redundant_evidence(
    compression: CompressionResponse,
    similarity_threshold: float = REDUNDANT_EVIDENCE_SIMILARITY_THRESHOLD,
    min_items: int = 1,
) -> CompressionResponse:
    """Collapse selected evidence clips that support the same claim."""

    if similarity_threshold < 0 or similarity_threshold > 1:
        raise ValueError("similarity_threshold must be between 0 and 1")
    if min_items < 0:
        raise ValueError("min_items must be non-negative")
    if len(compression.selected) <= min_items:
        return compression

    groups = _redundant_evidence_groups(compression.selected, similarity_threshold)
    if len(groups) == len(compression.selected):
        return compression

    selected = [
        _with_consolidation_reason(index, _best_group_item(compression, group), len(group))
        for index, group in enumerate(
            sorted(
                groups,
                key=lambda group: _best_group_item(compression, group).timestamp_seconds,
            ),
            start=1,
        )
    ]
    if len(selected) < min_items:
        return compression
    selected = _ensure_mixed_av_audio_selected(
        compression=compression,
        selected=selected,
    )
    selected = _ensure_global_summary_audio_selected(
        compression=compression,
        selected=selected,
    )
    return compression.model_copy(
        update={
            "selected": selected,
            "metrics": _metrics_for_pruned_selection(compression.metrics, selected),
        }
    )


def _support_score(
    compression: CompressionResponse,
    item: SelectedCandidate,
) -> EvidenceSupport:
    answer_similarity = text_similarity(compression.answer or "", item.text)
    query_similarity = text_similarity(compression.query, item.text)
    audio_score = _audio_support_score(item, answer_similarity, query_similarity)
    ocr_score = _ocr_support_score(item, answer_similarity, query_similarity)
    visual_score = _visual_support_score(compression, item, query_similarity)
    cross_modal_score = _cross_modal_support_score(item, audio_score, visual_score)
    modality_score = max(audio_score, ocr_score, visual_score)
    text_score = (0.7 * answer_similarity) + (0.3 * query_similarity)
    score = _combined_support_score(
        compression=compression,
        item=item,
        text_score=text_score,
        modality_score=modality_score,
        cross_modal_score=cross_modal_score,
    )
    return EvidenceSupport(
        item=item,
        score=score,
        answer_score=answer_similarity,
        query_score=query_similarity,
        audio_score=audio_score,
        ocr_score=ocr_score,
        visual_score=visual_score,
        cross_modal_score=cross_modal_score,
        grounding_label=_grounding_label(
            audio_score=audio_score,
            ocr_score=ocr_score,
            visual_score=visual_score,
            cross_modal_score=cross_modal_score,
        ),
        grounding_reason=_grounding_reason(
            item=item,
            answer_score=answer_similarity,
            query_score=query_similarity,
            audio_score=audio_score,
            ocr_score=ocr_score,
            visual_score=visual_score,
            cross_modal_score=cross_modal_score,
        ),
    )


def _support_filtered_citations(
    cited: list[EvidenceSupport],
    ranked: list[EvidenceSupport],
    min_items: int,
) -> list[EvidenceSupport]:
    if not cited:
        return []

    best_score = max((support.score for support in ranked), default=0.0)
    threshold = max(
        MIN_CITED_SUPPORT_SCORE,
        best_score * MIN_CITED_SUPPORT_RELATIVE_THRESHOLD,
    )
    target_count = max(min_items, len(cited))
    retained = [support for support in cited if support.score >= threshold]
    if len(retained) >= target_count:
        return retained

    retained_ids = {support.item.id for support in retained}
    alternatives = [
        support
        for support in sorted(
            ranked,
            key=lambda support: (
                support.score,
                support.item.relevance_score,
                -support.item.timestamp_seconds,
            ),
            reverse=True,
        )
        if support.score >= threshold and support.item.id not in retained_ids
    ]
    retained.extend(alternatives[: max(target_count - len(retained), 0)])
    if len(retained) >= target_count:
        return retained
    return cited


def _ensure_mixed_av_audio_retained(
    compression: CompressionResponse,
    retained: list[EvidenceSupport],
    ranked: list[EvidenceSupport],
    max_items: int,
) -> list[EvidenceSupport]:
    if not _is_mixed_av_query(compression) or _has_audio_evidence(retained):
        return retained

    best_audio = _best_audio_support(ranked)
    if best_audio is None:
        return retained

    updated = [*retained, best_audio]
    if len(updated) <= max_items:
        return updated

    removable = [
        support
        for support in updated
        if support.item.id != best_audio.item.id and support.item.modality != Modality.AUDIO
    ]
    if not removable:
        return updated[:max_items]

    weakest = min(
        removable,
        key=lambda support: (
            support.score,
            support.item.relevance_score,
            support.item.normalized_score,
        ),
    )
    return [support for support in updated if support.item.id != weakest.item.id]


def _ensure_mixed_av_audio_selected(
    compression: CompressionResponse,
    selected: list[SelectedCandidate],
) -> list[SelectedCandidate]:
    has_audio = any(item.modality == Modality.AUDIO for item in selected)
    if not _is_mixed_av_query(compression) or has_audio:
        return selected

    audio_items = [item for item in compression.selected if item.modality == Modality.AUDIO]
    if not audio_items:
        return selected

    best_audio = max(
        audio_items,
        key=lambda item: (
            item.evidence_support_score or 0.0,
            item.relevance_score,
            item.normalized_score,
        ),
    )
    return sorted([*selected, best_audio], key=lambda item: item.timestamp_seconds)


def _ensure_global_summary_audio_coverage(
    compression: CompressionResponse,
    retained: list[EvidenceSupport],
    ranked: list[EvidenceSupport],
    max_items: int,
) -> list[EvidenceSupport]:
    if not _is_global_summary_query(compression):
        return retained

    representatives = _global_summary_audio_representatives(ranked)
    if not representatives:
        return retained

    updated = list(retained)
    for representative in representatives:
        if any(support.item.id == representative.item.id for support in updated):
            continue
        updated.append(representative)
        if len(updated) > max_items:
            removable = _global_summary_removable_supports(updated, representatives)
            if not removable:
                updated = updated[:max_items]
                break
            weakest = min(
                removable,
                key=lambda support: (
                    support.score,
                    support.item.relevance_score,
                    support.item.normalized_score,
                ),
            )
            updated = [
                support for support in updated if support.item.id != weakest.item.id
            ]
    return updated


def _ensure_global_summary_audio_selected(
    compression: CompressionResponse,
    selected: list[SelectedCandidate],
) -> list[SelectedCandidate]:
    if not _is_global_summary_query(compression):
        return selected

    supports = [_support_score(compression, item) for item in compression.selected]
    retained = [_support_score(compression, item) for item in selected]
    balanced = _ensure_global_summary_audio_coverage(
        compression=compression,
        retained=retained,
        ranked=supports,
        max_items=len(compression.selected),
    )
    return sorted(
        [_with_support_metadata(support) for support in balanced],
        key=lambda item: item.timestamp_seconds,
    )


def _global_summary_audio_representatives(
    supports: list[EvidenceSupport],
) -> list[EvidenceSupport]:
    audio_supports = [
        support
        for support in supports
        if support.item.modality == Modality.AUDIO
        and not _is_visual_only_text(support.item.text)
    ]
    if not audio_supports:
        return []

    start = min(support.item.timestamp_seconds for support in audio_supports)
    end = max(support.item.timestamp_seconds for support in audio_supports)
    span = max(end - start, 1.0)

    def bucket_index(support: EvidenceSupport) -> int:
        offset = (support.item.timestamp_seconds - start) / span
        return min(int(offset * 3), 2)

    representatives: list[EvidenceSupport] = []
    for bucket in range(3):
        bucket_supports = [
            support for support in audio_supports if bucket_index(support) == bucket
        ]
        if not bucket_supports:
            continue
        representatives.append(
            max(
                bucket_supports,
                key=lambda support: (
                    _global_summary_audio_content_score(support),
                    support.score,
                    support.item.relevance_score,
                    support.item.normalized_score,
                ),
            )
        )
    return representatives


def _global_summary_audio_content_score(support: EvidenceSupport) -> float:
    text = support.item.text.lower()
    content_terms = {
        token
        for token in re.findall(r"[a-z][a-z0-9-]{2,}", text)
        if token
        not in {
            "and",
            "are",
            "for",
            "have",
            "that",
            "the",
            "this",
            "with",
            "you",
        }
    }
    technical_terms = {
        "biology",
        "control",
        "locomotion",
        "motor",
        "robot",
        "robotics",
        "sensor",
        "sensors",
        "velocity",
    }
    admin_terms = {
        "assignment",
        "course",
        "deadline",
        "evaluation",
        "lecture",
        "objective",
        "schedule",
        "seminar",
    }
    return (
        min(len(content_terms), 20) / 10
        + 0.4 * len(content_terms & technical_terms)
        - 0.5 * len(content_terms & admin_terms)
    )


def _global_summary_removable_supports(
    supports: list[EvidenceSupport],
    representatives: list[EvidenceSupport],
) -> list[EvidenceSupport]:
    representative_ids = {support.item.id for support in representatives}
    removable_visuals = [
        support
        for support in supports
        if support.item.id not in representative_ids
        and support.item.modality != Modality.AUDIO
    ]
    if removable_visuals:
        return removable_visuals
    return [
        support
        for support in supports
        if support.item.id not in representative_ids
    ]


def _is_global_summary_query(compression: CompressionResponse) -> bool:
    return getattr(compression.query_intent, "value", None) == "global_summary"


def _is_mixed_av_query(compression: CompressionResponse) -> bool:
    return getattr(compression.query_intent, "value", None) == "mixed_av"


def _has_audio_evidence(supports: list[EvidenceSupport]) -> bool:
    return any(support.item.modality == Modality.AUDIO for support in supports)


def _best_audio_support(supports: list[EvidenceSupport]) -> EvidenceSupport | None:
    audio_supports = [
        support
        for support in supports
        if support.item.modality == Modality.AUDIO and not _is_visual_only_text(support.item.text)
    ]
    if not audio_supports:
        return None
    return max(
        audio_supports,
        key=lambda support: (
            support.score,
            support.item.relevance_score,
            support.item.normalized_score,
        ),
    )


def _with_support_metadata(support: EvidenceSupport) -> SelectedCandidate:
    return support.item.model_copy(
        update={
            "answer_support_score": support.answer_score,
            "query_support_score": support.query_score,
            "evidence_support_score": support.score,
            "audio_support_score": support.audio_score,
            "ocr_support_score": support.ocr_score,
            "visual_support_score": support.visual_score,
            "cross_modal_support_score": support.cross_modal_score,
            "support_label": _support_label(support.score),
            "grounding_label": support.grounding_label,
            "grounding_reason": support.grounding_reason,
        }
    )


def _grounding_label(
    audio_score: float,
    ocr_score: float,
    visual_score: float,
    cross_modal_score: float,
) -> str:
    direct_score = max(audio_score, ocr_score, visual_score)
    if direct_score >= DIRECT_GROUNDING_SCORE:
        return "direct"
    if cross_modal_score >= CONTEXTUAL_GROUNDING_SCORE:
        return "contextual"
    return "weak"


def _grounding_reason(
    item: SelectedCandidate,
    answer_score: float,
    query_score: float,
    audio_score: float,
    ocr_score: float,
    visual_score: float,
    cross_modal_score: float,
) -> str:
    if audio_score >= DIRECT_GROUNDING_SCORE:
        return (
            "direct transcript support "
            f"(answer={answer_score:.3f}, query={query_score:.3f})"
        )
    if ocr_score >= DIRECT_GROUNDING_SCORE:
        return (
            "direct OCR/text support "
            f"(answer={answer_score:.3f}, query={query_score:.3f})"
        )
    if visual_score >= DIRECT_GROUNDING_SCORE:
        return f"direct visual support (visual={visual_score:.3f})"
    if cross_modal_score >= CONTEXTUAL_GROUNDING_SCORE:
        return (
            "contextual cross-modal support "
            f"(anchor={item.audio_anchor_score:.3f}, cross_modal={cross_modal_score:.3f})"
        )
    return (
        "weak grounding: no transcript, OCR, visual, or cross-modal score reached "
        f"{CONTEXTUAL_GROUNDING_SCORE:.3f}"
    )


def _audio_support_score(
    item: SelectedCandidate,
    answer_similarity: float,
    query_similarity: float,
) -> float:
    if item.modality != Modality.AUDIO or _is_visual_only_text(item.text):
        return 0.0
    return (0.7 * answer_similarity) + (0.3 * query_similarity)


def _ocr_support_score(
    item: SelectedCandidate,
    answer_similarity: float,
    query_similarity: float,
) -> float:
    if not _is_ocr_text(item.text):
        return 0.0
    return (0.65 * answer_similarity) + (0.35 * query_similarity)


def _visual_support_score(
    compression: CompressionResponse,
    item: SelectedCandidate,
    query_similarity: float,
) -> float:
    if item.modality != Modality.VISUAL:
        return 0.0
    if _should_demote_visual_support(compression):
        return min(query_similarity, 0.04)
    if _is_visual_text_query(compression):
        return min(query_similarity, 0.2) if _is_ocr_text(item.text) else 0.0
    score = max(item.relevance_score, item.normalized_score, query_similarity)
    if _is_ocr_text(item.text):
        score = max(score, query_similarity + 0.05)
    if item.audio_anchor_score > 0:
        score = max(score, min(item.audio_anchor_score, 1.0) * 0.5)
    return min(score, 1.0)


def _cross_modal_support_score(
    item: SelectedCandidate,
    audio_score: float,
    visual_score: float,
) -> float:
    if item.audio_anchor_timestamp_seconds is None and item.audio_anchor_score <= 0:
        return 0.0
    anchor_score = min(max(item.audio_anchor_score, 0.0), 1.0)
    if item.modality == Modality.VISUAL:
        return max(anchor_score, min(visual_score + 0.1, 1.0))
    return max(anchor_score * 0.5, audio_score * 0.25)


def _combined_support_score(
    compression: CompressionResponse,
    item: SelectedCandidate,
    text_score: float,
    modality_score: float,
    cross_modal_score: float,
) -> float:
    if _should_demote_visual_support(compression) and item.modality == Modality.VISUAL:
        return max(text_score, min(cross_modal_score, 0.04))
    return max(text_score, (0.75 * modality_score) + (0.25 * cross_modal_score))


def _should_demote_visual_support(compression: CompressionResponse) -> bool:
    return _is_transcript_first_query(compression) or _is_mixed_speech_answer_query(compression)


def _is_transcript_first_query(compression: CompressionResponse) -> bool:
    return getattr(compression.query_intent, "value", None) == "speech_semantic"


def _is_mixed_speech_answer_query(compression: CompressionResponse) -> bool:
    if not _is_mixed_av_query(compression):
        return False
    query = f" {compression.query.lower()} "
    return any(
        marker in query
        for marker in [
            " say ",
            " says ",
            " said ",
            " tell ",
            " tells ",
            " told ",
            " explain ",
            " explains ",
            " explained ",
            " presenter ",
            " speaker ",
        ]
    )


def _is_ocr_text(text: str) -> bool:
    return text.lower().startswith("on-screen text near")


def _is_visual_text_query(compression: CompressionResponse) -> bool:
    query = f" {compression.query.lower()} "
    return any(
        marker in query
        for marker in [
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


def _is_visual_only_text(text: str) -> bool:
    normalized = text.lower()
    return normalized.startswith("visual frame sampled at") or _is_ocr_text(normalized)


def _support_label(score: float) -> str:
    if score >= STRONG_SUPPORT_SCORE:
        return "strong"
    if score >= MEDIUM_SUPPORT_SCORE:
        return "medium"
    return "weak"


def _with_pruning_reason(index: int, support: EvidenceSupport) -> SelectedCandidate:
    reason = (
        f"{support.item.reason}; retained after answer-grounded pruning "
        f"(support score {support.score:.3f})"
    )
    return _with_support_metadata(support).model_copy(
        update={"selection_rank": index, "reason": reason}
    )


def _with_citation_reason(index: int, support: EvidenceSupport) -> SelectedCandidate:
    item = _with_support_metadata(support)
    reason = f"{item.reason}; retained because the final answer cited this evidence"
    return item.model_copy(update={"selection_rank": index, "reason": reason})


def _with_grounding_filter_reason(index: int, support: EvidenceSupport) -> SelectedCandidate:
    item = _with_support_metadata(support)
    reason = (
        f"{item.reason}; retained after grounding filter "
        f"({support.grounding_label}: {support.grounding_reason})"
    )
    return item.model_copy(update={"selection_rank": index, "reason": reason})


def _with_consolidation_reason(
    index: int,
    item: SelectedCandidate,
    group_size: int,
) -> SelectedCandidate:
    if group_size <= 1:
        return item.model_copy(update={"selection_rank": index})
    reason = (
        f"{item.reason}; retained as strongest representative of "
        f"{group_size} redundant evidence clips"
    )
    return item.model_copy(update={"selection_rank": index, "reason": reason})


def _redundant_evidence_groups(
    selected: list[SelectedCandidate],
    similarity_threshold: float,
) -> list[list[SelectedCandidate]]:
    groups: list[list[SelectedCandidate]] = []
    for item in selected:
        matching_group = None
        for group in groups:
            if any(
                _is_redundant_evidence_pair(
                    item,
                    existing,
                    similarity_threshold=similarity_threshold,
                )
                for existing in group
            ):
                matching_group = group
                break
        if matching_group is None:
            groups.append([item])
        else:
            matching_group.append(item)
    return groups


def _is_redundant_evidence_pair(
    left: SelectedCandidate,
    right: SelectedCandidate,
    similarity_threshold: float,
) -> bool:
    if text_similarity(left.text, right.text) >= similarity_threshold:
        return True
    return left.modality == right.modality and _clip_overlap_ratio(left, right) >= 0.5


def _clip_overlap_ratio(left: SelectedCandidate, right: SelectedCandidate) -> float:
    left_start, left_end = _evidence_span(left)
    right_start, right_end = _evidence_span(right)
    overlap = max(min(left_end, right_end) - max(left_start, right_start), 0.0)
    shortest = min(left_end - left_start, right_end - right_start)
    if shortest <= 0:
        return 0.0
    return overlap / shortest


def _evidence_span(item: SelectedCandidate) -> tuple[float, float]:
    start = item.clip_start_seconds
    end = item.clip_end_seconds
    if start is None or end is None:
        start = item.scene_start_seconds
        end = item.scene_end_seconds
    if start is None or end is None:
        start = item.timestamp_seconds
        end = item.timestamp_seconds
    return min(start, end), max(start, end)


def _best_group_item(
    compression: CompressionResponse,
    group: list[SelectedCandidate],
) -> SelectedCandidate:
    return max(
        group,
        key=lambda item: (
            _support_score(compression, item).score,
            item.relevance_score,
            -item.timestamp_seconds,
        ),
    )


def _cited_evidence_ranks(answer: str, selected_count: int) -> set[int]:
    lower = answer.lower()
    if "evidence" not in lower:
        return set()

    cited: set[int] = set()
    for citation in re.finditer(
        r"(?is)\bevidences?\s*(?:number\s*)?#?\s*[:.]?\s*((?:\d+|and|,|\s)+)",
        answer,
    ):
        cited.update(int(value) for value in re.findall(r"\d+", citation.group(1)))

    evidence_section = re.search(r"(?is)\bevidence\s*:\s*(.+)$", answer)
    if evidence_section is not None:
        for match in re.finditer(r"(?m)^\s*(\d+)[.)]\s+", evidence_section.group(1)):
            cited.add(int(match.group(1)))

    return {rank for rank in cited if 1 <= rank <= selected_count}


def _metrics_for_pruned_selection(
    metrics: CompressionMetrics,
    selected: list[SelectedCandidate],
) -> CompressionMetrics:
    selected_count = len(selected)
    visual_selected = sum(item.modality == Modality.VISUAL for item in selected)
    audio_selected = sum(item.modality == Modality.AUDIO for item in selected)
    reduction_ratio = (
        1.0 if metrics.input_candidates == 0 else selected_count / metrics.input_candidates
    )
    baseline_tokens = metrics.estimated_baseline_tokens
    config = TOKEN_ESTIMATE_PROFILES[metrics.token_estimator]
    compressed_tokens = sum(
        config.visual_tokens_per_candidate
        if item.modality == Modality.VISUAL
        else config.audio_tokens_per_candidate
        for item in selected
    )
    saved_tokens = max(baseline_tokens - compressed_tokens, 0)
    token_reduction_ratio = 0.0 if baseline_tokens == 0 else compressed_tokens / baseline_tokens
    token_reduction_percent = (
        (1.0 - token_reduction_ratio) * 100 if baseline_tokens else 0.0
    )

    updates = {
        "selected_candidates": selected_count,
        "visual_selected": visual_selected,
        "audio_selected": audio_selected,
        "estimated_candidate_reduction_ratio": reduction_ratio,
        "estimated_candidate_reduction_percent": (1.0 - reduction_ratio) * 100,
        "dropped_candidates": max(metrics.input_candidates - selected_count, 0),
        "estimated_compressed_tokens": compressed_tokens,
        "estimated_saved_tokens": saved_tokens,
        "estimated_token_reduction_ratio": token_reduction_ratio,
        "estimated_token_reduction_percent": token_reduction_percent,
    }
    return metrics.model_copy(update=updates)
