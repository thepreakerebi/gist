from dataclasses import dataclass
import re

from gist.core.schemas import CompressionMetrics, CompressionResponse, Modality, SelectedCandidate
from gist.core.scoring import text_similarity
from gist.core.token_estimation import TOKEN_ESTIMATE_PROFILES


DEFAULT_MAX_PRUNED_EVIDENCE = 4
DEFAULT_MIN_PRUNED_EVIDENCE = 3
ANSWER_SUPPORT_RELATIVE_THRESHOLD = 0.55


@dataclass(frozen=True, slots=True)
class EvidenceSupport:
    item: SelectedCandidate
    score: float


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

    selected = [
        _with_citation_reason(index, item)
        for index, item in enumerate(
            [
                item
                for rank, item in enumerate(compression.selected, start=1)
                if rank in cited_ranks
            ],
            start=1,
        )
    ]
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
    score = (0.7 * answer_similarity) + (0.3 * query_similarity)
    return EvidenceSupport(item=item, score=score)


def _with_pruning_reason(index: int, support: EvidenceSupport) -> SelectedCandidate:
    reason = (
        f"{support.item.reason}; retained after answer-grounded pruning "
        f"(support score {support.score:.3f})"
    )
    return support.item.model_copy(update={"selection_rank": index, "reason": reason})


def _with_citation_reason(index: int, item: SelectedCandidate) -> SelectedCandidate:
    reason = f"{item.reason}; retained because the final answer cited this evidence"
    return item.model_copy(update={"selection_rank": index, "reason": reason})


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
