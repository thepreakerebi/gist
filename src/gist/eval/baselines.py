from gist.core.presets import PRESETS, CompressionPreset
from gist.core.schemas import Candidate, Modality, SelectedCandidate
from gist.eval.metrics import modality_coverage, reduction_percent, timestamp_hit_rate
from gist.eval.schemas import BaselineResult, EvalExample


def uniform_baseline(example: EvalExample, preset: CompressionPreset) -> BaselineResult:
    budget = PRESETS[preset].max_items
    selected = _uniform_select(
        visual_candidates=example.visual_candidates,
        audio_candidates=example.audio_candidates,
        budget=budget,
    )
    input_count = len(example.visual_candidates) + len(example.audio_candidates)
    return BaselineResult(
        name="uniform",
        selected=selected,
        selected_candidates=len(selected),
        reduction_percent=reduction_percent(input_count, len(selected)),
        timestamp_hit_rate=timestamp_hit_rate(
            selected,
            example.relevant_timestamps,
            example.timestamp_tolerance_seconds,
        ),
        modality_coverage=modality_coverage(selected),
    )


def score_topk_baseline(example: EvalExample, preset: CompressionPreset) -> BaselineResult:
    budget = PRESETS[preset].max_items
    selected = _score_topk_select(
        visual_candidates=example.visual_candidates,
        audio_candidates=example.audio_candidates,
        budget=budget,
    )
    input_count = len(example.visual_candidates) + len(example.audio_candidates)
    return BaselineResult(
        name="score_topk",
        selected=selected,
        selected_candidates=len(selected),
        reduction_percent=reduction_percent(input_count, len(selected)),
        timestamp_hit_rate=timestamp_hit_rate(
            selected,
            example.relevant_timestamps,
            example.timestamp_tolerance_seconds,
        ),
        modality_coverage=modality_coverage(selected),
    )


def _uniform_select(
    visual_candidates: list[Candidate],
    audio_candidates: list[Candidate],
    budget: int,
) -> list[SelectedCandidate]:
    all_candidates = [
        (Modality.VISUAL, candidate) for candidate in visual_candidates
    ] + [
        (Modality.AUDIO, candidate) for candidate in audio_candidates
    ]
    all_candidates = sorted(all_candidates, key=lambda item: (item[1].timestamp_seconds, item[1].id))
    if not all_candidates or budget <= 0:
        return []
    if len(all_candidates) <= budget:
        indexes = list(range(len(all_candidates)))
    else:
        step = (len(all_candidates) - 1) / (budget - 1) if budget > 1 else 0
        indexes = [round(index * step) for index in range(budget)]

    selected: list[SelectedCandidate] = []
    for rank, index in enumerate(indexes, start=1):
        modality, candidate = all_candidates[index]
        selected.append(
            SelectedCandidate(
                id=candidate.id,
                modality=modality,
                timestamp_seconds=candidate.timestamp_seconds,
                text=candidate.text,
                selection_rank=rank,
                relevance_score=0.0,
                normalized_score=0.0,
                mmr_score=0.0,
                source_score_type="uniform",
                reason="Uniform baseline selection by timestamp spacing.",
            )
        )
    return selected


def _score_topk_select(
    visual_candidates: list[Candidate],
    audio_candidates: list[Candidate],
    budget: int,
) -> list[SelectedCandidate]:
    all_candidates = [
        (Modality.VISUAL, candidate) for candidate in visual_candidates
    ] + [
        (Modality.AUDIO, candidate) for candidate in audio_candidates
    ]
    ranked = sorted(
        all_candidates,
        key=lambda item: (
            item[1].saliency_score if item[1].saliency_score is not None else -1.0,
            -item[1].timestamp_seconds,
            item[1].id,
        ),
        reverse=True,
    )
    selected: list[SelectedCandidate] = []
    for rank, (modality, candidate) in enumerate(ranked[: max(budget, 0)], start=1):
        score = candidate.saliency_score or 0.0
        selected.append(
            SelectedCandidate(
                id=candidate.id,
                modality=modality,
                timestamp_seconds=candidate.timestamp_seconds,
                text=candidate.text,
                asset_path=candidate.asset_path,
                segment_id=candidate.segment_id,
                scene_start_seconds=candidate.scene_start_seconds,
                scene_end_seconds=candidate.scene_end_seconds,
                selection_rank=rank,
                relevance_score=score,
                normalized_score=score,
                mmr_score=score,
                source_score_type="score_topk",
                reason="Score Top-K baseline selection by candidate saliency.",
            )
        )
    return sorted(selected, key=lambda item: item.timestamp_seconds)
