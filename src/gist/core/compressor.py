from dataclasses import dataclass

from gist.core.decomposition import (
    QueryAspect,
    QueryAspectModality,
    RuleBasedQueryDecomposer,
)
from gist.core.presets import PRESETS, CompressionPreset
from gist.core.schemas import (
    Candidate,
    CompressionMetrics,
    CompressionRequest,
    CompressionResponse,
    Modality,
    SelectedCandidate,
)
from gist.core.scoring import lexical_relevance, temporal_similarity, z_scores
from gist.core.token_estimation import estimate_tokens


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    id: str
    modality: Modality
    timestamp_seconds: float
    text: str
    relevance_score: float
    normalized_score: float
    source_score_type: str
    aspect: str


@dataclass(frozen=True, slots=True)
class Selection:
    candidate: ScoredCandidate
    selection_rank: int
    mmr_score: float
    reason: str


class GistCompressor:
    def __init__(self) -> None:
        self.query_decomposer = RuleBasedQueryDecomposer()

    def compress(self, request: CompressionRequest) -> CompressionResponse:
        query_aspects = self._query_aspects_for(request)
        scored = self._score_candidates(request, query_aspects)
        preset, selections, expansion_reason = self._select_for_budget(request, scored)

        return self._build_response(
            request=request,
            preset=preset,
            query_aspects=query_aspects,
            selections=selections,
            budget_mode="adaptive" if request.adaptive_budget else "fixed",
            budget_expanded=expansion_reason is not None,
            expansion_reason=expansion_reason,
        )

    def _select_for_budget(
        self,
        request: CompressionRequest,
        scored: list[ScoredCandidate],
    ) -> tuple[CompressionPreset, list[Selection], str | None]:
        if not request.adaptive_budget:
            return request.preset, self._select_with_preset(request.preset, scored), None

        aggressive = self._select_with_preset(CompressionPreset.AGGRESSIVE, scored)
        should_expand, reason = self._should_expand_budget(aggressive)
        if not should_expand:
            return CompressionPreset.AGGRESSIVE, aggressive, None

        expanded_preset = (
            CompressionPreset.CONSERVATIVE
            if request.preset == CompressionPreset.CONSERVATIVE
            else CompressionPreset.BALANCED
        )
        return expanded_preset, self._select_with_preset(expanded_preset, scored), reason

    def _select_with_preset(
        self,
        preset: CompressionPreset,
        scored: list[ScoredCandidate],
    ) -> list[Selection]:
        config = PRESETS[preset]
        return self._select_with_mmr(
            candidates=scored,
            max_items=config.max_items,
            relevance_weight=config.relevance_weight,
            temporal_sigma_seconds=config.temporal_sigma_seconds,
        )

    def _build_response(
        self,
        request: CompressionRequest,
        preset: CompressionPreset,
        query_aspects: list[QueryAspect],
        selections: list[Selection],
        budget_mode: str,
        budget_expanded: bool,
        expansion_reason: str | None,
    ) -> CompressionResponse:
        input_count = len(request.visual_candidates) + len(request.audio_candidates)
        selected_count = len(selections)
        reduction_ratio = 1.0 if input_count == 0 else selected_count / input_count
        reduction_percent = (1.0 - reduction_ratio) * 100
        dropped_count = max(input_count - selected_count, 0)
        token_estimate = estimate_tokens(
            input_visual_candidates=len(request.visual_candidates),
            input_audio_candidates=len(request.audio_candidates),
            selected_modalities=[selection.candidate.modality for selection in selections],
            profile=request.token_estimator,
        )

        return CompressionResponse(
            video_id=request.video_id,
            query=request.query,
            preset=preset,
            query_aspects=query_aspects,
            selected=[
                SelectedCandidate(
                    id=selection.candidate.id,
                    modality=selection.candidate.modality,
                    timestamp_seconds=selection.candidate.timestamp_seconds,
                    text=selection.candidate.text,
                    selection_rank=selection.selection_rank,
                    relevance_score=selection.candidate.relevance_score,
                    normalized_score=selection.candidate.normalized_score,
                    mmr_score=selection.mmr_score,
                    source_score_type=selection.candidate.source_score_type,
                    reason=selection.reason,
                )
                for selection in sorted(
                    selections,
                    key=lambda item: item.candidate.timestamp_seconds,
                )
            ],
            metrics=CompressionMetrics(
                input_candidates=input_count,
                selected_candidates=selected_count,
                visual_selected=sum(
                    item.candidate.modality == Modality.VISUAL for item in selections
                ),
                audio_selected=sum(
                    item.candidate.modality == Modality.AUDIO for item in selections
                ),
                estimated_candidate_reduction_ratio=reduction_ratio,
                estimated_candidate_reduction_percent=reduction_percent,
                dropped_candidates=dropped_count,
                budget_mode=budget_mode,
                budget_preset_used=preset,
                budget_expanded=budget_expanded,
                expansion_reason=expansion_reason,
                estimated_baseline_tokens=token_estimate.baseline_tokens,
                estimated_compressed_tokens=token_estimate.compressed_tokens,
                estimated_saved_tokens=token_estimate.saved_tokens,
                estimated_token_reduction_ratio=token_estimate.reduction_ratio,
                estimated_token_reduction_percent=token_estimate.reduction_percent,
                token_estimator=token_estimate.profile,
            ),
        )

    def _should_expand_budget(self, selections: list[Selection]) -> tuple[bool, str | None]:
        if not selections:
            return True, "no evidence selected at aggressive budget"

        best_relevance = max(selection.candidate.relevance_score for selection in selections)
        if best_relevance < 0.15:
            return True, "low best relevance at aggressive budget"

        modalities = {selection.candidate.modality for selection in selections}
        if len(selections) > 1 and len(modalities) == 1:
            return True, "aggressive budget selected only one modality"

        return False, None

    def _query_aspects_for(self, request: CompressionRequest) -> list[QueryAspect]:
        if request.decompose_query:
            return self.query_decomposer.decompose(request.query)
        return [QueryAspect(text=request.query)]

    def _score_candidates(
        self,
        request: CompressionRequest,
        query_aspects: list[QueryAspect],
    ) -> list[ScoredCandidate]:
        visual: list[ScoredCandidate] = []
        audio: list[ScoredCandidate] = []
        for aspect in query_aspects:
            if aspect.modality in {QueryAspectModality.VISUAL, QueryAspectModality.BOTH}:
                visual.extend(
                    self._score_modality(aspect.text, request.visual_candidates, Modality.VISUAL)
                )
            if aspect.modality in {QueryAspectModality.AUDIO, QueryAspectModality.BOTH}:
                audio.extend(
                    self._score_modality(aspect.text, request.audio_candidates, Modality.AUDIO)
                )
        return self._collapse_duplicate_scores(visual + audio)

    def _score_modality(
        self,
        query: str,
        candidates: list[Candidate],
        modality: Modality,
    ) -> list[ScoredCandidate]:
        raw_scores = [lexical_relevance(query, candidate) for candidate in candidates]
        normalized_scores = z_scores(raw_scores)

        return [
            ScoredCandidate(
                id=candidate.id,
                modality=modality,
                timestamp_seconds=candidate.timestamp_seconds,
                text=candidate.text,
                relevance_score=raw_score,
                normalized_score=normalized_score,
                source_score_type="model_saliency"
                if candidate.saliency_score is not None
                else "lexical_overlap",
                aspect=query,
            )
            for candidate, raw_score, normalized_score in zip(
                candidates,
                raw_scores,
                normalized_scores,
                strict=True,
            )
        ]

    def _collapse_duplicate_scores(
        self,
        candidates: list[ScoredCandidate],
    ) -> list[ScoredCandidate]:
        best_by_id: dict[str, ScoredCandidate] = {}
        for candidate in candidates:
            current = best_by_id.get(candidate.id)
            if current is None or candidate.normalized_score > current.normalized_score:
                best_by_id[candidate.id] = candidate
        return list(best_by_id.values())

    def _select_with_mmr(
        self,
        candidates: list[ScoredCandidate],
        max_items: int,
        relevance_weight: float,
        temporal_sigma_seconds: float,
    ) -> list[Selection]:
        selected: list[Selection] = []
        remaining = sorted(candidates, key=lambda item: (item.timestamp_seconds, item.id))

        while remaining and len(selected) < max_items:
            best = max(
                remaining,
                key=lambda item: self._mmr_score(
                    item=item,
                    selected=[selection.candidate for selection in selected],
                    relevance_weight=relevance_weight,
                    temporal_sigma_seconds=temporal_sigma_seconds,
                ),
            )
            mmr_score = self._mmr_score(
                item=best,
                selected=[selection.candidate for selection in selected],
                relevance_weight=relevance_weight,
                temporal_sigma_seconds=temporal_sigma_seconds,
            )
            selected.append(
                Selection(
                    candidate=best,
                    selection_rank=len(selected) + 1,
                    mmr_score=mmr_score,
                    reason=self._selection_reason(best, selected),
                )
            )
            remaining.remove(best)

        return selected

    def _mmr_score(
        self,
        item: ScoredCandidate,
        selected: list[ScoredCandidate],
        relevance_weight: float,
        temporal_sigma_seconds: float,
    ) -> float:
        if not selected:
            return item.normalized_score

        nearest_selected_similarity = max(
            temporal_similarity(
                item.timestamp_seconds,
                selected_item.timestamp_seconds,
                temporal_sigma_seconds,
            )
            for selected_item in selected
        )
        return (relevance_weight * item.normalized_score) - (
            (1 - relevance_weight) * nearest_selected_similarity
        )

    def _selection_reason(
        self,
        item: ScoredCandidate,
        selected: list[Selection],
    ) -> str:
        if not selected:
            return (
                f"Selected first because it had the strongest normalized "
                f"{item.modality} relevance signal for aspect '{item.aspect}'."
            )

        previous = [selection.candidate for selection in selected]
        nearest_delta = min(
            abs(item.timestamp_seconds - candidate.timestamp_seconds)
            for candidate in previous
        )
        return (
            f"Selected for query relevance while preserving temporal diversity; "
            f"nearest selected evidence is {nearest_delta:.2f}s away; "
            f"matched aspect '{item.aspect}'."
        )
