from dataclasses import dataclass

from gist.core.presets import PRESETS
from gist.core.schemas import (
    Candidate,
    CompressionMetrics,
    CompressionRequest,
    CompressionResponse,
    Modality,
    SelectedCandidate,
)
from gist.core.scoring import lexical_relevance, temporal_similarity, z_scores


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    id: str
    modality: Modality
    timestamp_seconds: float
    text: str
    relevance_score: float
    normalized_score: float
    source_score_type: str


@dataclass(frozen=True, slots=True)
class Selection:
    candidate: ScoredCandidate
    selection_rank: int
    mmr_score: float
    reason: str


class GistCompressor:
    def compress(self, request: CompressionRequest) -> CompressionResponse:
        config = PRESETS[request.preset]
        scored = self._score_candidates(request)
        selections = self._select_with_mmr(
            candidates=scored,
            max_items=config.max_items,
            relevance_weight=config.relevance_weight,
            temporal_sigma_seconds=config.temporal_sigma_seconds,
        )

        input_count = len(request.visual_candidates) + len(request.audio_candidates)
        selected_count = len(selections)
        reduction_ratio = 1.0 if input_count == 0 else selected_count / input_count
        reduction_percent = (1.0 - reduction_ratio) * 100
        dropped_count = max(input_count - selected_count, 0)

        return CompressionResponse(
            video_id=request.video_id,
            query=request.query,
            preset=request.preset,
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
            ),
        )

    def _score_candidates(self, request: CompressionRequest) -> list[ScoredCandidate]:
        visual = self._score_modality(request.query, request.visual_candidates, Modality.VISUAL)
        audio = self._score_modality(request.query, request.audio_candidates, Modality.AUDIO)
        return visual + audio

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
            )
            for candidate, raw_score, normalized_score in zip(
                candidates,
                raw_scores,
                normalized_scores,
                strict=True,
            )
        ]

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
                f"{item.modality} relevance signal."
            )

        previous = [selection.candidate for selection in selected]
        nearest_delta = min(
            abs(item.timestamp_seconds - candidate.timestamp_seconds)
            for candidate in previous
        )
        return (
            f"Selected for query relevance while preserving temporal diversity; "
            f"nearest selected evidence is {nearest_delta:.2f}s away."
        )
