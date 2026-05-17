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


class GistCompressor:
    def compress(self, request: CompressionRequest) -> CompressionResponse:
        config = PRESETS[request.preset]
        scored = self._score_candidates(request)
        selected = self._select_with_mmr(
            candidates=scored,
            max_items=config.max_items,
            relevance_weight=config.relevance_weight,
            temporal_sigma_seconds=config.temporal_sigma_seconds,
        )

        input_count = len(request.visual_candidates) + len(request.audio_candidates)
        selected_count = len(selected)
        reduction_ratio = 1.0 if input_count == 0 else selected_count / input_count

        return CompressionResponse(
            video_id=request.video_id,
            query=request.query,
            preset=request.preset,
            selected=[
                SelectedCandidate(
                    id=item.id,
                    modality=item.modality,
                    timestamp_seconds=item.timestamp_seconds,
                    text=item.text,
                    relevance_score=item.relevance_score,
                    normalized_score=item.normalized_score,
                )
                for item in selected
            ],
            metrics=CompressionMetrics(
                input_candidates=input_count,
                selected_candidates=selected_count,
                visual_selected=sum(item.modality == Modality.VISUAL for item in selected),
                audio_selected=sum(item.modality == Modality.AUDIO for item in selected),
                estimated_candidate_reduction_ratio=reduction_ratio,
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
    ) -> list[ScoredCandidate]:
        selected: list[ScoredCandidate] = []
        remaining = sorted(candidates, key=lambda item: (item.timestamp_seconds, item.id))

        while remaining and len(selected) < max_items:
            best = max(
                remaining,
                key=lambda item: self._mmr_score(
                    item=item,
                    selected=selected,
                    relevance_weight=relevance_weight,
                    temporal_sigma_seconds=temporal_sigma_seconds,
                ),
            )
            selected.append(best)
            remaining.remove(best)

        return sorted(selected, key=lambda item: item.timestamp_seconds)

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

