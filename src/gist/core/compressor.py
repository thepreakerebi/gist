from dataclasses import dataclass

from gist.core.decomposition import (
    QueryAspect,
    QueryAspectModality,
    RuleBasedQueryDecomposer,
)
from gist.core.presets import PRESETS, CompressionPreset
from gist.core.query_intent import QueryIntent, route_query_intent
from gist.core.schemas import (
    Candidate,
    CompressionMetrics,
    CompressionRequest,
    CompressionResponse,
    Modality,
    SelectedCandidate,
)
from gist.core.scoring import (
    lexical_relevance,
    temporal_similarity,
    text_similarity,
    unique_token_count,
    z_scores,
)
from gist.core.token_estimation import estimate_tokens


AUDIO_VISUAL_ANCHOR_MIN_RELEVANCE = 0.15
AUDIO_VISUAL_ANCHOR_RELATIVE_RELEVANCE = 0.6
AUDIO_VISUAL_ANCHOR_SIGMA_SECONDS = 8.0
AUDIO_VISUAL_ANCHOR_MIN_SCORE = 0.25
AUDIO_VISUAL_ANCHOR_NORMALIZED_BOOST = 1.25
AUDIO_VISUAL_ANCHOR_RELEVANCE_BOOST = 0.1
CROSS_MODAL_TEMPORAL_REDUNDANCY_WEIGHT = 0.15


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    id: str
    modality: Modality
    timestamp_seconds: float
    text: str
    asset_path: str | None
    segment_id: str | None
    scene_start_seconds: float | None
    scene_end_seconds: float | None
    spatial_mask_path: str | None
    relevance_score: float
    normalized_score: float
    source_score_type: str
    aspect: str
    audio_anchor_timestamp_seconds: float | None
    audio_anchor_score: float


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
        if request.query_intent is None:
            query_intent, routing_reason = route_query_intent(request.query)
            request = request.model_copy(
                update={
                    "query_intent": query_intent,
                    "routing_reason": routing_reason,
                }
            )
        elif request.routing_reason is None:
            request = request.model_copy(
                update={"routing_reason": "query intent provided by caller"}
            )

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
            return (
                request.preset,
                self._select_with_preset(
                    request.preset,
                    scored,
                    request.query_intent if request.task_aware_selection else None,
                ),
                None,
            )

        aggressive = self._select_with_preset(
            CompressionPreset.AGGRESSIVE,
            scored,
            request.query_intent if request.task_aware_selection else None,
        )
        should_expand, reason = self._should_expand_budget(aggressive)
        if not should_expand:
            return CompressionPreset.AGGRESSIVE, aggressive, None

        expanded_preset = (
            CompressionPreset.CONSERVATIVE
            if request.preset == CompressionPreset.CONSERVATIVE
            else CompressionPreset.BALANCED
        )
        return (
            expanded_preset,
            self._select_with_preset(
                expanded_preset,
                scored,
                request.query_intent if request.task_aware_selection else None,
            ),
            reason,
        )

    def _select_with_preset(
        self,
        preset: CompressionPreset,
        scored: list[ScoredCandidate],
        query_intent: QueryIntent | None,
    ) -> list[Selection]:
        config = PRESETS[preset]
        scored = self._apply_scene_aware_visual_budget(scored, config.max_items)
        selections = self._select_with_mmr(
            candidates=scored,
            max_items=config.max_items,
            relevance_weight=config.relevance_weight,
            temporal_sigma_seconds=config.temporal_sigma_seconds,
        )
        if query_intent == QueryIntent.COUNTING_COMPARISON:
            selections = self._ensure_counting_visual_neighbors(
                selections=selections,
                candidates=scored,
                max_items=config.max_items,
            )
        if query_intent == QueryIntent.NEGATIVE_EVIDENCE:
            selections = self._ensure_negative_audio_coverage(
                selections=selections,
                candidates=scored,
                max_items=config.max_items,
            )
        return selections

    def _apply_scene_aware_visual_budget(
        self,
        candidates: list[ScoredCandidate],
        max_items: int,
    ) -> list[ScoredCandidate]:
        scene_visuals = [
            candidate
            for candidate in candidates
            if candidate.modality == Modality.VISUAL and candidate.segment_id is not None
        ]
        scene_ids = {candidate.segment_id for candidate in scene_visuals}
        if len(scene_ids) < 2:
            return candidates

        audio_candidates = [
            candidate for candidate in candidates if candidate.modality == Modality.AUDIO
        ]
        unscened_visuals = [
            candidate
            for candidate in candidates
            if candidate.modality == Modality.VISUAL and candidate.segment_id is None
        ]
        visual_budget = min(
            len(scene_visuals),
            max(1, max_items - min(2, len(audio_candidates))),
        )

        grouped: dict[str, list[ScoredCandidate]] = {}
        for candidate in scene_visuals:
            assert candidate.segment_id is not None
            grouped.setdefault(candidate.segment_id, []).append(candidate)
        for group in grouped.values():
            group.sort(
                key=lambda candidate: (
                    candidate.normalized_score,
                    candidate.relevance_score,
                    -candidate.timestamp_seconds,
                ),
                reverse=True,
            )

        ranked_scene_ids = sorted(
            grouped,
            key=lambda scene_id: (
                max(candidate.normalized_score for candidate in grouped[scene_id]),
                max(candidate.relevance_score for candidate in grouped[scene_id]),
            ),
            reverse=True,
        )
        selected: list[ScoredCandidate] = []
        for scene_id in ranked_scene_ids[:visual_budget]:
            selected.append(grouped[scene_id][0])

        remaining_budget = visual_budget - len(selected)
        if remaining_budget > 0:
            selected_ids = {candidate.id for candidate in selected}
            remaining = [
                candidate
                for group in grouped.values()
                for candidate in group[1:]
                if candidate.id not in selected_ids
            ]
            remaining.sort(
                key=lambda candidate: (
                    candidate.normalized_score,
                    candidate.relevance_score,
                    -candidate.timestamp_seconds,
                ),
                reverse=True,
            )
            selected.extend(remaining[:remaining_budget])

        return audio_candidates + unscened_visuals + selected

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
                    asset_path=selection.candidate.asset_path,
                    segment_id=selection.candidate.segment_id,
                    scene_start_seconds=selection.candidate.scene_start_seconds,
                    scene_end_seconds=selection.candidate.scene_end_seconds,
                    spatial_mask_path=selection.candidate.spatial_mask_path,
                    audio_anchor_timestamp_seconds=(
                        selection.candidate.audio_anchor_timestamp_seconds
                    ),
                    audio_anchor_score=selection.candidate.audio_anchor_score,
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
            query_intent=request.query_intent,
            routing_reason=request.routing_reason,
        )

    def _should_expand_budget(self, selections: list[Selection]) -> tuple[bool, str | None]:
        if not selections:
            return True, "no evidence selected at aggressive budget"

        best_relevance = max(selection.candidate.relevance_score for selection in selections)
        if best_relevance < 0.15:
            return True, "low best relevance at aggressive budget"

        audio_selected = sum(
            selection.candidate.modality == Modality.AUDIO for selection in selections
        )
        visual_selected = sum(
            selection.candidate.modality == Modality.VISUAL for selection in selections
        )
        has_audio_anchored_visuals = any(
            selection.candidate.modality == Modality.VISUAL
            and selection.candidate.audio_anchor_score >= AUDIO_VISUAL_ANCHOR_MIN_SCORE
            for selection in selections
        )
        if has_audio_anchored_visuals and audio_selected < 2 and visual_selected >= 2:
            return True, "aggressive budget underrepresented source audio evidence"

        modalities = {selection.candidate.modality for selection in selections}
        if len(selections) > 1 and len(modalities) == 1:
            if all(_is_grounded_transcript_moment(selection.candidate) for selection in selections):
                return False, None
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
        collapsed = self._collapse_duplicate_scores(visual + audio)
        return self._apply_audio_visual_anchors(collapsed)

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
                asset_path=str(candidate.asset_path) if candidate.asset_path else None,
                segment_id=candidate.segment_id,
                scene_start_seconds=candidate.scene_start_seconds,
                scene_end_seconds=candidate.scene_end_seconds,
                spatial_mask_path=(
                    str(candidate.spatial_mask_path) if candidate.spatial_mask_path else None
                ),
                relevance_score=raw_score,
                normalized_score=normalized_score,
                source_score_type="model_saliency"
                if candidate.saliency_score is not None
                else "lexical_overlap",
                aspect=query,
                audio_anchor_timestamp_seconds=None,
                audio_anchor_score=0.0,
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

    def _apply_audio_visual_anchors(
        self,
        candidates: list[ScoredCandidate],
    ) -> list[ScoredCandidate]:
        audio_candidates = [
            candidate for candidate in candidates if candidate.modality == Modality.AUDIO
        ]
        visual_candidates = [
            candidate for candidate in candidates if candidate.modality == Modality.VISUAL
        ]
        if not audio_candidates or not visual_candidates:
            return candidates

        best_audio_relevance = max(candidate.relevance_score for candidate in audio_candidates)
        if best_audio_relevance < AUDIO_VISUAL_ANCHOR_MIN_RELEVANCE:
            return candidates

        relevance_floor = max(
            AUDIO_VISUAL_ANCHOR_MIN_RELEVANCE,
            best_audio_relevance * AUDIO_VISUAL_ANCHOR_RELATIVE_RELEVANCE,
        )
        anchors = [
            candidate
            for candidate in audio_candidates
            if candidate.relevance_score >= relevance_floor
        ]
        if not anchors:
            return candidates

        anchored: list[ScoredCandidate] = []
        for candidate in candidates:
            if candidate.modality != Modality.VISUAL:
                anchored.append(candidate)
                continue

            anchor, anchor_score = self._nearest_audio_anchor(candidate, anchors)
            if anchor is None or anchor_score < AUDIO_VISUAL_ANCHOR_MIN_SCORE:
                anchored.append(candidate)
                continue

            anchored.append(
                ScoredCandidate(
                    id=candidate.id,
                    modality=candidate.modality,
                    timestamp_seconds=candidate.timestamp_seconds,
                    text=candidate.text,
                    asset_path=candidate.asset_path,
                    segment_id=candidate.segment_id,
                    scene_start_seconds=candidate.scene_start_seconds,
                    scene_end_seconds=candidate.scene_end_seconds,
                    spatial_mask_path=candidate.spatial_mask_path,
                    relevance_score=(
                        candidate.relevance_score
                        + (AUDIO_VISUAL_ANCHOR_RELEVANCE_BOOST * anchor_score)
                    ),
                    normalized_score=(
                        candidate.normalized_score
                        + (AUDIO_VISUAL_ANCHOR_NORMALIZED_BOOST * anchor_score)
                    ),
                    source_score_type=candidate.source_score_type,
                    aspect=candidate.aspect,
                    audio_anchor_timestamp_seconds=anchor.timestamp_seconds,
                    audio_anchor_score=anchor_score,
                )
            )
        return anchored

    def _nearest_audio_anchor(
        self,
        visual_candidate: ScoredCandidate,
        anchors: list[ScoredCandidate],
    ) -> tuple[ScoredCandidate | None, float]:
        best_anchor: ScoredCandidate | None = None
        best_score = 0.0
        for anchor in anchors:
            score = temporal_similarity(
                visual_candidate.timestamp_seconds,
                anchor.timestamp_seconds,
                AUDIO_VISUAL_ANCHOR_SIGMA_SECONDS,
            )
            if score > best_score:
                best_anchor = anchor
                best_score = score
        return best_anchor, best_score

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
            remaining = self._drop_redundant_neighbors(
                selected=best,
                remaining=remaining,
                temporal_sigma_seconds=temporal_sigma_seconds,
            )

        return self._ensure_audio_anchor_sources(
            selections=selected,
            candidates=candidates,
            max_items=max_items,
        )

    def _ensure_audio_anchor_sources(
        self,
        selections: list[Selection],
        candidates: list[ScoredCandidate],
        max_items: int,
    ) -> list[Selection]:
        anchored_visuals = [
            selection
            for selection in selections
            if selection.candidate.modality == Modality.VISUAL
            and selection.candidate.audio_anchor_timestamp_seconds is not None
            and selection.candidate.audio_anchor_score >= AUDIO_VISUAL_ANCHOR_MIN_SCORE
        ]
        if not anchored_visuals:
            return selections

        target_audio_count = min(4, max(2, max_items // 3))
        selected_audio_count = sum(
            selection.candidate.modality == Modality.AUDIO for selection in selections
        )
        if selected_audio_count >= target_audio_count:
            return selections

        selected_ids = {selection.candidate.id for selection in selections}
        anchor_timestamps = {
            selection.candidate.audio_anchor_timestamp_seconds
            for selection in anchored_visuals
            if selection.candidate.audio_anchor_timestamp_seconds is not None
        }
        source_audio = [
            candidate
            for candidate in candidates
            if candidate.modality == Modality.AUDIO
            and candidate.id not in selected_ids
            and any(
                abs(candidate.timestamp_seconds - timestamp) < 1e-6
                for timestamp in anchor_timestamps
            )
        ]
        source_audio.sort(
            key=lambda candidate: (candidate.relevance_score, candidate.normalized_score),
            reverse=True,
        )

        balanced = list(selections)
        for candidate in source_audio:
            if selected_audio_count >= target_audio_count:
                break
            if len(balanced) >= max_items and not self._drop_weakest_anchored_visual(balanced):
                break
            balanced.append(
                Selection(
                    candidate=candidate,
                    selection_rank=0,
                    mmr_score=candidate.normalized_score,
                    reason=(
                        "Included because selected visual evidence was anchored "
                        "to this audio evidence."
                    ),
                )
            )
            selected_audio_count += 1

        return self._rerank_selections(balanced)

    def _ensure_counting_visual_neighbors(
        self,
        selections: list[Selection],
        candidates: list[ScoredCandidate],
        max_items: int,
    ) -> list[Selection]:
        selected_ids = {selection.candidate.id for selection in selections}
        selected_visuals = [
            selection.candidate
            for selection in selections
            if selection.candidate.modality == Modality.VISUAL
        ]
        if not selected_visuals:
            return selections

        visual_candidates = [
            candidate for candidate in candidates if candidate.modality == Modality.VISUAL
        ]
        if len(visual_candidates) <= len(selected_visuals):
            return selections

        neighbors: list[ScoredCandidate] = []
        for selected in sorted(
            selected_visuals,
            key=lambda candidate: (candidate.normalized_score, candidate.relevance_score),
            reverse=True,
        ):
            nearby = [
                candidate
                for candidate in visual_candidates
                if candidate.id not in selected_ids
                and abs(candidate.timestamp_seconds - selected.timestamp_seconds) <= 12.0
            ]
            nearby.sort(
                key=lambda candidate: (
                    abs(candidate.timestamp_seconds - selected.timestamp_seconds),
                    -candidate.normalized_score,
                )
            )
            for candidate in nearby:
                if candidate.id in selected_ids:
                    continue
                neighbors.append(candidate)
                selected_ids.add(candidate.id)
                break

        if not neighbors:
            return selections

        balanced = list(selections)
        for candidate in neighbors:
            if len(balanced) >= max_items and not self._drop_weakest_audio_or_visual(balanced):
                break
            balanced.append(
                Selection(
                    candidate=candidate,
                    selection_rank=0,
                    mmr_score=candidate.normalized_score,
                    reason=(
                        "Included as neighboring visual evidence because counting/comparison "
                        "queries need denser frames around relevant moments."
                    ),
                )
            )
        return self._rerank_selections(balanced)

    def _ensure_negative_audio_coverage(
        self,
        selections: list[Selection],
        candidates: list[ScoredCandidate],
        max_items: int,
    ) -> list[Selection]:
        selected_ids = {selection.candidate.id for selection in selections}
        selected_audio = sum(
            selection.candidate.modality == Modality.AUDIO for selection in selections
        )
        target_audio = min(max_items, max(3, max_items // 2))
        if selected_audio >= target_audio:
            return selections

        audio_candidates = [
            candidate
            for candidate in candidates
            if candidate.modality == Modality.AUDIO and candidate.id not in selected_ids
        ]
        audio_candidates.sort(
            key=lambda candidate: (
                candidate.normalized_score,
                candidate.relevance_score,
                -candidate.timestamp_seconds,
            ),
            reverse=True,
        )
        if not audio_candidates:
            return selections

        balanced = list(selections)
        for candidate in audio_candidates:
            if selected_audio >= target_audio:
                break
            if len(balanced) >= max_items and not self._drop_weakest_visual(balanced):
                break
            balanced.append(
                Selection(
                    candidate=candidate,
                    selection_rank=0,
                    mmr_score=candidate.normalized_score,
                    reason=(
                        "Included for negative-evidence coverage so the model can compare "
                        "which alternatives are discussed versus absent."
                    ),
                )
            )
            selected_audio += 1
        return self._rerank_selections(balanced)

    def _drop_weakest_anchored_visual(self, selections: list[Selection]) -> bool:
        removable = [
            selection
            for selection in selections
            if selection.candidate.modality == Modality.VISUAL
            and selection.candidate.audio_anchor_score >= AUDIO_VISUAL_ANCHOR_MIN_SCORE
        ]
        if not removable:
            return False

        weakest = min(removable, key=lambda selection: selection.mmr_score)
        selections.remove(weakest)
        return True

    def _drop_weakest_visual(self, selections: list[Selection]) -> bool:
        removable = [
            selection
            for selection in selections
            if selection.candidate.modality == Modality.VISUAL
        ]
        if not removable:
            return False

        weakest = min(removable, key=lambda selection: selection.mmr_score)
        selections.remove(weakest)
        return True

    def _drop_weakest_audio_or_visual(self, selections: list[Selection]) -> bool:
        removable_audio = [
            selection
            for selection in selections
            if selection.candidate.modality == Modality.AUDIO
        ]
        if removable_audio:
            weakest = min(removable_audio, key=lambda selection: selection.mmr_score)
            selections.remove(weakest)
            return True
        return self._drop_weakest_visual(selections)

    def _rerank_selections(self, selections: list[Selection]) -> list[Selection]:
        return [
            Selection(
                candidate=selection.candidate,
                selection_rank=index + 1,
                mmr_score=selection.mmr_score,
                reason=selection.reason,
            )
            for index, selection in enumerate(selections)
        ]

    def _drop_redundant_neighbors(
        self,
        selected: ScoredCandidate,
        remaining: list[ScoredCandidate],
        temporal_sigma_seconds: float,
    ) -> list[ScoredCandidate]:
        min_gap_seconds = temporal_sigma_seconds * 0.75
        return [
            candidate
            for candidate in remaining
            if not self._is_redundant_neighbor(
                selected=selected,
                candidate=candidate,
                min_gap_seconds=min_gap_seconds,
            )
        ]

    def _is_redundant_neighbor(
        self,
        selected: ScoredCandidate,
        candidate: ScoredCandidate,
        min_gap_seconds: float,
    ) -> bool:
        if selected.modality != Modality.AUDIO or candidate.modality != Modality.AUDIO:
            return False

        if abs(selected.timestamp_seconds - candidate.timestamp_seconds) > min_gap_seconds:
            return False

        if min(unique_token_count(selected.text), unique_token_count(candidate.text)) < 5:
            return False

        return text_similarity(selected.text, candidate.text) >= 0.35

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
            self._temporal_redundancy_similarity(
                item=item,
                selected_item=selected_item,
                temporal_sigma_seconds=temporal_sigma_seconds,
            )
            for selected_item in selected
        )
        return (relevance_weight * item.normalized_score) - (
            (1 - relevance_weight) * nearest_selected_similarity
        )

    def _temporal_redundancy_similarity(
        self,
        item: ScoredCandidate,
        selected_item: ScoredCandidate,
        temporal_sigma_seconds: float,
    ) -> float:
        similarity = temporal_similarity(
            item.timestamp_seconds,
            selected_item.timestamp_seconds,
            temporal_sigma_seconds,
        )
        if item.modality == selected_item.modality:
            return similarity

        if self._is_audio_visual_anchor_pair(item, selected_item):
            return 0.0

        return similarity * CROSS_MODAL_TEMPORAL_REDUNDANCY_WEIGHT

    def _is_audio_visual_anchor_pair(
        self,
        item: ScoredCandidate,
        selected_item: ScoredCandidate,
    ) -> bool:
        if item.modality == Modality.VISUAL and selected_item.modality == Modality.AUDIO:
            return item.audio_anchor_timestamp_seconds == selected_item.timestamp_seconds

        if item.modality == Modality.AUDIO and selected_item.modality == Modality.VISUAL:
            return selected_item.audio_anchor_timestamp_seconds == item.timestamp_seconds

        return False

    def _selection_reason(
        self,
        item: ScoredCandidate,
        selected: list[Selection],
    ) -> str:
        if not selected:
            reason = (
                f"Selected first because it had the strongest normalized "
                f"{item.modality} relevance signal for aspect '{item.aspect}'."
            )
            return self._with_anchor_reason(item, reason)

        previous = [selection.candidate for selection in selected]
        nearest_delta = min(
            abs(item.timestamp_seconds - candidate.timestamp_seconds)
            for candidate in previous
        )
        reason = (
            f"Selected for query relevance while preserving temporal diversity; "
            f"nearest selected evidence is {nearest_delta:.2f}s away; "
            f"matched aspect '{item.aspect}'."
        )
        return self._with_anchor_reason(item, reason)

    def _with_anchor_reason(self, item: ScoredCandidate, reason: str) -> str:
        if (
            item.modality != Modality.VISUAL
            or item.audio_anchor_timestamp_seconds is None
            or item.audio_anchor_score < AUDIO_VISUAL_ANCHOR_MIN_SCORE
        ):
            return reason

        return (
            f"{reason} Boosted because it is near relevant audio evidence at "
            f"{item.audio_anchor_timestamp_seconds:.2f}s "
            f"(anchor score {item.audio_anchor_score:.2f})."
        )


def _is_grounded_transcript_moment(candidate: ScoredCandidate) -> bool:
    return (
        candidate.modality == Modality.AUDIO
        and ":audio:" in candidate.id
        and ":visual:" in candidate.id
        and candidate.asset_path is not None
    )
