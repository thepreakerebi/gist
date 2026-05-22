from gist.core.evidence_pruning import prune_evidence_to_answer
from gist.core.presets import CompressionPreset
from gist.core.schemas import (
    CompressionMetrics,
    CompressionResponse,
    Modality,
    SelectedCandidate,
)
from gist.core.token_estimation import TokenEstimatorProfile


def test_prune_evidence_to_answer_drops_weak_final_clips() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="How do builders use AI?",
        answer="Builders use AI for code generation and research automation.",
        preset=CompressionPreset.BALANCED,
        selected=[
            _item("intro", 0, "Welcome back to the show."),
            _item("code", 10, "AI helps with code generation and developer workflows."),
            _item("research", 20, "The system automates research for builders."),
            _item("open", 30, "Open source tools are used as harnesses."),
            _item("noise", 40, "Lunch and travel plans are discussed."),
            _item("closing", 50, "Thanks for watching."),
            _item("machine", 60, "Machine time does work for the founder."),
        ],
        metrics=CompressionMetrics(
            input_candidates=20,
            selected_candidates=7,
            visual_selected=0,
            audio_selected=7,
            estimated_candidate_reduction_ratio=0.35,
            estimated_candidate_reduction_percent=65,
            dropped_candidates=13,
            budget_preset_used=CompressionPreset.BALANCED,
            estimated_baseline_tokens=640,
            estimated_compressed_tokens=224,
            estimated_saved_tokens=416,
            estimated_token_reduction_ratio=0.35,
            estimated_token_reduction_percent=65,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    pruned = prune_evidence_to_answer(compression, max_items=4, min_items=2)

    selected_ids = {item.id for item in pruned.selected}
    assert selected_ids <= {"code", "research", "open", "machine"}
    assert "code" in selected_ids
    assert "research" in selected_ids
    assert pruned.metrics.selected_candidates <= 4
    assert pruned.metrics.audio_selected == pruned.metrics.selected_candidates
    assert pruned.metrics.estimated_compressed_tokens == pruned.metrics.selected_candidates * 32
    assert pruned.metrics.estimated_token_reduction_percent > 65
    assert all("answer-grounded pruning" in item.reason for item in pruned.selected)


def test_prune_evidence_to_answer_keeps_short_selection_unchanged() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="pricing",
        answer="Pricing starts at ten dollars.",
        preset=CompressionPreset.BALANCED,
        selected=[_item("pricing", 10, "Pricing starts at ten dollars.")],
        metrics=CompressionMetrics(
            input_candidates=1,
            selected_candidates=1,
            visual_selected=0,
            audio_selected=1,
            estimated_candidate_reduction_ratio=1,
            estimated_candidate_reduction_percent=0,
            dropped_candidates=0,
            budget_preset_used=CompressionPreset.BALANCED,
            estimated_baseline_tokens=32,
            estimated_compressed_tokens=32,
            estimated_saved_tokens=0,
            estimated_token_reduction_ratio=1,
            estimated_token_reduction_percent=0,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    assert prune_evidence_to_answer(compression) == compression


def _item(id_: str, timestamp_seconds: float, text: str) -> SelectedCandidate:
    return SelectedCandidate(
        id=id_,
        modality=Modality.AUDIO,
        timestamp_seconds=timestamp_seconds,
        text=text,
        selection_rank=1,
        relevance_score=0.5,
        normalized_score=1,
        mmr_score=1,
        source_score_type="lexical_overlap",
        reason="selected",
    )
