from gist.core.evidence_pruning import (
    prune_evidence_to_answer,
    prune_evidence_to_answer_citations,
)
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


def test_prune_evidence_to_answer_drops_loose_answer_overlap_by_default() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="How do top builders use AI to do the work of hundreds of engineers?",
        answer=(
            "Top builders use AI for research, writing, code generation, and "
            "analysis to produce work that previously required many people."
        ),
        preset=CompressionPreset.BALANCED,
        selected=[
            _item(
                "research",
                10,
                "AI does research, reads books, annotates sources, and writes articles.",
            ),
            _item(
                "code",
                20,
                "Token maxing applies to writing code and other knowledge work.",
            ),
            _item(
                "analysis",
                30,
                "AI analysis lets builders make decisions quickly and efficiently.",
            ),
            _item(
                "personal-ai",
                40,
                "Next year everyone may have their own personal AI with integrations.",
            ),
            _item(
                "machine-time",
                50,
                "Machine consciousness can create a time billionaire feeling.",
            ),
        ],
        metrics=CompressionMetrics(
            input_candidates=20,
            selected_candidates=5,
            visual_selected=0,
            audio_selected=5,
            estimated_candidate_reduction_ratio=0.25,
            estimated_candidate_reduction_percent=75,
            dropped_candidates=15,
            budget_preset_used=CompressionPreset.BALANCED,
            estimated_baseline_tokens=640,
            estimated_compressed_tokens=160,
            estimated_saved_tokens=480,
            estimated_token_reduction_ratio=0.25,
            estimated_token_reduction_percent=75,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    pruned = prune_evidence_to_answer(compression)

    assert [item.id for item in pruned.selected] == ["research", "code", "analysis"]


def test_prune_evidence_to_answer_citations_keeps_only_cited_evidence() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="How do builders use AI?",
        answer=(
            "Builders use AI for research and code.\n\n"
            "Evidence:\n"
            "1. Research automation is described.\n"
            "3. Code generation is described.\n"
        ),
        preset=CompressionPreset.BALANCED,
        selected=[
            _item("research", 10, "AI automates research."),
            _item("weak", 20, "Everyone may have a personal AI."),
            _item("code", 30, "AI writes code."),
            _item("noise", 40, "Closing remarks."),
        ],
        metrics=CompressionMetrics(
            input_candidates=20,
            selected_candidates=4,
            visual_selected=0,
            audio_selected=4,
            estimated_candidate_reduction_ratio=0.2,
            estimated_candidate_reduction_percent=80,
            dropped_candidates=16,
            budget_preset_used=CompressionPreset.BALANCED,
            estimated_baseline_tokens=640,
            estimated_compressed_tokens=128,
            estimated_saved_tokens=512,
            estimated_token_reduction_ratio=0.2,
            estimated_token_reduction_percent=80,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    pruned = prune_evidence_to_answer_citations(compression, min_items=2)

    assert [item.id for item in pruned.selected] == ["research", "code"]
    assert pruned.metrics.selected_candidates == 2
    assert all("final answer cited" in item.reason for item in pruned.selected)


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
