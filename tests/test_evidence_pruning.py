from gist.core.evidence_pruning import (
    annotate_evidence_support,
    consolidate_redundant_evidence,
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
    assert all(item.support_label is not None for item in pruned.selected)


def test_prune_evidence_to_answer_citations_replaces_weak_cited_evidence() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="How do builders use AI?",
        answer=(
            "Builders use AI for research and code.\n\n"
            "Evidence:\n"
            "1. Lunch is discussed.\n"
            "3. Code generation is described.\n"
        ),
        preset=CompressionPreset.BALANCED,
        selected=[
            _item("weak-cited", 10, "Lunch and travel plans are discussed."),
            _item("research", 20, "AI automates research for builders."),
            _item("code", 30, "AI writes code for builders."),
        ],
        metrics=CompressionMetrics(
            input_candidates=20,
            selected_candidates=3,
            visual_selected=0,
            audio_selected=3,
            estimated_candidate_reduction_ratio=0.15,
            estimated_candidate_reduction_percent=85,
            dropped_candidates=17,
            budget_preset_used=CompressionPreset.BALANCED,
            estimated_baseline_tokens=640,
            estimated_compressed_tokens=96,
            estimated_saved_tokens=544,
            estimated_token_reduction_ratio=0.15,
            estimated_token_reduction_percent=85,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    pruned = prune_evidence_to_answer_citations(compression, min_items=2)

    assert [item.id for item in pruned.selected] == ["research", "code"]
    assert all(item.support_label in {"medium", "strong"} for item in pruned.selected)


def test_annotate_evidence_support_adds_support_metadata() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="How do builders use AI?",
        answer="Builders use AI for research.",
        preset=CompressionPreset.BALANCED,
        selected=[
            _item("research", 10, "AI automates research for builders."),
            _item("noise", 20, "Lunch and travel plans are discussed."),
        ],
        metrics=CompressionMetrics(
            input_candidates=20,
            selected_candidates=2,
            visual_selected=0,
            audio_selected=2,
            estimated_candidate_reduction_ratio=0.1,
            estimated_candidate_reduction_percent=90,
            dropped_candidates=18,
            budget_preset_used=CompressionPreset.BALANCED,
            estimated_baseline_tokens=640,
            estimated_compressed_tokens=64,
            estimated_saved_tokens=576,
            estimated_token_reduction_ratio=0.1,
            estimated_token_reduction_percent=90,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    annotated = annotate_evidence_support(compression)

    assert annotated.selected[0].evidence_support_score is not None
    assert annotated.selected[0].answer_support_score is not None
    assert annotated.selected[0].query_support_score is not None
    assert annotated.selected[0].audio_support_score is not None
    assert annotated.selected[0].audio_support_score > 0
    assert annotated.selected[0].ocr_support_score == 0
    assert annotated.selected[0].visual_support_score == 0
    assert annotated.selected[0].support_label in {"medium", "strong"}
    assert annotated.selected[1].support_label == "weak"


def test_annotate_evidence_support_scores_visual_ocr_and_cross_modal() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="what text is shown on screen",
        answer="The screen says GIST TOKEN SAVER.",
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="ocr",
                modality=Modality.VISUAL,
                timestamp_seconds=4,
                text="on-screen text near 4.00 seconds: GIST TOKEN SAVER",
                audio_anchor_timestamp_seconds=5,
                audio_anchor_score=0.82,
                selection_rank=1,
                relevance_score=0.4,
                normalized_score=0.6,
                mmr_score=0.5,
                source_score_type="ocr",
                reason="selected",
            )
        ],
        metrics=CompressionMetrics(
            input_candidates=10,
            selected_candidates=1,
            visual_selected=1,
            audio_selected=0,
            estimated_candidate_reduction_ratio=0.1,
            estimated_candidate_reduction_percent=90,
            dropped_candidates=9,
            budget_preset_used=CompressionPreset.BALANCED,
            estimated_baseline_tokens=320,
            estimated_compressed_tokens=128,
            estimated_saved_tokens=192,
            estimated_token_reduction_ratio=0.4,
            estimated_token_reduction_percent=60,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    annotated = annotate_evidence_support(compression)
    item = annotated.selected[0]

    assert item.ocr_support_score is not None and item.ocr_support_score > 0
    assert item.visual_support_score is not None and item.visual_support_score > 0
    assert item.cross_modal_support_score == 0.82
    assert item.audio_support_score == 0
    assert item.support_label == "strong"


def test_prune_evidence_to_answer_citations_parses_inline_citation_lists() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="Why is he afraid?",
        answer="He is afraid because of nightmares (Evidence 1, 2, and 4).",
        preset=CompressionPreset.BALANCED,
        selected=[
            _item("first", 10, "nightmares"),
            _item("second", 20, "freaked out"),
            _item("uncited", 30, "unrelated context"),
            _item("fourth", 40, "chased by a robot"),
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

    pruned = prune_evidence_to_answer_citations(compression)

    assert [item.id for item in pruned.selected] == ["first", "second", "fourth"]


def test_prune_evidence_to_answer_citations_allows_single_cited_evidence() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="What kills startups?",
        answer="The mistake is building something users do not like. Evidence: 3.",
        preset=CompressionPreset.BALANCED,
        selected=[
            _item("context", 10, "customer problem"),
            _item("more-context", 20, "startup framing"),
            _item("answer", 30, "users do not like the product"),
        ],
        metrics=CompressionMetrics(
            input_candidates=20,
            selected_candidates=3,
            visual_selected=0,
            audio_selected=3,
            estimated_candidate_reduction_ratio=0.15,
            estimated_candidate_reduction_percent=85,
            dropped_candidates=17,
            budget_preset_used=CompressionPreset.BALANCED,
            estimated_baseline_tokens=640,
            estimated_compressed_tokens=96,
            estimated_saved_tokens=544,
            estimated_token_reduction_ratio=0.15,
            estimated_token_reduction_percent=85,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    pruned = prune_evidence_to_answer_citations(compression)

    assert [item.id for item in pruned.selected] == ["answer"]


def test_consolidate_redundant_evidence_keeps_strongest_representative() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="What does she say about robotics and space?",
        answer="She says they should follow their passions: robotics and space.",
        preset=CompressionPreset.BALANCED,
        selected=[
            _item(
                "early",
                10,
                "We have to follow our passions. You have robotics and I want space.",
            ),
            _item(
                "repeat",
                30,
                "We have to follow our passions. You have robotics and I want space.",
            ),
            _item("distinct", 90, "The launch countdown begins."),
        ],
        metrics=CompressionMetrics(
            input_candidates=20,
            selected_candidates=3,
            visual_selected=0,
            audio_selected=3,
            estimated_candidate_reduction_ratio=0.15,
            estimated_candidate_reduction_percent=85,
            dropped_candidates=17,
            budget_preset_used=CompressionPreset.BALANCED,
            estimated_baseline_tokens=640,
            estimated_compressed_tokens=96,
            estimated_saved_tokens=544,
            estimated_token_reduction_ratio=0.15,
            estimated_token_reduction_percent=85,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    consolidated = consolidate_redundant_evidence(compression)

    assert [item.id for item in consolidated.selected] == ["early", "distinct"]
    assert consolidated.metrics.selected_candidates == 2
    assert "redundant evidence clips" in consolidated.selected[0].reason


def test_consolidate_redundant_evidence_keeps_distinct_claims() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="How do builders use AI?",
        answer="Builders use AI for research and code.",
        preset=CompressionPreset.BALANCED,
        selected=[
            _item("research", 10, "AI researches articles and annotates books."),
            _item("code", 20, "AI writes code and builds product features."),
        ],
        metrics=CompressionMetrics(
            input_candidates=20,
            selected_candidates=2,
            visual_selected=0,
            audio_selected=2,
            estimated_candidate_reduction_ratio=0.1,
            estimated_candidate_reduction_percent=90,
            dropped_candidates=18,
            budget_preset_used=CompressionPreset.BALANCED,
            estimated_baseline_tokens=640,
            estimated_compressed_tokens=64,
            estimated_saved_tokens=576,
            estimated_token_reduction_ratio=0.1,
            estimated_token_reduction_percent=90,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    assert consolidate_redundant_evidence(compression) == compression


def test_consolidate_redundant_evidence_collapses_overlapping_audio_windows() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="architecture missions",
        answer="The architecture for these missions is taking shape.",
        preset=CompressionPreset.BALANCED,
        selected=[
            _item("precontext", 53, "possible and further power understood. The architecture."),
            _item("answer", 55, "The architecture for these missions is already taking."),
        ],
        metrics=CompressionMetrics(
            input_candidates=20,
            selected_candidates=2,
            visual_selected=0,
            audio_selected=2,
            estimated_candidate_reduction_ratio=0.1,
            estimated_candidate_reduction_percent=90,
            dropped_candidates=18,
            budget_preset_used=CompressionPreset.BALANCED,
            estimated_baseline_tokens=640,
            estimated_compressed_tokens=64,
            estimated_saved_tokens=576,
            estimated_token_reduction_ratio=0.1,
            estimated_token_reduction_percent=90,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )
    compression = compression.model_copy(
        update={
            "selected": [
                compression.selected[0].model_copy(
                    update={"clip_start_seconds": 50.0, "clip_end_seconds": 56.0}
                ),
                compression.selected[1].model_copy(
                    update={"clip_start_seconds": 52.0, "clip_end_seconds": 58.0}
                ),
            ]
        }
    )

    consolidated = consolidate_redundant_evidence(compression)

    assert [item.id for item in consolidated.selected] == ["answer"]


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
