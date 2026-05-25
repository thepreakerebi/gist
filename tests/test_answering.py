from gist.core.answering import answer_from_evidence, verify_answer_claims
from gist.core.presets import CompressionPreset
from gist.core.schemas import (
    CompressionMetrics,
    CompressionResponse,
    Modality,
    SelectedCandidate,
)
from gist.core.token_estimation import TokenEstimatorProfile


def test_answer_from_evidence_extracts_why_answer_signal() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="Why is the man afraid of the robot hand?",
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="weak",
                modality=Modality.AUDIO,
                timestamp_seconds=1,
                text="The robot hand is visible.",
                selection_rank=1,
                relevance_score=0.3,
                normalized_score=1.0,
                mmr_score=0.5,
                source_score_type="lexical_overlap",
                reason="test",
            ),
            SelectedCandidate(
                id="answer",
                modality=Modality.AUDIO,
                timestamp_seconds=2,
                text="I'm freaked out. I'm having nightmares that I'm being chased.",
                selection_rank=2,
                relevance_score=0.4,
                normalized_score=1.0,
                mmr_score=0.5,
                source_score_type="lexical_overlap",
                reason="test",
            ),
        ],
        metrics=CompressionMetrics(
            input_candidates=2,
            selected_candidates=2,
            visual_selected=0,
            audio_selected=2,
            estimated_candidate_reduction_ratio=1,
            estimated_candidate_reduction_percent=0,
            dropped_candidates=0,
            budget_preset_used=CompressionPreset.BALANCED,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    assert (
        answer_from_evidence(compression)
        == "The evidence suggests: I'm having nightmares that I'm being chased."
    )


def test_answer_from_evidence_prefers_post_anchor_evidence_for_after_queries() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="what text appears after START",
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="start",
                modality=Modality.VISUAL,
                timestamp_seconds=1,
                text="on-screen text near 1.00 seconds: START",
                selection_rank=1,
                relevance_score=1,
                normalized_score=1.0,
                mmr_score=0.5,
                source_score_type="lexical_overlap",
                reason="test",
            ),
            SelectedCandidate(
                id="blue",
                modality=Modality.VISUAL,
                timestamp_seconds=5,
                text="on-screen text near 5.00 seconds: BLUE BOX",
                selection_rank=2,
                relevance_score=0.1,
                normalized_score=0.1,
                mmr_score=0.1,
                source_score_type="lexical_overlap",
                reason="test",
            ),
        ],
        metrics=CompressionMetrics(
            input_candidates=2,
            selected_candidates=2,
            visual_selected=2,
            audio_selected=0,
            estimated_candidate_reduction_ratio=1,
            estimated_candidate_reduction_percent=0,
            dropped_candidates=0,
            budget_preset_used=CompressionPreset.BALANCED,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    assert answer_from_evidence(compression) == "on-screen text near 5.00 seconds: BLUE BOX"


def test_answer_from_evidence_describes_visual_object_instead_of_noisy_ocr() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="show the robot hand on screen",
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="visual",
                modality=Modality.VISUAL,
                timestamp_seconds=34.41,
                text="on-screen text near 34.41 seconds: ‘hat-4 Veta bi . A |",
                clip_start_seconds=30.4,
                clip_end_seconds=38.4,
                selection_rank=1,
                relevance_score=0.9,
                normalized_score=1.0,
                mmr_score=0.5,
                source_score_type="lexical_overlap",
                reason="test",
            )
        ],
        metrics=CompressionMetrics(
            input_candidates=1,
            selected_candidates=1,
            visual_selected=1,
            audio_selected=0,
            estimated_candidate_reduction_ratio=1,
            estimated_candidate_reduction_percent=0,
            dropped_candidates=0,
            budget_preset_used=CompressionPreset.BALANCED,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    assert answer_from_evidence(compression) == (
        "Visual evidence shows robot hand from 30.40s to 38.40s."
    )


def test_verify_answer_claims_drops_unsupported_sentences() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="How do builders use AI?",
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="research",
                modality=Modality.AUDIO,
                timestamp_seconds=1,
                text="AI helps builders research articles and annotate books.",
                selection_rank=1,
                relevance_score=1,
                normalized_score=1,
                mmr_score=1,
                source_score_type="test",
                reason="test",
            )
        ],
        metrics=CompressionMetrics(
            input_candidates=1,
            selected_candidates=1,
            visual_selected=0,
            audio_selected=1,
            estimated_candidate_reduction_ratio=1,
            estimated_candidate_reduction_percent=0,
            dropped_candidates=0,
            budget_preset_used=CompressionPreset.BALANCED,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )
    answer = (
        "Builders use AI to research articles and annotate books. "
        "They also drive trucks across Mars."
    )

    verified = verify_answer_claims(answer, compression)

    assert verified == "Builders use AI to research articles and annotate books."


def test_verify_answer_claims_keeps_single_sentence_answers() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="What appears?",
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="visual",
                modality=Modality.VISUAL,
                timestamp_seconds=1,
                text="on-screen text near 1.00 seconds: BLUE BOX",
                selection_rank=1,
                relevance_score=1,
                normalized_score=1,
                mmr_score=1,
                source_score_type="test",
                reason="test",
            )
        ],
        metrics=CompressionMetrics(
            input_candidates=1,
            selected_candidates=1,
            visual_selected=1,
            audio_selected=0,
            estimated_candidate_reduction_ratio=1,
            estimated_candidate_reduction_percent=0,
            dropped_candidates=0,
            budget_preset_used=CompressionPreset.BALANCED,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    assert verify_answer_claims("A blue box appears.", compression) == "A blue box appears."


def test_verify_answer_claims_preserves_evidence_section() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="How do builders use AI?",
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="research",
                modality=Modality.AUDIO,
                timestamp_seconds=1,
                text="AI helps builders research articles and annotate books.",
                selection_rank=1,
                relevance_score=1,
                normalized_score=1,
                mmr_score=1,
                source_score_type="test",
                reason="test",
            ),
            SelectedCandidate(
                id="code",
                modality=Modality.AUDIO,
                timestamp_seconds=2,
                text="AI also helps builders generate code and review pull requests.",
                selection_rank=2,
                relevance_score=1,
                normalized_score=1,
                mmr_score=1,
                source_score_type="test",
                reason="test",
            ),
        ],
        metrics=CompressionMetrics(
            input_candidates=2,
            selected_candidates=2,
            visual_selected=0,
            audio_selected=2,
            estimated_candidate_reduction_ratio=1,
            estimated_candidate_reduction_percent=0,
            dropped_candidates=0,
            budget_preset_used=CompressionPreset.BALANCED,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )
    answer = (
        "Builders use AI to research articles and annotate books. "
        "They also drive trucks across Mars.\n\n"
        "Evidence:\n"
        "1. Research and annotation support.\n"
        "2. Code generation support."
    )

    verified = verify_answer_claims(answer, compression)

    assert verified == (
        "Builders use AI to research articles and annotate books.\n\n"
        "Evidence:\n"
        "1. Research and annotation support.\n"
        "2. Code generation support."
    )
