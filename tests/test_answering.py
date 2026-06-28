from gist.core.answering import answer_from_evidence, verify_answer_claims
from gist.core.presets import CompressionPreset
from gist.core.query_intent import QueryIntent
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


def test_answer_from_evidence_summarizes_global_topics_across_timeline() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="What are the main topics covered throughout this lecture?",
        query_intent=QueryIntent.GLOBAL_SUMMARY,
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="admin",
                modality=Modality.AUDIO,
                timestamp_seconds=100,
                text="The course has weekly exercises and a final project.",
                selection_rank=1,
                relevance_score=0.5,
                normalized_score=1,
                mmr_score=0.5,
                source_score_type="lexical_overlap",
                reason="test",
            ),
            SelectedCandidate(
                id="control",
                modality=Modality.AUDIO,
                timestamp_seconds=1700,
                text="The robot uses sensors and a simple control arm.",
                selection_rank=2,
                relevance_score=0.2,
                normalized_score=0.5,
                mmr_score=0.4,
                source_score_type="lexical_overlap",
                reason="test",
            ),
            SelectedCandidate(
                id="loop",
                modality=Modality.VISUAL,
                timestamp_seconds=3000,
                text=(
                    "on-screen text near 3000 seconds: "
                    "Biology-Robotics Loop on Legged Locomotion Studies"
                ),
                selection_rank=3,
                relevance_score=0.3,
                normalized_score=0.7,
                mmr_score=0.4,
                source_score_type="model_saliency",
                reason="test",
            ),
        ],
        metrics=CompressionMetrics(
            input_candidates=3,
            selected_candidates=3,
            visual_selected=1,
            audio_selected=2,
            estimated_candidate_reduction_ratio=1,
            estimated_candidate_reduction_percent=0,
            dropped_candidates=0,
            budget_preset_used=CompressionPreset.BALANCED,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    assert answer_from_evidence(compression) == (
        "The video covers: robotics, sensors, and control; "
        "biology, robotics, and legged locomotion."
    )


def test_answer_from_evidence_prefers_global_agenda_slide_over_noisy_audio() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="What are the main topics covered throughout this lecture?",
        query_intent=QueryIntent.GLOBAL_SUMMARY,
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="noisy-audio",
                modality=Modality.AUDIO,
                timestamp_seconds=2500,
                text=(
                    "I will pick that equal to the length of the gamma that determine "
                    "that the term is low-band wall."
                ),
                selection_rank=1,
                relevance_score=0.4,
                normalized_score=0.6,
                mmr_score=0.3,
                source_score_type="lexical_overlap",
                reason="test",
            ),
            SelectedCandidate(
                id="agenda",
                modality=Modality.VISUAL,
                timestamp_seconds=222,
                text=(
                    "on-screen text near 222.73 seconds: Today: Lecture 2 "
                    "Legged Locomotion and Cont eptual Models * Legged Locomotion "
                    "in Nature * Characterization of Locomotion * Model for Running"
                ),
                selection_rank=2,
                relevance_score=0.8,
                normalized_score=0.9,
                mmr_score=0.7,
                source_score_type="ocr",
                reason="test",
            ),
        ],
        metrics=CompressionMetrics(
            input_candidates=2,
            selected_candidates=2,
            visual_selected=1,
            audio_selected=1,
            estimated_candidate_reduction_ratio=1,
            estimated_candidate_reduction_percent=0,
            dropped_candidates=0,
            budget_preset_used=CompressionPreset.BALANCED,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    assert answer_from_evidence(compression) == (
        "The video covers: Legged Locomotion and Conceptual Models; "
        "Legged Locomotion in Nature; Characterization of Locomotion."
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


def test_answer_from_evidence_uses_model_temporal_target_scores() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="What title appears after the editing interface?",
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="opening",
                modality=Modality.VISUAL,
                timestamp_seconds=8,
                text="on-screen text near 8.00 seconds: KINECT",
                selection_rank=1,
                relevance_score=1,
                normalized_score=1,
                mmr_score=0.8,
                source_score_type="model_saliency",
                reason="test",
                temporal_anchor_score=0.1,
                temporal_target_score=0.98,
                temporal_direction="after",
            ),
            SelectedCandidate(
                id="anchor",
                modality=Modality.VISUAL,
                timestamp_seconds=3529,
                text="video editing interface",
                selection_rank=2,
                relevance_score=0.9,
                normalized_score=0.9,
                mmr_score=0.7,
                source_score_type="model_saliency",
                reason="test",
                temporal_anchor_score=0.99,
                temporal_target_score=0.1,
                temporal_direction="after",
            ),
            SelectedCandidate(
                id="target",
                modality=Modality.VISUAL,
                timestamp_seconds=3537,
                text="on-screen text near 3537.00 seconds: KINECT for Windows",
                selection_rank=3,
                relevance_score=0.7,
                normalized_score=0.7,
                mmr_score=0.6,
                source_score_type="model_saliency",
                reason="test",
                temporal_anchor_score=0.1,
                temporal_target_score=0.9,
                temporal_direction="after",
            ),
        ],
        metrics=CompressionMetrics(
            input_candidates=3,
            selected_candidates=3,
            visual_selected=3,
            audio_selected=0,
            estimated_candidate_reduction_ratio=1,
            estimated_candidate_reduction_percent=0,
            dropped_candidates=0,
            budget_preset_used=CompressionPreset.BALANCED,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    assert (
        answer_from_evidence(compression)
        == "on-screen text near 3537.00 seconds: KINECT for Windows"
    )


def test_answer_from_evidence_treats_slide_query_as_ocr_text() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="What slide appears after the Further Reading Materials slide?",
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="anchor",
                modality=Modality.VISUAL,
                timestamp_seconds=100,
                text="on-screen text near 100 seconds: Further Reading Materials",
                segment_id="scene-1",
                selection_rank=1,
                relevance_score=1,
                normalized_score=1,
                mmr_score=1,
                source_score_type="model_saliency",
                reason="test",
                temporal_anchor_score=0.95,
                temporal_target_score=0.1,
                temporal_direction="after",
            ),
            SelectedCandidate(
                id="target",
                modality=Modality.VISUAL,
                timestamp_seconds=110,
                text="on-screen text near 110 seconds: Next Week",
                segment_id="scene-2",
                selection_rank=2,
                relevance_score=0.8,
                normalized_score=0.8,
                mmr_score=0.8,
                source_score_type="model_saliency",
                reason="test",
                temporal_anchor_score=0.1,
                temporal_target_score=0.7,
                temporal_direction="after",
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

    assert answer_from_evidence(compression) == "on-screen text near 110 seconds: Next Week"


def test_answer_from_evidence_prefers_earliest_ocr_for_opening_title_query() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="What title and product image appear on the opening projection screen?",
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="opening",
                modality=Modality.VISUAL,
                timestamp_seconds=8,
                text="on-screen text near 8.00 seconds: KINECT",
                selection_rank=1,
                relevance_score=0.8,
                normalized_score=0.8,
                mmr_score=0.5,
                source_score_type="ocr",
                reason="test",
            ),
            SelectedCandidate(
                id="later",
                modality=Modality.VISUAL,
                timestamp_seconds=2700,
                text="on-screen text near 2700.00 seconds: product projection",
                selection_rank=2,
                relevance_score=1,
                normalized_score=1,
                mmr_score=0.6,
                source_score_type="ocr",
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

    assert (
        answer_from_evidence(compression)
        == "on-screen text near 8.00 seconds: KINECT"
    )


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


def test_answer_from_evidence_summarizes_short_entity_queries() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="robot hand",
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="visual",
                modality=Modality.VISUAL,
                timestamp_seconds=43.01,
                text="visual frame sampled at 43.01 seconds",
                clip_start_seconds=37.01,
                clip_end_seconds=49.01,
                selection_rank=1,
                relevance_score=0.8,
                normalized_score=1.0,
                mmr_score=0.5,
                source_score_type="clip_scene",
                reason="test",
            ),
            SelectedCandidate(
                id="audio",
                modality=Modality.AUDIO,
                timestamp_seconds=45.0,
                text="Why don't you just admit that you're freaked out by my robot hand?",
                clip_start_seconds=40.0,
                clip_end_seconds=50.0,
                selection_rank=2,
                relevance_score=0.7,
                normalized_score=0.9,
                mmr_score=0.4,
                source_score_type="lexical_overlap",
                reason="test",
            ),
        ],
        metrics=CompressionMetrics(
            input_candidates=2,
            selected_candidates=2,
            visual_selected=1,
            audio_selected=1,
            estimated_candidate_reduction_ratio=1,
            estimated_candidate_reduction_percent=0,
            dropped_candidates=0,
            budget_preset_used=CompressionPreset.BALANCED,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    assert answer_from_evidence(compression) == (
        "Selected evidence indicates that visual evidence shows robot hand from 37.01s "
        "to 49.01s; transcript evidence mentions robot hand from 40.00s to 50.00s: "
        "Why don't you just admit that you're freaked out by my robot hand?"
    )


def test_answer_from_evidence_uses_transcript_for_question_not_visual_placeholder() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="How do top builders use AI to do the work of hundreds of engineers?",
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="visual",
                modality=Modality.VISUAL,
                timestamp_seconds=359.76,
                text="visual frame sampled at 359.76 seconds",
                selection_rank=1,
                relevance_score=1.0,
                normalized_score=1.0,
                mmr_score=0.5,
                source_score_type="clip_scene",
                reason="test",
            ),
            SelectedCandidate(
                id="audio",
                modality=Modality.AUDIO,
                timestamp_seconds=382.0,
                text="Top builders use AI to automate research, write code, and review work.",
                selection_rank=2,
                relevance_score=0.7,
                normalized_score=0.9,
                mmr_score=0.4,
                source_score_type="lexical_overlap",
                reason="test",
            ),
        ],
        metrics=CompressionMetrics(
            input_candidates=2,
            selected_candidates=2,
            visual_selected=1,
            audio_selected=1,
            estimated_candidate_reduction_ratio=1,
            estimated_candidate_reduction_percent=0,
            dropped_candidates=0,
            budget_preset_used=CompressionPreset.BALANCED,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )

    assert answer_from_evidence(compression) == (
        "Top builders use AI to automate research, write code, and review work."
    )


def test_answer_from_evidence_uses_transcript_for_mixed_av_question() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query=(
            "What does the presenter say about the demonstration "
            "showing a person and an on-screen skeleton?"
        ),
        preset=CompressionPreset.BALANCED,
        query_intent=QueryIntent.MIXED_AV,
        selected=[
            SelectedCandidate(
                id="visual",
                modality=Modality.VISUAL,
                timestamp_seconds=100,
                text="visual frame sampled during the demonstration",
                selection_rank=1,
                relevance_score=1,
                normalized_score=1,
                mmr_score=1,
                source_score_type="clip_scene",
                reason="test",
            ),
            SelectedCandidate(
                id="audio",
                modality=Modality.AUDIO,
                timestamp_seconds=105,
                text="The system tracks the person's joints in real time.",
                selection_rank=2,
                relevance_score=0.5,
                normalized_score=0.5,
                mmr_score=0.5,
                source_score_type="whisper",
                reason="test",
            ),
        ],
        metrics=CompressionMetrics(
            input_candidates=2,
            selected_candidates=2,
            visual_selected=1,
            audio_selected=1,
            estimated_candidate_reduction_ratio=1,
            estimated_candidate_reduction_percent=0,
            dropped_candidates=0,
            budget_preset_used=CompressionPreset.BALANCED,
        ),
    )

    assert (
        answer_from_evidence(compression)
        == "The system tracks the person's joints in real time."
    )


def test_answer_from_evidence_rejects_visual_only_ocr_for_non_text_question() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="How do top builders use AI?",
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="ocr",
                modality=Modality.VISUAL,
                timestamp_seconds=884.81,
                text="on-screen text near 884.81 seconds: ee ere",
                selection_rank=1,
                relevance_score=1.0,
                normalized_score=1.0,
                mmr_score=0.5,
                source_score_type="ocr",
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

    assert answer_from_evidence(compression) is None


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


def test_verify_answer_claims_accepts_presenter_attribution_prefix() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="What does the presenter say about the product demo?",
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="audio",
                modality=Modality.AUDIO,
                timestamp_seconds=1,
                text=(
                    "This is him actually running the product. "
                    "He creates a fun pose. "
                    "In this case, he's putting himself into the background."
                ),
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
        "The presenter says this is him actually running the product. "
        "He creates a fun pose."
    )

    assert verify_answer_claims(answer, compression) == answer


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
