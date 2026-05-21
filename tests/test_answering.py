from gist.core.answering import answer_from_evidence
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
