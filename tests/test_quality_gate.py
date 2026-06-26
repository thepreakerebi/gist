from gist.core.presets import CompressionPreset
from gist.core.quality_gate import apply_quality_gate, quality_warnings
from gist.core.query_intent import QueryIntent
from gist.core.schemas import CompressionMetrics, CompressionResponse, Modality, SelectedCandidate


def test_quality_gate_passes_grounded_transcript_answer() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="What does the presenter say?",
        answer="The presenter says he creates a fun pose.",
        preset=CompressionPreset.BALANCED,
        query_intent=QueryIntent.MIXED_AV,
        selected=[
            _selected_audio(
                text="He creates a fun pose.",
                support_label="strong",
                grounding_label="direct",
            )
        ],
        metrics=_metrics(token_reduction=99.0),
    )

    gated = apply_quality_gate(compression)

    assert gated.quality_warnings == []


def test_quality_gate_flags_weak_speech_report() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="What does the presenter say?",
        answer="Yes.",
        preset=CompressionPreset.BALANCED,
        query_intent=QueryIntent.MIXED_AV,
        selected=[
            SelectedCandidate(
                id="v1",
                modality=Modality.VISUAL,
                timestamp_seconds=10,
                text="on-screen text near 10.00 seconds: x",
                selection_rank=1,
                relevance_score=0.1,
                normalized_score=0.1,
                mmr_score=0.1,
                support_label="weak",
                grounding_label="weak",
                source_score_type="clip_scene",
                reason="selected",
            )
        ],
        metrics=_metrics(token_reduction=25.0),
    )

    warnings = quality_warnings(compression)

    assert {warning.code for warning in warnings} == {
        "weak_answer",
        "missing_transcript_evidence",
        "ungrounded_evidence",
        "weak_evidence_support",
        "low_token_reduction",
    }
    assert any(warning.severity == "error" for warning in warnings)


def test_quality_gate_flags_noisy_global_summary_transcripts() -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="What are the main topics covered throughout this lecture?",
        answer="The video covers: sensors and control; power.",
        preset=CompressionPreset.BALANCED,
        query_intent=QueryIntent.GLOBAL_SUMMARY,
        selected=[
            _selected_audio(
                text=(
                    "So this is a bit of a vlog in the place but this is all "
                    "of you all the courses and we have a lot of."
                ),
                support_label="strong",
                grounding_label="direct",
            ),
            _selected_audio(
                text=(
                    "The interesting thing is that this role is not having "
                    "any sensors and a simple control arm."
                ),
                support_label="strong",
                grounding_label="direct",
            ),
        ],
        metrics=_metrics(token_reduction=99.0),
    )

    warnings = quality_warnings(compression)

    assert "noisy_transcript_evidence" in {warning.code for warning in warnings}


def _selected_audio(
    text: str,
    support_label: str,
    grounding_label: str,
) -> SelectedCandidate:
    return SelectedCandidate(
        id="a1",
        modality=Modality.AUDIO,
        timestamp_seconds=10,
        text=text,
        selection_rank=1,
        relevance_score=1,
        normalized_score=1,
        mmr_score=1,
        support_label=support_label,
        grounding_label=grounding_label,
        source_score_type="lexical_overlap",
        reason="selected",
    )


def _metrics(token_reduction: float) -> CompressionMetrics:
    return CompressionMetrics(
        input_candidates=10,
        selected_candidates=1,
        visual_selected=0,
        audio_selected=1,
        estimated_candidate_reduction_ratio=0.1,
        estimated_candidate_reduction_percent=90,
        dropped_candidates=9,
        budget_preset_used=CompressionPreset.BALANCED,
        estimated_token_reduction_percent=token_reduction,
    )
