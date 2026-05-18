from gist.core.query_intent import QueryIntent
from gist.core.schemas import Modality, SelectedCandidate
from gist.media.clips import adaptive_clip_span


def _selected(
    modality: Modality = Modality.VISUAL,
    timestamp_seconds: float = 50.0,
    scene_start_seconds: float | None = None,
    scene_end_seconds: float | None = None,
) -> SelectedCandidate:
    return SelectedCandidate(
        id="evidence",
        modality=modality,
        timestamp_seconds=timestamp_seconds,
        text="evidence",
        scene_start_seconds=scene_start_seconds,
        scene_end_seconds=scene_end_seconds,
        selection_rank=1,
        relevance_score=1.0,
        normalized_score=1.0,
        mmr_score=1.0,
        source_score_type="test",
        reason="test",
    )


def test_adaptive_clip_span_uses_scene_bounds_for_visual_evidence() -> None:
    span = adaptive_clip_span(
        item=_selected(scene_start_seconds=42.0, scene_end_seconds=48.0),
        query="show the vehicle",
        query_intent=QueryIntent.VISUAL_OBJECT_ACTION,
        video_duration_seconds=100.0,
    )

    assert span.start_seconds == 42.0
    assert span.end_seconds == 48.0
    assert "scene" in span.reason


def test_adaptive_clip_span_keeps_pre_context_for_before_queries() -> None:
    span = adaptive_clip_span(
        item=_selected(timestamp_seconds=50.0),
        query="what happens before launch",
        query_intent=QueryIntent.TEMPORAL_BEFORE_AFTER,
        video_duration_seconds=100.0,
    )

    assert span.start_seconds == 38.0
    assert span.end_seconds == 50.0


def test_adaptive_clip_span_keeps_wider_speech_window() -> None:
    span = adaptive_clip_span(
        item=_selected(modality=Modality.AUDIO, timestamp_seconds=4.0),
        query="what does the speaker say",
        query_intent=QueryIntent.SPEECH_SEMANTIC,
        video_duration_seconds=100.0,
    )

    assert span.start_seconds == 0.0
    assert span.end_seconds == 10.0
