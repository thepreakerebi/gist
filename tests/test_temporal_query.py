from gist.core.schemas import Candidate
from gist.core.temporal_query import parse_temporal_query, rank_temporal_pairs


def test_parse_temporal_query_extracts_target_anchor_and_direction() -> None:
    parsed = parse_temporal_query(
        "What title appears after the video editing interface is shown?"
    )

    assert parsed is not None
    assert parsed.direction == "after"
    assert parsed.target == "What title appears"
    assert parsed.anchor == "the video editing interface is shown"


def test_parse_temporal_query_rejects_incomplete_query() -> None:
    assert parse_temporal_query("What happens after?") is None


def test_rank_temporal_pairs_prefers_title_bearing_scene_transition() -> None:
    candidates = [
        Candidate(
            id="strong-anchor",
            timestamp_seconds=10,
            text="editing interface",
            segment_id="scene-1",
            temporal_anchor_score=0.9,
            temporal_target_score=0.1,
        ),
        Candidate(
            id="noisy-successor",
            timestamp_seconds=18,
            text="on-screen text near 18 seconds: | = te |",
            segment_id="scene-1",
            temporal_anchor_score=0.1,
            temporal_target_score=0.8,
        ),
        Candidate(
            id="slightly-weaker-anchor",
            timestamp_seconds=40,
            text="another editing interface",
            segment_id="scene-2",
            temporal_anchor_score=0.82,
            temporal_target_score=0.1,
        ),
        Candidate(
            id="title-successor",
            timestamp_seconds=48,
            text="on-screen text near 48 seconds: KINECT",
            segment_id="scene-3",
            temporal_anchor_score=0.1,
            temporal_target_score=0.76,
        ),
    ]

    pairs = rank_temporal_pairs(
        candidates,
        direction="after",
        target_query="What title appears",
    )

    assert pairs[0][1].id == "slightly-weaker-anchor"
    assert pairs[0][2].id == "title-successor"
