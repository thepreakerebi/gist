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

    assert pairs[0][2].id == "title-successor"


def test_rank_temporal_pairs_skips_frames_from_same_persistent_scene() -> None:
    candidates = [
        Candidate(
            id="anchor",
            timestamp_seconds=100,
            text="on-screen text: Further Reading Materials",
            segment_id="scene-1",
            temporal_anchor_score=0.95,
            temporal_target_score=0.1,
        ),
        Candidate(
            id="same-slide",
            timestamp_seconds=108,
            text="on-screen text: Further Reading Materials",
            segment_id="scene-1",
            temporal_anchor_score=0.9,
            temporal_target_score=0.8,
        ),
        Candidate(
            id="next-slide",
            timestamp_seconds=116,
            text="on-screen text: Next Week",
            segment_id="scene-2",
            temporal_anchor_score=0.1,
            temporal_target_score=0.7,
        ),
    ]

    pairs = rank_temporal_pairs(
        candidates,
        direction="after",
        target_query="What slide appears",
    )

    anchor_pair = next(pair for pair in pairs if pair[1].id == "anchor")
    assert anchor_pair[2].id == "next-slide"


def test_rank_temporal_pairs_prefers_later_named_slide_over_intermediate_demo() -> None:
    candidates = [
        Candidate(
            id="anchor",
            timestamp_seconds=100,
            text="on-screen text: WorldWide Telescope",
            segment_id="scene-1",
            temporal_anchor_score=0.95,
            temporal_target_score=0.1,
        ),
        Candidate(
            id="demo",
            timestamp_seconds=120,
            text="on-screen text: | = @ |",
            segment_id="scene-2",
            temporal_anchor_score=0.1,
            temporal_target_score=0.7,
        ),
        Candidate(
            id="next-title",
            timestamp_seconds=160,
            text="on-screen text: FUN LABS",
            segment_id="scene-3",
            temporal_anchor_score=0.1,
            temporal_target_score=0.65,
        ),
    ]

    pairs = rank_temporal_pairs(
        candidates,
        direction="after",
        target_query="What slide appears",
    )

    anchor_pair = next(pair for pair in pairs if pair[1].id == "anchor")
    assert anchor_pair[2].id == "next-title"


def test_rank_temporal_pairs_penalizes_noisy_ocr_targets() -> None:
    candidates = [
        Candidate(
            id="anchor",
            timestamp_seconds=100,
            text="on-screen text: Bio-Inspired Motor Control",
            segment_id="scene-1",
            temporal_anchor_score=0.95,
            temporal_target_score=0.1,
        ),
        Candidate(
            id="noisy-target",
            timestamp_seconds=140,
            text="on-screen text near 140 seconds: SBIRU— aS ‘ .w",
            segment_id="scene-2",
            temporal_anchor_score=0.1,
            temporal_target_score=0.95,
        ),
        Candidate(
            id="clean-target",
            timestamp_seconds=180,
            text="on-screen text near 180 seconds: Legged Locomotion in Nature",
            segment_id="scene-3",
            temporal_anchor_score=0.1,
            temporal_target_score=0.7,
        ),
    ]

    pairs = rank_temporal_pairs(
        candidates,
        direction="after",
        target_query="What slide appears",
    )

    anchor_pair = next(pair for pair in pairs if pair[1].id == "anchor")
    assert anchor_pair[2].id == "clean-target"


def test_rank_temporal_pairs_prefers_early_anchor_for_opening_queries() -> None:
    candidates = [
        Candidate(
            id="opening-title",
            timestamp_seconds=5,
            text="on-screen text: Bio-Inspired Motor Control",
            segment_id="scene-1",
            temporal_anchor_score=0.55,
            temporal_target_score=0.1,
        ),
        Candidate(
            id="opening-target",
            timestamp_seconds=220,
            text="on-screen text: Legged Locomotion and Conceptual Models",
            segment_id="scene-2",
            temporal_anchor_score=0.1,
            temporal_target_score=0.7,
        ),
        Candidate(
            id="late-anchor",
            timestamp_seconds=4600,
            text="on-screen text: Course title recap",
            segment_id="scene-3",
            temporal_anchor_score=0.95,
            temporal_target_score=0.1,
        ),
        Candidate(
            id="late-target",
            timestamp_seconds=4680,
            text="on-screen text: Further Reading Materials",
            segment_id="scene-4",
            temporal_anchor_score=0.1,
            temporal_target_score=0.7,
        ),
    ]

    pairs = rank_temporal_pairs(
        candidates,
        direction="after",
        target_query="What slide appears",
        anchor_query="the opening course title slide",
    )

    assert pairs[0][1].id == "opening-title"
    assert pairs[0][2].id == "opening-target"
