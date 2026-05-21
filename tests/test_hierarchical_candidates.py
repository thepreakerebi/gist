from pathlib import Path

from gist.candidates.baseline import CandidateSet
from gist.candidates.hierarchical import shortlist_relevant_segments
from gist.core.schemas import Candidate


def test_shortlist_relevant_segments_keeps_best_query_segments() -> None:
    candidates = CandidateSet(
        visual=[
            Candidate(id="v-early", timestamp_seconds=10, text="intro slide"),
            Candidate(id="v-hit", timestamp_seconds=130, text="refund policy diagram"),
            Candidate(id="v-late", timestamp_seconds=260, text="closing slide"),
        ],
        audio=[
            Candidate(id="a-early", timestamp_seconds=12, text="welcome everyone"),
            Candidate(id="a-hit", timestamp_seconds=135, text="refund policy explained"),
            Candidate(id="a-late", timestamp_seconds=270, text="thanks for watching"),
        ],
    )

    shortlisted = shortlist_relevant_segments(
        candidates=candidates,
        query="refund policy",
        duration_seconds=360,
        segment_seconds=120,
        max_segments=1,
    )

    assert [candidate.id for candidate in shortlisted.visual] == ["v-hit"]
    assert [candidate.id for candidate in shortlisted.audio] == ["a-hit"]
    assert shortlisted.visual[0].segment_id == "long-segment-0001"
    assert shortlisted.visual[0].scene_start_seconds == 120
    assert shortlisted.visual[0].scene_end_seconds == 240


def test_shortlist_relevant_segments_preserves_existing_scene_bounds() -> None:
    candidates = CandidateSet(
        visual=[
            Candidate(
                id="v-scene",
                timestamp_seconds=35,
                text="speaker enters",
                asset_path=Path("frame.jpg"),
                segment_id="scene-7",
                scene_start_seconds=30,
                scene_end_seconds=45,
            )
        ],
        audio=[],
    )

    shortlisted = shortlist_relevant_segments(
        candidates=candidates,
        query="speaker",
        duration_seconds=120,
        max_segments=1,
    )

    assert shortlisted.visual[0].segment_id == "scene-7"
    assert shortlisted.visual[0].scene_start_seconds == 30
    assert shortlisted.visual[0].scene_end_seconds == 45


def test_shortlist_relevant_segments_handles_segment_ids_without_bounds() -> None:
    candidates = CandidateSet(
        visual=[
            Candidate(
                id="v-custom",
                timestamp_seconds=130,
                text="refund policy",
                segment_id="custom-segment",
            )
        ],
        audio=[],
    )

    shortlisted = shortlist_relevant_segments(
        candidates=candidates,
        query="refund policy",
        duration_seconds=360,
        segment_seconds=120,
        max_segments=1,
    )

    assert shortlisted.visual[0].segment_id == "custom-segment"
    assert shortlisted.visual[0].scene_start_seconds == 120
    assert shortlisted.visual[0].scene_end_seconds == 240


def test_shortlist_relevant_segments_uses_model_saliency_when_available() -> None:
    candidates = CandidateSet(
        visual=[
            Candidate(
                id="v-low",
                timestamp_seconds=10,
                text="anything",
                saliency_score=0.1,
            ),
            Candidate(
                id="v-high",
                timestamp_seconds=250,
                text="anything",
                saliency_score=0.9,
            ),
        ],
        audio=[],
    )

    shortlisted = shortlist_relevant_segments(
        candidates=candidates,
        query="unmatched query",
        duration_seconds=360,
        segment_seconds=120,
        max_segments=1,
    )

    assert [candidate.id for candidate in shortlisted.visual] == ["v-high"]
