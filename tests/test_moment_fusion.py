from pathlib import Path

import pytest

from gist.candidates.baseline import CandidateSet
from gist.candidates.moments import fuse_transcript_moments
from gist.core.schemas import Candidate


def test_fuse_transcript_moments_attaches_nearest_visual_grounding() -> None:
    candidates = CandidateSet(
        visual=[
            Candidate(
                id="v-near",
                timestamp_seconds=44,
                text="visual frame sampled",
                asset_path=Path("frame.jpg"),
                segment_id="scene-1",
            ),
            Candidate(id="v-far", timestamp_seconds=90, text="visual frame sampled"),
        ],
        audio=[
            Candidate(
                id="a-hit",
                timestamp_seconds=45,
                text="the speaker is freaked out by the robot hand",
                asset_path=Path("audio.wav"),
                scene_start_seconds=30,
                scene_end_seconds=60,
            )
        ],
    )

    fused = fuse_transcript_moments(candidates, query="why is he afraid of robot hand")

    assert fused.visual == []
    assert len(fused.audio) == 1
    moment = fused.audio[0]
    assert moment.id == "a-hit+v-near"
    assert moment.asset_path == Path("frame.jpg")
    assert moment.segment_id == "scene-1"
    assert moment.scene_start_seconds == 30
    assert moment.scene_end_seconds == 60


def test_fuse_transcript_moments_drops_weak_audio_noise() -> None:
    candidates = CandidateSet(
        visual=[],
        audio=[
            Candidate(id="a-noise", timestamp_seconds=10, text="ambient music"),
            Candidate(id="a-hit", timestamp_seconds=20, text="robot hand explanation"),
        ],
    )

    fused = fuse_transcript_moments(candidates, query="robot hand")

    assert [candidate.id for candidate in fused.audio] == ["a-hit"]


def test_fuse_transcript_moments_validates_radius() -> None:
    with pytest.raises(ValueError, match="visual_radius_seconds"):
        fuse_transcript_moments(
            CandidateSet(visual=[], audio=[]),
            query="x",
            visual_radius_seconds=0,
        )


def test_fuse_transcript_moments_caps_audio_moments_by_relevance() -> None:
    candidates = CandidateSet(
        visual=[],
        audio=[
            Candidate(id="a-low", timestamp_seconds=10, text="robot hand"),
            Candidate(
                id="a-high",
                timestamp_seconds=20,
                text="robot hand afraid chased nightmares",
            ),
        ],
    )

    fused = fuse_transcript_moments(candidates, query="why afraid robot hand", max_audio_moments=1)

    assert [candidate.id for candidate in fused.audio] == ["a-high"]


def test_fuse_transcript_moments_reranks_why_answer_signals() -> None:
    candidates = CandidateSet(
        visual=[],
        audio=[
            Candidate(id="a-keyword", timestamp_seconds=10, text="robot hand robot hand"),
            Candidate(
                id="a-answer",
                timestamp_seconds=20,
                text="he is freaked out and has nightmares about being chased",
            ),
        ],
    )

    fused = fuse_transcript_moments(
        candidates,
        query="Why is the man afraid of the robot hand?",
        min_audio_relevance=0.0,
        max_audio_moments=1,
    )

    assert [candidate.id for candidate in fused.audio] == ["a-answer"]


def test_fuse_transcript_moments_preserves_audio_near_relevant_mixed_visual() -> None:
    candidates = CandidateSet(
        visual=[
            Candidate(
                id="visual-hit",
                timestamp_seconds=100,
                text="visual frame sampled during body tracking",
                saliency_score=0.9,
            ),
            Candidate(
                id="visual-noise",
                timestamp_seconds=500,
                text="unrelated frame",
                saliency_score=0.1,
            ),
        ],
        audio=[
            Candidate(
                id="audio-near",
                timestamp_seconds=105,
                text="the joints move in real time",
            ),
            Candidate(
                id="audio-far",
                timestamp_seconds=700,
                text="closing remarks",
            ),
        ],
    )

    fused = fuse_transcript_moments(
        candidates,
        query="person and on-screen skeleton",
        preserve_visual_context_audio=True,
    )

    assert [candidate.id for candidate in fused.audio] == ["audio-near+visual-hit"]
