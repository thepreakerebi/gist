from gist.vision.scene import (
    FrameEmbedding,
    SceneSegment,
    allocate_segment_budget,
    cosine_similarity,
    detect_scene_segments,
)


def test_cosine_similarity_handles_orthogonal_vectors() -> None:
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == 0.0


def test_detect_scene_segments_splits_on_adjacent_embedding_drop() -> None:
    embeddings = [
        FrameEmbedding(frame_index=0, timestamp_seconds=0.0, vector=(1.0, 0.0)),
        FrameEmbedding(frame_index=1, timestamp_seconds=1.0, vector=(0.95, 0.05)),
        FrameEmbedding(frame_index=2, timestamp_seconds=2.0, vector=(0.0, 1.0)),
    ]

    segments = detect_scene_segments(
        embeddings,
        relevance_by_frame={0: 0.1, 1: 0.5, 2: 0.9},
        boundary_similarity_threshold=0.8,
    )

    assert [segment.frame_indexes for segment in segments] == [(0, 1), (2,)]
    assert segments[0].id == "scene-0"
    assert segments[0].peak_relevance == 0.5
    assert segments[1].start_seconds == 2.0


def test_allocate_segment_budget_prioritizes_relevant_segments() -> None:
    allocation = allocate_segment_budget(
        [
            SceneSegment(
                id="scene-low",
                start_seconds=0.0,
                end_seconds=2.0,
                frame_indexes=(0, 1),
                mean_relevance=0.1,
                peak_relevance=0.2,
            ),
            SceneSegment(
                id="scene-high",
                start_seconds=3.0,
                end_seconds=5.0,
                frame_indexes=(2, 3),
                mean_relevance=0.8,
                peak_relevance=0.9,
            ),
        ],
        total_budget=2,
    )

    assert allocation["scene-high"] == 1
    assert allocation["scene-low"] == 1


def test_allocate_segment_budget_caps_at_segment_capacity() -> None:
    allocation = allocate_segment_budget(
        [
            SceneSegment(
                id="scene",
                start_seconds=0.0,
                end_seconds=0.0,
                frame_indexes=(0,),
            ),
        ],
        total_budget=10,
    )

    assert allocation == {"scene": 1}
