from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True, slots=True)
class FrameEmbedding:
    frame_index: int
    timestamp_seconds: float
    vector: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SceneSegment:
    id: str
    start_seconds: float
    end_seconds: float
    frame_indexes: tuple[int, ...]
    mean_relevance: float = 0.0
    peak_relevance: float = 0.0


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding vectors must have equal dimensions")
    if not left:
        return 0.0

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def detect_scene_segments(
    embeddings: list[FrameEmbedding],
    relevance_by_frame: dict[int, float] | None = None,
    boundary_similarity_threshold: float = 0.82,
) -> list[SceneSegment]:
    if not 0 <= boundary_similarity_threshold <= 1:
        raise ValueError("boundary_similarity_threshold must be between 0 and 1")
    if not embeddings:
        return []

    ordered = sorted(embeddings, key=lambda item: (item.timestamp_seconds, item.frame_index))
    relevance_by_frame = relevance_by_frame or {}
    groups: list[list[FrameEmbedding]] = [[ordered[0]]]

    for previous, current in zip(ordered, ordered[1:]):
        similarity = cosine_similarity(previous.vector, current.vector)
        if similarity < boundary_similarity_threshold:
            groups.append([current])
        else:
            groups[-1].append(current)

    return [
        _segment_from_group(
            group=group,
            index=index,
            relevance_by_frame=relevance_by_frame,
        )
        for index, group in enumerate(groups)
    ]


def allocate_segment_budget(
    segments: list[SceneSegment],
    total_budget: int,
) -> dict[str, int]:
    if total_budget < 0:
        raise ValueError("total_budget must be non-negative")
    if not segments or total_budget == 0:
        return {}
    total_budget = min(
        total_budget,
        sum(max(len(segment.frame_indexes), 1) for segment in segments),
    )

    ordered = sorted(
        segments,
        key=lambda segment: (
            _segment_priority(segment),
            len(segment.frame_indexes),
            -segment.start_seconds,
        ),
        reverse=True,
    )
    allocation = {segment.id: 0 for segment in segments}

    for segment in ordered[:total_budget]:
        allocation[segment.id] = 1

    remaining = total_budget - sum(allocation.values())
    while remaining > 0:
        target = max(
            ordered,
            key=lambda segment: _budget_pressure(
                segment=segment,
                allocated=allocation[segment.id],
            ),
        )
        allocation[target.id] += 1
        remaining -= 1

    return allocation


def scene_by_frame_index(segments: list[SceneSegment]) -> dict[int, SceneSegment]:
    return {
        frame_index: segment
        for segment in segments
        for frame_index in segment.frame_indexes
    }


def _segment_from_group(
    group: list[FrameEmbedding],
    index: int,
    relevance_by_frame: dict[int, float],
) -> SceneSegment:
    relevance_scores = [relevance_by_frame.get(frame.frame_index, 0.0) for frame in group]
    return SceneSegment(
        id=f"scene-{index}",
        start_seconds=group[0].timestamp_seconds,
        end_seconds=group[-1].timestamp_seconds,
        frame_indexes=tuple(frame.frame_index for frame in group),
        mean_relevance=(
            sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        ),
        peak_relevance=max(relevance_scores, default=0.0),
    )


def _segment_priority(segment: SceneSegment) -> float:
    return segment.peak_relevance + (0.5 * segment.mean_relevance)


def _budget_pressure(segment: SceneSegment, allocated: int) -> float:
    capacity = max(len(segment.frame_indexes), 1)
    if allocated >= capacity:
        return -1.0
    return _segment_priority(segment) / (allocated + 1)
