import math
from dataclasses import dataclass

from gist.candidates.baseline import CandidateSet
from gist.core.query_intent import QueryIntent, route_query_intent
from gist.core.schemas import Candidate
from gist.core.scoring import lexical_relevance
from gist.core.temporal_query import parse_temporal_query, rank_temporal_pairs

DEFAULT_SEGMENT_SECONDS = 120.0
DEFAULT_MAX_SEGMENTS = 12


@dataclass(frozen=True, slots=True)
class SegmentCandidateGroup:
    id: str
    start_seconds: float
    end_seconds: float
    score: float
    candidates: tuple[Candidate, ...]


def shortlist_relevant_segments(
    candidates: CandidateSet,
    query: str,
    duration_seconds: float,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
) -> CandidateSet:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero")
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be greater than zero")
    if max_segments <= 0:
        raise ValueError("max_segments must be greater than zero")

    all_candidates = [*candidates.visual, *candidates.audio]
    groups = _segment_groups(
        candidates=all_candidates,
        query=query,
        duration_seconds=duration_seconds,
        segment_seconds=segment_seconds,
    )
    if len(groups) <= max_segments:
        return _with_segment_metadata(candidates, groups)

    selected_groups = sorted(
        groups,
        key=lambda group: (group.score, -group.start_seconds),
        reverse=True,
    )[:max_segments]
    selected_ids = {group.id for group in selected_groups}
    selected_ids.update(
        _temporal_group_ids(
            candidates=candidates.visual,
            groups=groups,
            query=query,
        )
    )
    opening_group_id = _opening_group_id(candidates.visual, groups, query)
    if opening_group_id is not None:
        selected_ids.add(opening_group_id)
    selected_ids.update(
        _global_audio_group_ids(
            groups,
            query,
            audio_ids={candidate.id for candidate in candidates.audio},
        )
    )
    visual = [
        _candidate_with_group(candidate, groups)
        for candidate in candidates.visual
        if _candidate_group_id(candidate, duration_seconds, segment_seconds) in selected_ids
    ]
    audio = [
        _candidate_with_group(candidate, groups)
        for candidate in candidates.audio
        if _candidate_group_id(candidate, duration_seconds, segment_seconds) in selected_ids
    ]
    return CandidateSet(visual=visual, audio=audio)


def _with_segment_metadata(
    candidates: CandidateSet,
    groups: list[SegmentCandidateGroup],
) -> CandidateSet:
    return CandidateSet(
        visual=[_candidate_with_group(candidate, groups) for candidate in candidates.visual],
        audio=[_candidate_with_group(candidate, groups) for candidate in candidates.audio],
    )


def _segment_groups(
    candidates: list[Candidate],
    query: str,
    duration_seconds: float,
    segment_seconds: float,
) -> list[SegmentCandidateGroup]:
    buckets: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        bucket_id = _candidate_group_id(candidate, duration_seconds, segment_seconds)
        buckets.setdefault(bucket_id, []).append(candidate)

    groups = [
        _group_from_candidates(
            group_id=group_id,
            candidates=items,
            query=query,
            duration_seconds=duration_seconds,
            segment_seconds=segment_seconds,
        )
        for group_id, items in buckets.items()
    ]
    return sorted(groups, key=lambda group: group.start_seconds)


def _group_from_candidates(
    group_id: str,
    candidates: list[Candidate],
    query: str,
    duration_seconds: float,
    segment_seconds: float,
) -> SegmentCandidateGroup:
    start, end = _group_bounds(
        group_id=group_id,
        candidates=candidates,
        duration_seconds=duration_seconds,
        segment_seconds=segment_seconds,
    )
    scores = [
        max(
            lexical_relevance(query, candidate),
            float(candidate.saliency_score or 0.0),
            float(candidate.temporal_anchor_score or 0.0),
            float(candidate.temporal_target_score or 0.0),
        )
        for candidate in candidates
    ]
    score = max(scores, default=0.0)
    return SegmentCandidateGroup(
        id=group_id,
        start_seconds=start,
        end_seconds=end,
        score=score,
        candidates=tuple(candidates),
    )


def _group_bounds(
    group_id: str,
    candidates: list[Candidate],
    duration_seconds: float,
    segment_seconds: float,
) -> tuple[float, float]:
    bounded = [
        (candidate.scene_start_seconds, candidate.scene_end_seconds)
        for candidate in candidates
        if candidate.scene_start_seconds is not None and candidate.scene_end_seconds is not None
    ]
    if bounded:
        start = min(item[0] for item in bounded if item[0] is not None)
        end = max(item[1] for item in bounded if item[1] is not None)
        return max(start, 0.0), min(max(end, start + 1.0), duration_seconds)

    try:
        bucket_index = int(group_id.rsplit("-", maxsplit=1)[-1])
        start = bucket_index * segment_seconds
    except ValueError:
        center = min(candidate.timestamp_seconds for candidate in candidates)
        start = math.floor(center / segment_seconds) * segment_seconds
    end = min(start + segment_seconds, duration_seconds)
    return start, max(end, start + 1.0)


def _candidate_group_id(
    candidate: Candidate,
    duration_seconds: float,
    segment_seconds: float,
) -> str:
    if candidate.segment_id and not candidate.segment_id.startswith("audio-window-"):
        return candidate.segment_id

    max_index = max(math.ceil(duration_seconds / segment_seconds) - 1, 0)
    index = min(int(candidate.timestamp_seconds // segment_seconds), max_index)
    return f"long-segment-{index:04d}"


def _candidate_with_group(
    candidate: Candidate,
    groups: list[SegmentCandidateGroup],
) -> Candidate:
    if candidate.segment_id and candidate.scene_start_seconds is not None:
        return candidate

    group = next(
        (
            item
            for item in groups
            if candidate in item.candidates or candidate.segment_id == item.id
        ),
        None,
    )
    if group is None:
        return candidate

    return candidate.model_copy(
        update={
            "segment_id": candidate.segment_id or group.id,
            "scene_start_seconds": candidate.scene_start_seconds
            if candidate.scene_start_seconds is not None
            else group.start_seconds,
            "scene_end_seconds": candidate.scene_end_seconds
            if candidate.scene_end_seconds is not None
            else group.end_seconds,
        }
    )


def _temporal_group_ids(
    candidates: list[Candidate],
    groups: list[SegmentCandidateGroup],
    query: str,
    max_distance_seconds: float = 300.0,
) -> set[str]:
    temporal_query = parse_temporal_query(query)
    scored = [
        candidate
        for candidate in candidates
        if candidate.temporal_anchor_score is not None
        and candidate.temporal_target_score is not None
    ]
    if temporal_query is None or not scored:
        return set()

    pairs = rank_temporal_pairs(
        scored,
        direction=temporal_query.direction,
        target_query=temporal_query.target,
        anchor_query=temporal_query.anchor,
        max_distance_seconds=max_distance_seconds,
    )
    selected_ids: set[str] = set()
    for _, anchor, target in pairs[:12]:
        selected_ids.add(_group_id_for_candidate(anchor, groups))
        selected_ids.add(_group_id_for_candidate(target, groups))
    return {group_id for group_id in selected_ids if group_id}


def _group_id_for_candidate(
    candidate: Candidate,
    groups: list[SegmentCandidateGroup],
) -> str:
    group = next((item for item in groups if candidate in item.candidates), None)
    return group.id if group is not None else candidate.segment_id or ""


def _opening_group_id(
    candidates: list[Candidate],
    groups: list[SegmentCandidateGroup],
    query: str,
) -> str | None:
    query_lower = query.lower()
    if not any(
        marker in query_lower
        for marker in ("opening", "beginning", " start", "first")
    ):
        return None
    if not candidates:
        return None

    opening = min(
        candidates,
        key=lambda candidate: (candidate.timestamp_seconds, candidate.id),
    )
    return _group_id_for_candidate(opening, groups) or None


def _global_audio_group_ids(
    groups: list[SegmentCandidateGroup],
    query: str,
    audio_ids: set[str],
) -> set[str]:
    query_intent, _reason = route_query_intent(query)
    if query_intent != QueryIntent.GLOBAL_SUMMARY:
        return set()

    audio_groups = [
        group
        for group in groups
        if any(candidate.id in audio_ids for candidate in group.candidates)
    ]
    if not audio_groups:
        return set()

    start = min(group.start_seconds for group in audio_groups)
    end = max(group.end_seconds for group in audio_groups)
    targets = (
        start + ((end - start) * fraction)
        for fraction in (1 / 6, 1 / 2, 5 / 6)
    )
    return {
        min(
            audio_groups,
            key=lambda group: abs(
                ((group.start_seconds + group.end_seconds) / 2) - target
            ),
        ).id
        for target in targets
    }
