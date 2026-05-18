from dataclasses import dataclass
import re

from gist.core.query_intent import QueryIntent
from gist.core.schemas import Modality, SelectedCandidate


DEFAULT_CLIP_DURATION_SECONDS = 8.0
MAX_CLIP_DURATION_SECONDS = 12.0

_PRE_CONTEXT_TERMS = {"before", "prior"}
_POST_CONTEXT_TERMS = {"after", "then", "next", "following"}


@dataclass(frozen=True, slots=True)
class ClipSpan:
    start_seconds: float
    end_seconds: float
    reason: str

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def adaptive_clip_span(
    item: SelectedCandidate,
    query: str,
    query_intent: QueryIntent | None,
    video_duration_seconds: float,
    default_duration_seconds: float = DEFAULT_CLIP_DURATION_SECONDS,
    max_duration_seconds: float = MAX_CLIP_DURATION_SECONDS,
) -> ClipSpan:
    if video_duration_seconds <= 0:
        raise ValueError("video_duration_seconds must be greater than zero")
    if default_duration_seconds <= 0:
        raise ValueError("default_duration_seconds must be greater than zero")
    if max_duration_seconds < default_duration_seconds:
        raise ValueError("max_duration_seconds must be at least default_duration_seconds")

    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    if item.modality == Modality.VISUAL and _has_scene_bounds(item):
        return _scene_span(item, video_duration_seconds, max_duration_seconds)

    if query_intent == QueryIntent.TEMPORAL_BEFORE_AFTER:
        if query_terms & _PRE_CONTEXT_TERMS:
            return _bounded_span(
                start=item.timestamp_seconds - max_duration_seconds,
                end=item.timestamp_seconds + (default_duration_seconds / 2),
                video_duration_seconds=video_duration_seconds,
                max_duration_seconds=max_duration_seconds,
                reason="temporal query kept stronger pre-context",
            )
        if query_terms & _POST_CONTEXT_TERMS:
            return _bounded_span(
                start=item.timestamp_seconds - (default_duration_seconds / 2),
                end=item.timestamp_seconds + max_duration_seconds,
                video_duration_seconds=video_duration_seconds,
                max_duration_seconds=max_duration_seconds,
                reason="temporal query kept stronger post-context",
            )
        return _centered_span(
            center_seconds=item.timestamp_seconds,
            duration_seconds=max_duration_seconds,
            video_duration_seconds=video_duration_seconds,
            reason="temporal query kept wider continuity context",
        )

    if item.modality == Modality.AUDIO or query_intent == QueryIntent.SPEECH_SEMANTIC:
        return _centered_span(
            center_seconds=item.timestamp_seconds,
            duration_seconds=min(max_duration_seconds, default_duration_seconds + 2.0),
            video_duration_seconds=video_duration_seconds,
            reason="speech evidence kept a wider transcript-aligned window",
        )

    return _centered_span(
        center_seconds=item.timestamp_seconds,
        duration_seconds=default_duration_seconds,
        video_duration_seconds=video_duration_seconds,
        reason="default evidence clip window",
    )


def _has_scene_bounds(item: SelectedCandidate) -> bool:
    return item.scene_start_seconds is not None and item.scene_end_seconds is not None


def _scene_span(
    item: SelectedCandidate,
    video_duration_seconds: float,
    max_duration_seconds: float,
) -> ClipSpan:
    assert item.scene_start_seconds is not None
    assert item.scene_end_seconds is not None
    return _bounded_span(
        start=item.scene_start_seconds,
        end=max(item.scene_end_seconds, item.scene_start_seconds + 1.0),
        video_duration_seconds=video_duration_seconds,
        max_duration_seconds=max_duration_seconds,
        reason="visual evidence used scene-aware clip bounds",
    )


def _centered_span(
    center_seconds: float,
    duration_seconds: float,
    video_duration_seconds: float,
    reason: str,
) -> ClipSpan:
    half_duration = duration_seconds / 2
    start = center_seconds - half_duration
    end = center_seconds + half_duration
    if start < 0:
        end = min(end - start, video_duration_seconds)
        start = 0.0
    if end > video_duration_seconds:
        start = max(video_duration_seconds - duration_seconds, 0.0)
        end = video_duration_seconds
    return ClipSpan(start_seconds=start, end_seconds=end, reason=reason)


def _bounded_span(
    start: float,
    end: float,
    video_duration_seconds: float,
    max_duration_seconds: float,
    reason: str,
) -> ClipSpan:
    start = max(start, 0.0)
    end = min(max(end, start), video_duration_seconds)
    if end - start > max_duration_seconds:
        end = min(start + max_duration_seconds, video_duration_seconds)
    if end <= start:
        end = min(start + 1.0, video_duration_seconds)
    if end - start < 1.0 and start > 0:
        start = max(end - 1.0, 0.0)
    return ClipSpan(start_seconds=start, end_seconds=end, reason=reason)
