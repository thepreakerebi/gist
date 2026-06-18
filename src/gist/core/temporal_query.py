import re
from dataclasses import dataclass
from typing import Protocol, TypeVar


@dataclass(frozen=True, slots=True)
class TemporalQuery:
    direction: str
    target: str
    anchor: str


class TemporalCandidate(Protocol):
    timestamp_seconds: float
    text: str
    segment_id: str | None
    temporal_anchor_score: float | None
    temporal_target_score: float | None


TemporalCandidateT = TypeVar("TemporalCandidateT", bound=TemporalCandidate)


def parse_temporal_query(query: str) -> TemporalQuery | None:
    match = re.search(r"\b(after|before)\b", query, flags=re.IGNORECASE)
    if match is None:
        return None

    target = query[: match.start()].strip(" ,?.")
    anchor = query[match.end() :].strip(" ,?.")
    if not target or not anchor:
        return None
    return TemporalQuery(
        direction=match.group(1).lower(),
        target=target,
        anchor=anchor,
    )


def rank_temporal_pairs(
    candidates: list[TemporalCandidateT],
    direction: str,
    target_query: str,
    max_distance_seconds: float = 120.0,
) -> list[tuple[float, TemporalCandidateT, TemporalCandidateT]]:
    scored = [
        candidate
        for candidate in candidates
        if candidate.temporal_anchor_score is not None
        and candidate.temporal_target_score is not None
    ]
    pairs: list[tuple[float, TemporalCandidateT, TemporalCandidateT]] = []
    for anchor in scored:
        directional = [
            candidate
            for candidate in scored
            if _is_directional(
                candidate.timestamp_seconds,
                anchor.timestamp_seconds,
                direction,
            )
            and abs(candidate.timestamp_seconds - anchor.timestamp_seconds)
            <= max_distance_seconds
        ]
        for target in sorted(
            directional,
            key=lambda candidate: abs(
                candidate.timestamp_seconds - anchor.timestamp_seconds
            ),
        )[:1]:
            distance = abs(target.timestamp_seconds - anchor.timestamp_seconds)
            proximity = max(1.0 - (distance / max_distance_seconds), 0.0)
            transition_bonus = (
                0.1
                if anchor.segment_id
                and target.segment_id
                and anchor.segment_id != target.segment_id
                else 0.0
            )
            score = (
                float(anchor.temporal_anchor_score or 0.0)
                + float(target.temporal_target_score or 0.0)
                + (0.05 * proximity)
                + transition_bonus
                + _target_text_bonus(target_query, target.text)
            )
            pairs.append((score, anchor, target))
    return sorted(
        pairs,
        key=lambda item: (
            item[0],
            -abs(item[2].timestamp_seconds - item[1].timestamp_seconds),
            -item[1].timestamp_seconds,
        ),
        reverse=True,
    )


def _is_directional(
    candidate_timestamp: float,
    anchor_timestamp: float,
    direction: str,
) -> bool:
    if direction == "after":
        return candidate_timestamp > anchor_timestamp
    return candidate_timestamp < anchor_timestamp


def _target_text_bonus(target_query: str, evidence_text: str) -> float:
    target_terms = set(re.findall(r"[a-z0-9]+", target_query.lower()))
    if not target_terms & {"name", "text", "title", "word", "words"}:
        return 0.0
    if ":" not in evidence_text:
        return 0.0

    ocr_text = evidence_text.split(":", maxsplit=1)[1]
    informative = [
        token
        for token in re.findall(r"[A-Za-z0-9]+", ocr_text)
        if len(token) >= 3
    ]
    if not informative:
        return 0.0
    bonus = 0.2 if len(informative) <= 8 else 0.05
    if any(len(token) >= 4 and token.isupper() for token in informative):
        bonus += 0.15
    return bonus
