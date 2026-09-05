"""ToMe-style tail merging for low-salience Gist candidates.

Bolya et al. (2023) accelerate a ViT by merging redundant *tokens* inside the
encoder using bipartite soft matching: the token set is split into two disjoint
halves, every token in one half is matched to its most similar partner in the
other, and the most similar pairs are averaged away. Gist adapts that idea one
level up and one stage earlier — it merges *candidates* (sampled frames and
audio windows) **before** any encoder runs.

The motivation is that hard pruning throws away the entire tail. A candidate
that ranks 40th is usually not worthless; it is usually near-duplicate evidence
for something already selected, or weak evidence that only becomes meaningful in
aggregate. Merging keeps a bounded, cheap summary of that tail instead of
discarding it outright, while the hard-kept head is never touched — merging what
matters is exactly what ToMe warns against.

This module is deliberately free of any dependency on the compressor. It
consumes anything satisfying :class:`MergeableCandidate` and returns
:class:`TailMergeGroup` *decisions*, leaving the caller to materialize whatever
candidate type it uses. That keeps the merge policy unit-testable in isolation
and avoids a circular import with ``gist.core.compressor``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from gist.core.schemas import Modality
from gist.core.scoring import temporal_similarity, text_similarity

# Default blend between "these two candidates are close in time" and "these two
# candidates say the same thing". Temporal dominates because pre-encoder
# redundancy in video is overwhelmingly temporal (the DyCoke/MMG-Vid finding),
# and because a frame candidate's text is often sparse OCR with little signal.
DEFAULT_TEXT_SIMILARITY_WEIGHT = 0.35

# A pair must be at least this similar to be worth merging. Below it the two
# candidates are genuinely different evidence and merging would blur them.
DEFAULT_MIN_SIMILARITY = 0.35

# Hard temporal gate, applied before the blended score. Two candidates that look
# or read alike but sit far apart in time are *different moments*, not redundant
# evidence — a lecture slide that recurs at 02:00 and at 20:00 is two events. A
# blended score alone would let identical text carry a pair over the threshold on
# its own, producing a merged group whose weighted timestamp points at dead air
# between the two clusters. Roughly 1.5 sigma; beyond it, no merge is possible.
MIN_TEMPORAL_SIMILARITY = 0.1

# Fraction of the tail eligible to be merged away (ToMe's ``r``, expressed as a
# proportion rather than an absolute count so it scales with pool size).
DEFAULT_MERGE_RATIO = 0.5

# Hard ceiling on how many merged groups may be appended to a selection. Tail
# merging must never quietly inflate the evidence budget it was introduced to
# protect.
DEFAULT_MAX_GROUPS = 2

# Merged text is a summary, not a transcript; keep it bounded so a merged group
# cannot cost more downstream tokens than the candidates it replaced.
MAX_MERGED_TEXT_CHARS = 320


class MergeableCandidate(Protocol):
    """Structural type for anything tail merging can operate on."""

    id: str
    modality: Modality
    timestamp_seconds: float
    text: str
    relevance_score: float
    normalized_score: float


@dataclass(frozen=True, slots=True)
class TailMergeGroup:
    """One merged cluster of tail candidates.

    ``representative_id`` is the highest-scoring member, whose asset (frame path
    or audio window) stands in for the group downstream. The remaining
    ``merged_ids`` are folded into it and are recoverable from the artifact, so
    a merge is always auditable after the fact.
    """

    representative_id: str
    merged_ids: tuple[str, ...]
    modality: Modality
    timestamp_seconds: float
    text: str
    relevance_score: float
    normalized_score: float
    similarity: float

    @property
    def size(self) -> int:
        return len(self.merged_ids) + 1


def merge_tail(
    tail: list[MergeableCandidate],
    *,
    merge_ratio: float = DEFAULT_MERGE_RATIO,
    max_groups: int = DEFAULT_MAX_GROUPS,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    temporal_sigma_seconds: float,
    text_similarity_weight: float = DEFAULT_TEXT_SIMILARITY_WEIGHT,
) -> list[TailMergeGroup]:
    """Merge redundant candidates in the unselected tail via bipartite matching.

    Returns at most ``max_groups`` groups, ranked by the normalized score of
    their representative, so the caller can append the strongest summaries of
    the tail without unbounded budget growth. Returns ``[]`` when the tail is
    too small or too dissimilar to merge, which is the common and correct case
    for short candidate pools.
    """

    if merge_ratio <= 0 or max_groups <= 0 or len(tail) < 2:
        return []

    groups: list[TailMergeGroup] = []
    # Cross-modal merging is never valid: a frame and an audio window are not
    # redundant with one another even when they are simultaneous, and collapsing
    # them would destroy the modality accounting the budget arbitration depends on.
    for modality in (Modality.VISUAL, Modality.AUDIO):
        members = [item for item in tail if item.modality == modality]
        groups.extend(
            _merge_within_modality(
                members,
                merge_ratio=merge_ratio,
                min_similarity=min_similarity,
                temporal_sigma_seconds=temporal_sigma_seconds,
                text_similarity_weight=text_similarity_weight,
            )
        )

    groups.sort(key=lambda group: (-group.normalized_score, group.timestamp_seconds))
    return groups[:max_groups]


def _merge_within_modality(
    members: list[MergeableCandidate],
    *,
    merge_ratio: float,
    min_similarity: float,
    temporal_sigma_seconds: float,
    text_similarity_weight: float,
) -> list[TailMergeGroup]:
    if len(members) < 2:
        return []

    # ToMe's bipartite split. Alternating along the time axis (rather than a
    # random or contiguous split) guarantees that every candidate's temporal
    # neighbours land in the opposite set, so the near-duplicates we most want
    # to merge are always eligible to match each other.
    ordered = sorted(members, key=lambda item: (item.timestamp_seconds, item.id))
    set_a = ordered[0::2]
    set_b = ordered[1::2]
    if not set_a or not set_b:
        return []

    budget = math.floor(merge_ratio * len(ordered))
    if budget <= 0:
        return []

    edges: list[tuple[float, MergeableCandidate, MergeableCandidate]] = []
    for source in set_a:
        best_target: MergeableCandidate | None = None
        best_similarity = 0.0
        for target in set_b:
            similarity = _pair_similarity(
                source,
                target,
                temporal_sigma_seconds=temporal_sigma_seconds,
                text_similarity_weight=text_similarity_weight,
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_target = target
        if best_target is not None and best_similarity >= min_similarity:
            edges.append((best_similarity, source, best_target))

    if not edges:
        return []

    # Keep only the strongest ``budget`` merges — the ToMe rule that you merge
    # the most redundant pairs first and leave everything else intact.
    edges.sort(key=lambda edge: (-edge[0], edge[1].id))
    edges = edges[:budget]

    # Multiple sources may fold into the same target, which is how a run of
    # near-identical frames collapses into one group rather than a chain of pairs.
    clusters: dict[str, list[MergeableCandidate]] = {}
    similarities: dict[str, list[float]] = {}
    targets_by_id: dict[str, MergeableCandidate] = {}
    for similarity, source, target in edges:
        clusters.setdefault(target.id, []).append(source)
        similarities.setdefault(target.id, []).append(similarity)
        targets_by_id[target.id] = target

    return [
        _build_group(
            members=[targets_by_id[target_id], *sources],
            similarity=sum(similarities[target_id]) / len(similarities[target_id]),
        )
        for target_id, sources in clusters.items()
    ]


def _build_group(
    members: list[MergeableCandidate],
    similarity: float,
) -> TailMergeGroup:
    # The representative carries the group's asset downstream, so it must be the
    # strongest member rather than an average: there is no meaningful way to
    # average two JPEGs pre-encoder, and a real frame stays inspectable.
    representative = max(
        members,
        key=lambda item: (item.normalized_score, item.relevance_score, -item.timestamp_seconds),
    )
    others = tuple(item.id for item in members if item.id != representative.id)

    return TailMergeGroup(
        representative_id=representative.id,
        merged_ids=others,
        modality=representative.modality,
        # Score-weighted centroid in time: the group points at where its
        # evidence actually concentrates, which keeps timestamp grounding honest.
        timestamp_seconds=_weighted_timestamp(members),
        text=_merged_text(members),
        relevance_score=representative.relevance_score,
        normalized_score=representative.normalized_score,
        similarity=similarity,
    )


def _weighted_timestamp(members: list[MergeableCandidate]) -> float:
    weights = [max(item.relevance_score, 0.0) for item in members]
    total = sum(weights)
    if total <= 0:
        return sum(item.timestamp_seconds for item in members) / len(members)
    return (
        sum(item.timestamp_seconds * weight for item, weight in zip(members, weights, strict=True))
        / total
    )


def _merged_text(members: list[MergeableCandidate]) -> str:
    seen: set[str] = set()
    parts: list[str] = []
    for item in sorted(members, key=lambda entry: entry.timestamp_seconds):
        text = " ".join(item.text.split())
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        parts.append(text)

    merged = " ".join(parts)
    if len(merged) <= MAX_MERGED_TEXT_CHARS:
        return merged
    return merged[: MAX_MERGED_TEXT_CHARS - 1].rstrip() + "…"


def _pair_similarity(
    left: MergeableCandidate,
    right: MergeableCandidate,
    *,
    temporal_sigma_seconds: float,
    text_similarity_weight: float,
) -> float:
    temporal = temporal_similarity(
        left.timestamp_seconds,
        right.timestamp_seconds,
        temporal_sigma_seconds,
    )
    if temporal < MIN_TEMPORAL_SIMILARITY:
        return 0.0

    lexical = text_similarity(left.text, right.text)
    temporal_weight = 1.0 - text_similarity_weight
    return (temporal_weight * temporal) + (text_similarity_weight * lexical)
