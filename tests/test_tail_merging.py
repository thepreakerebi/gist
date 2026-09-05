"""Regression tests for ToMe-style tail merging (Objective 2, component 3).

Covers the merge policy in isolation and its integration through the compressor,
so the component can be independently enabled and measured.
"""

from dataclasses import dataclass

import pytest

from gist.core.compressor import GistCompressor
from gist.core.presets import PRESETS, CompressionPreset
from gist.core.schemas import Candidate, CompressionRequest, Modality
from gist.core.tail_merging import merge_tail


@dataclass(frozen=True, slots=True)
class FakeCandidate:
    id: str
    modality: Modality
    timestamp_seconds: float
    text: str
    relevance_score: float
    normalized_score: float


def _tail(
    count: int,
    *,
    spacing: float = 1.0,
    text: str = "robot arm on the workbench",
    modality: Modality = Modality.VISUAL,
    start: float = 0.0,
) -> list[FakeCandidate]:
    return [
        FakeCandidate(
            id=f"{modality.value}-{index}",
            modality=modality,
            timestamp_seconds=start + (index * spacing),
            text=text,
            relevance_score=0.4,
            normalized_score=0.4,
        )
        for index in range(count)
    ]


def test_merges_temporally_adjacent_near_duplicates() -> None:
    groups = merge_tail(_tail(6, spacing=1.0), temporal_sigma_seconds=14.0)

    assert groups, "tightly clustered identical candidates must merge"
    assert all(group.size >= 2 for group in groups)


def test_does_not_merge_temporally_distant_candidates() -> None:
    # Spacing far beyond the kernel width: these are different moments, and
    # merging them would fabricate evidence that never co-occurred.
    groups = merge_tail(_tail(6, spacing=600.0), temporal_sigma_seconds=14.0)

    assert groups == []


def test_never_merges_across_modalities() -> None:
    # Simultaneous frame + audio window: temporally identical, but not redundant.
    tail = [
        *_tail(3, spacing=1.0, modality=Modality.VISUAL),
        *_tail(3, spacing=1.0, modality=Modality.AUDIO),
    ]

    for group in merge_tail(tail, temporal_sigma_seconds=14.0):
        members = {group.representative_id, *group.merged_ids}
        prefixes = {member.split("-")[0] for member in members}
        assert len(prefixes) == 1, f"cross-modal merge: {members}"


def test_respects_max_groups_ceiling() -> None:
    tail = [
        *_tail(6, spacing=1.0, start=0.0),
        *_tail(6, spacing=1.0, start=500.0),
        *_tail(6, spacing=1.0, start=1000.0),
    ]

    assert len(merge_tail(tail, temporal_sigma_seconds=14.0, max_groups=2)) <= 2


def test_representative_is_the_strongest_member() -> None:
    tail = list(_tail(4, spacing=1.0))
    strongest = FakeCandidate(
        id="visual-strong",
        modality=Modality.VISUAL,
        timestamp_seconds=1.5,
        text="robot arm on the workbench",
        relevance_score=0.9,
        normalized_score=2.5,
    )
    groups = merge_tail([*tail, strongest], temporal_sigma_seconds=14.0)

    owning = [g for g in groups if "visual-strong" in {g.representative_id, *g.merged_ids}]
    assert owning, "the strong candidate should participate in a merge"
    assert all(group.representative_id == "visual-strong" for group in owning)


def test_merged_timestamp_stays_inside_the_group_span() -> None:
    tail = _tail(6, spacing=1.0)
    lo = min(item.timestamp_seconds for item in tail)
    hi = max(item.timestamp_seconds for item in tail)

    for group in merge_tail(tail, temporal_sigma_seconds=14.0):
        assert lo <= group.timestamp_seconds <= hi


@pytest.mark.parametrize("size", [0, 1])
def test_degenerate_tails_are_a_no_op(size: int) -> None:
    assert merge_tail(_tail(size), temporal_sigma_seconds=14.0) == []


def test_disabled_ratio_is_a_no_op() -> None:
    assert merge_tail(_tail(8, spacing=1.0), temporal_sigma_seconds=14.0, merge_ratio=0.0) == []


def _request(**overrides: object) -> CompressionRequest:
    # A dense cluster of near-identical frames: mostly tail, highly redundant.
    visual = [
        Candidate(
            id=f"frame-{index}",
            timestamp_seconds=float(index),
            text="robot arm on the workbench",
            saliency_score=0.9 if index == 0 else 0.30,
        )
        for index in range(40)
    ]
    payload: dict[str, object] = {
        "video_id": "video-1",
        "query": "what is on the workbench",
        "duration_seconds": 120.0,
        "preset": CompressionPreset.AGGRESSIVE,
        "visual_candidates": visual,
    }
    payload.update(overrides)
    return CompressionRequest(**payload)  # type: ignore[arg-type]


def test_compressor_is_unchanged_when_tail_merging_is_off() -> None:
    response = GistCompressor().compress(_request())

    assert response.metrics.tail_merged_groups == 0
    assert response.metrics.tail_merged_candidates == 0
    assert all(item.merged_from_count == 0 for item in response.selected)


def test_compressor_emits_auditable_merge_provenance() -> None:
    response = GistCompressor().compress(_request(tail_merging=True))

    merged = [item for item in response.selected if item.merged_from_count]
    assert merged, "a dense redundant pool must produce at least one merge"
    for item in merged:
        assert item.merged_from_count == len(item.merged_from_ids)
        assert item.merge_similarity is not None
        assert "tail-merged" in item.reason
    assert response.metrics.tail_merged_groups == len(merged)


def test_tail_merging_never_exceeds_its_ceiling() -> None:
    budget = PRESETS[CompressionPreset.AGGRESSIVE].max_items
    response = GistCompressor().compress(_request(tail_merging=True, tail_merge_max_items=2))

    assert response.metrics.tail_merged_groups <= 2
    assert len(response.selected) <= budget + 2


def test_tail_merging_leaves_the_hard_kept_head_intact() -> None:
    baseline = GistCompressor().compress(_request())
    merged = GistCompressor().compress(_request(tail_merging=True))

    head = {item.id for item in baseline.selected}
    assert head <= {item.id for item in merged.selected}, "merging must not evict the MMR head"


# --- OCR plausibility gate (library ingestion) -------------------------------


@pytest.mark.parametrize(
    "noise",
    [
        '| _"! el ae Ey " Se (a* 268 Coal bss se a =e \'o~ - 7 4 ey ae) a #',
        '~~ Hep (me ae --- A --) " =~ = -" - & ae 2',
        "_-- + ys all <= alia au se = --- -",
        # Tesseract noise is *alphabetic*, which is why an "is it letters?" gate
        # let all of these through; what separates them is token shape.
        "Te: ast sexy ss >",
        "se cle tall ol Seay 7 a aoe a ae",
        "z Te dou wa tpt Pen = a=",
        "Ppa tee ase i Beer a i Ele oe",
        "Kt ie & a cater ead - -- Nae",
        "a",
        "",
    ],
)
def test_ocr_noise_is_discarded(noise: str) -> None:
    """Tesseract on a frame with no text returns confident-looking garbage.

    Left in, it shows up as evidence text in the UI and pollutes the lexical
    fallback the selector uses when an encoder is unavailable.
    """

    from gist.library.ingest import _plausible_ocr_text

    assert _plausible_ocr_text(noise) is None


@pytest.mark.parametrize(
    "real",
    [
        "Behavior-Based Robotics",
        "Sense Think Act Control Scheme",
        "Chapter 1: Why Bio-Inspired Motor Control?",
        "Further reading materials",
        # Short function words are legitimate in real titles and must not count
        # against the short-token share.
        "Legged Locomotion in Nature",
        "How to make equations of motion",
        "Conceptual Models of Legged Locomotion",
    ],
)
def test_real_slide_text_survives(real: str) -> None:
    from gist.library.ingest import _plausible_ocr_text

    assert _plausible_ocr_text(real) == real
