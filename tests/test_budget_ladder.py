"""Regression tests for the three-stage confidence-routed budget ladder.

The proposal's Figure 8 specifies escalation "bounded to three stages so that
worst-case cost stays finite while amortized cost stays well below it". These
tests pin both halves of that claim: the ladder can reach stage 3, and it cannot
go past it.
"""

from gist.core.compressor import GistCompressor
from gist.core.presets import PRESETS, CompressionPreset
from gist.core.schemas import Candidate, CompressionRequest


def _request(**overrides: object) -> CompressionRequest:
    # Uniformly weak, single-modality candidates: nothing clears the relevance
    # floor, so every escalation trigger stays live and the ladder runs to its end.
    payload: dict[str, object] = {
        "video_id": "video-1",
        "query": "what does the presenter say about scheduling",
        "duration_seconds": 600.0,
        "adaptive_budget": True,
        "preset": CompressionPreset.CONSERVATIVE,
        "visual_candidates": [
            Candidate(
                id=f"frame-{index}",
                timestamp_seconds=float(index * 20),
                text="unrelated filler content",
                saliency_score=0.01,
            )
            for index in range(30)
        ],
    }
    payload.update(overrides)
    return CompressionRequest(**payload)  # type: ignore[arg-type]


def test_fixed_budget_reports_a_single_stage() -> None:
    response = GistCompressor().compress(_request(adaptive_budget=False))

    assert response.metrics.budget_mode == "fixed"
    assert response.metrics.budget_stages_used == 1
    assert response.metrics.budget_stage_reasons == []


def test_ladder_escalates_to_three_stages_when_confidence_stays_low() -> None:
    response = GistCompressor().compress(_request())

    assert response.metrics.budget_mode == "adaptive"
    assert response.metrics.budget_stages_used == 3
    assert response.metrics.budget_preset_used == CompressionPreset.CONSERVATIVE
    assert response.metrics.budget_expanded is True


def test_escalation_is_bounded_at_three_stages() -> None:
    response = GistCompressor().compress(_request())

    # The bound is what keeps worst-case cost finite; without it a pathological
    # pool could walk the ladder indefinitely.
    assert response.metrics.budget_stages_used <= 3
    assert len(response.metrics.budget_stage_reasons) <= 2


def test_stage_reasons_are_recorded_in_escalation_order() -> None:
    reasons = GistCompressor().compress(_request()).metrics.budget_stage_reasons

    assert [reason.split(":")[0] for reason in reasons] == ["stage 2", "stage 3"]
    assert all(reason.split(": ", 1)[1] for reason in reasons)


def test_expansion_reason_stays_the_bare_trigger() -> None:
    # Existing consumers read expansion_reason directly; the "stage N" prefix
    # belongs only to the staged trail.
    response = GistCompressor().compress(_request())

    assert response.metrics.expansion_reason is not None
    assert not response.metrics.expansion_reason.startswith("stage ")


def test_requested_preset_is_an_upper_bound_on_escalation() -> None:
    response = GistCompressor().compress(_request(preset=CompressionPreset.BALANCED))

    assert response.metrics.budget_preset_used == CompressionPreset.BALANCED
    assert response.metrics.budget_stages_used <= 2
    assert len(response.selected) <= PRESETS[CompressionPreset.BALANCED].max_items


def test_confident_selection_stops_at_stage_one() -> None:
    # A strong match in *both* modalities: the amortized case the ladder exists
    # for. Both are needed — a single-modality selection trips the existing
    # "only one modality" escalation trigger no matter how confident it is.
    response = GistCompressor().compress(
        _request(
            visual_candidates=[
                Candidate(
                    id="frame-hit",
                    timestamp_seconds=10.0,
                    text="the scheduling policy slide",
                    saliency_score=0.98,
                ),
                Candidate(
                    id="frame-miss",
                    timestamp_seconds=400.0,
                    text="unrelated filler content",
                    saliency_score=0.02,
                ),
            ],
            audio_candidates=[
                Candidate(
                    id="audio-hit",
                    timestamp_seconds=12.0,
                    text="here is the scheduling policy we use",
                    saliency_score=0.95,
                ),
                # Two audio windows, not one: the existing "underrepresented
                # source audio" trigger fires whenever an audio-anchored visual
                # is selected alongside fewer than two audio candidates.
                Candidate(
                    id="audio-hit-2",
                    timestamp_seconds=30.0,
                    text="the scheduling policy applies to every team",
                    saliency_score=0.92,
                ),
            ],
        )
    )

    assert response.metrics.budget_stages_used == 1
    assert response.metrics.budget_preset_used == CompressionPreset.AGGRESSIVE
    assert response.metrics.budget_expanded is False
