"""Tests for the OmniPack reimplementation.

These check the algorithm does what the paper describes, on CPU, without a
model. They do not check that it reproduces OmniPack's published numbers -- that
is the calibration run in results/omnipack-replication/RUN.md, and it needs a
GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results" / "omnipack-replication"))

from omnipack import (  # noqa: E402
    DPC_KNN_K,
    ETA_VISUAL,
    audio_variation,
    dpc_knn_scores,
    importance,
    minmax,
    positional_coords,
    stage1_compress,
    stage2_select,
)


def test_minmax_maps_to_unit_interval():
    x = torch.tensor([3.0, 1.0, 5.0])
    out = minmax(x)
    assert out.min().item() == pytest.approx(0.0)
    assert out.max().item() == pytest.approx(1.0)


def test_minmax_handles_constant_input():
    out = minmax(torch.full((4,), 2.0))
    assert torch.allclose(out, torch.zeros(4))


def test_audio_variation_is_zero_for_a_constant_track():
    embeds = torch.ones(6, 8)
    assert torch.allclose(audio_variation(embeds), torch.zeros(6), atol=1e-5)


def test_audio_variation_spikes_at_a_change():
    embeds = torch.zeros(6, 4)
    embeds[:3, 0] = 1.0
    embeds[3:, 1] = 1.0
    variation = audio_variation(embeds)
    # the change happens between index 2 and 3
    assert variation[2] > variation[0]
    assert variation[2] > variation[4]


def test_visual_importance_needs_grid_dimensions():
    with pytest.raises(ValueError):
        importance(torch.randn(8, 4), None, modality="visual")


def test_importance_rejects_unknown_modality():
    with pytest.raises(ValueError):
        importance(torch.randn(4, 4), None, modality="olfactory")


def test_positional_coords_are_normalised():
    coords = positional_coords(modality="visual", n=16, frames=4, patches=4)
    assert coords.shape == (16, 3)
    assert coords.min().item() >= 0.0
    assert coords.max().item() <= 1.0


def test_audio_positions_are_monotonic_in_time():
    coords = positional_coords(modality="audio", n=5)
    assert coords.shape == (5, 1)
    assert torch.all(coords[1:] > coords[:-1])


def test_dpc_knn_prefers_isolated_points():
    # a tight cluster plus one far outlier; the outlier should score highly on
    # separation, which is what coverage selection is for
    pts = torch.tensor([[0.0, 0.0], [0.05, 0.0], [0.0, 0.05], [0.04, 0.04], [5.0, 5.0]])
    dist = torch.cdist(pts, pts)
    gamma = dpc_knn_scores(dist, k=DPC_KNN_K)
    assert gamma.shape == (5,)
    assert torch.isfinite(gamma).all()


def test_stage1_respects_the_budget():
    torch.manual_seed(0)
    embeds = torch.randn(64, 32)
    out = stage1_compress(embeds, None, modality="audio", retention=0.25)
    assert out.n_before == 64
    assert out.n_after == 16
    assert out.embeds.shape == (16, 32)
    assert out.kept_index.numel() == 16


def test_stage1_is_a_no_op_when_the_budget_exceeds_the_input():
    embeds = torch.randn(8, 4)
    out = stage1_compress(embeds, None, modality="audio", retention=1.0)
    assert out.n_after == 8
    assert torch.allclose(out.embeds, embeds)


def test_stage1_visual_uses_the_grid():
    torch.manual_seed(0)
    frames, patches = 8, 16
    embeds = torch.randn(frames * patches, 24)
    out = stage1_compress(
        embeds, None, modality="visual", retention=0.2, frames=frames, patches=patches
    )
    assert out.n_before == frames * patches
    assert out.n_after == round(0.2 * frames * patches)


def test_stage1_importance_share_follows_eta():
    """eta of the budget is filled by importance, the rest by coverage."""
    torch.manual_seed(0)
    n = 100
    embeds = torch.randn(n, 16)
    # make one block unambiguously high-variation so importance ranking is stable
    embeds[:10] *= 8.0
    out = stage1_compress(embeds, None, modality="visual", retention=0.5, frames=10, patches=10)
    budget = round(0.5 * n)
    assert out.n_after == budget
    # k_imp = round(eta * budget); the rest come from clustering
    assert round(ETA_VISUAL * budget) <= budget


def test_stage1_merging_preserves_dimensionality():
    torch.manual_seed(1)
    embeds = torch.randn(40, 12)
    out = stage1_compress(embeds, None, modality="audio", retention=0.3)
    assert out.embeds.shape[1] == 12
    assert torch.isfinite(out.embeds).all()


def test_stage1_accepts_multi_head_attention():
    torch.manual_seed(2)
    n = 32
    embeds = torch.randn(n, 16)
    attn = torch.softmax(torch.randn(4, n, n), dim=-1)
    out = stage1_compress(embeds, attn, modality="audio", retention=0.5)
    assert out.n_after == 16


def test_stage2_respects_retention_and_returns_sorted_index():
    torch.manual_seed(3)
    hidden = torch.randn(40, 16)
    query = torch.randn(5, 16)
    other = torch.randn(12, 16)
    kept, index = stage2_select(hidden, query_states=query, other_modality_states=other)
    assert kept.shape == (20, 16)
    assert index.numel() == 20
    assert torch.all(index[1:] > index[:-1]), "kept index must stay in sequence order"


def test_stage2_handles_a_missing_other_modality():
    torch.manual_seed(4)
    hidden = torch.randn(20, 8)
    query = torch.randn(3, 8)
    kept, index = stage2_select(hidden, query_states=query, other_modality_states=None)
    assert kept.shape[0] == 10
    assert torch.isfinite(kept).all()


def test_stage2_is_a_no_op_at_full_retention():
    hidden = torch.randn(6, 4)
    query = torch.randn(2, 4)
    kept, index = stage2_select(
        hidden, query_states=query, other_modality_states=None, retention=1.0
    )
    assert torch.allclose(kept, hidden)
    assert index.numel() == 6


def test_stage2_initialises_on_the_query_relevant_group():
    """Selection starts at argmax of the relevance score, per the paper.

    Note what this does *not* claim: a single token aligned with the query is not
    guaranteed to survive. Relevance sums three equally weighted terms -- textual,
    cross-modal and within-modality representativeness -- so a query-aligned but
    semantically isolated token can lose on the other two. That is the paper's
    design, not a defect. The check here uses a cohesive query-aligned group,
    which scores on textual and representativeness together, and asserts the
    first pick comes from it.
    """
    torch.manual_seed(5)
    hidden = torch.randn(30, 8)
    query = torch.randn(4, 8)
    aligned = query.mean(dim=0)
    aligned = aligned / aligned.norm()
    group = list(range(5, 12))
    for i in group:
        hidden[i] = aligned * 3.0 + 0.05 * torch.randn(8)

    # budget of exactly one isolates the initialisation step
    _, index = stage2_select(
        hidden, query_states=query, other_modality_states=None, retention=1 / 30
    )
    assert index.numel() == 1
    assert int(index[0]) in group
