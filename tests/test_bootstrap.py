"""Regression tests for the bootstrap intervals and the frozen split.

The statistics here decide what the project is allowed to claim, so the
properties that matter are correctness ones — a paired interval must actually
be paired, a frozen split must actually stay frozen — rather than snapshots of
particular numbers.
"""

import json
from pathlib import Path

import pytest

from gist.eval.bootstrap import (
    compare,
    outcomes_from_predictions,
    paired_difference,
    proportion_interval,
    render,
    standard_error,
)
from gist.eval.splits import build_manifest, source_video

FAST = {"resamples": 400, "seed": 0}


# ------------------------------------------------------------- intervals ----


def test_interval_brackets_the_point_estimate() -> None:
    outcomes = [True] * 27 + [False] * 12
    interval = proportion_interval(outcomes, **FAST)

    assert interval.low <= interval.point <= interval.high
    assert interval.point == pytest.approx(27 / 39)


def test_unanimous_outcomes_give_a_degenerate_interval() -> None:
    # Every resample of an all-pass vector is all-pass; an interval with any
    # width here would mean the resampling is not actually resampling.
    interval = proportion_interval([True] * 20, **FAST)

    assert (interval.low, interval.point, interval.high) == (1.0, 1.0, 1.0)


def test_larger_samples_give_tighter_intervals() -> None:
    small = proportion_interval([True, False] * 10, **FAST)
    large = proportion_interval([True, False] * 200, **FAST)

    assert (large.high - large.low) < (small.high - small.low)


def test_paired_difference_is_zero_for_identical_conditions() -> None:
    outcomes = [True, False, True, True, False] * 4
    difference = paired_difference(outcomes, list(outcomes), **FAST)

    assert (difference.low, difference.point, difference.high) == (0.0, 0.0, 0.0)


def test_pairing_is_tighter_than_treating_conditions_independently() -> None:
    """The point of pairing: shared case difficulty must not inflate the interval.

    Two conditions that differ on exactly one case should produce a narrow
    interval on the difference, even though each condition's own interval is
    wide.
    """

    a = [True] * 20 + [False] * 20
    b = [True] * 19 + [False] * 21

    difference = paired_difference(a, b, **FAST)
    own = proportion_interval(a, **FAST)

    assert (difference.high - difference.low) < (own.high - own.low)


def test_standard_error_matches_the_binomial_formula() -> None:
    outcomes = [True] * 5 + [False] * 13  # 5/18, the fp16 full-context baseline
    assert standard_error(outcomes) == pytest.approx(0.1057, abs=1e-3)


def test_non_inferiority_uses_the_lower_bound_against_the_margin() -> None:
    # Identical conditions: difference is exactly zero, so it clears any margin.
    same = [True, False] * 9
    assert compare("a", "b", same, list(same), **FAST).non_inferior

    # Catastrophically worse: the lower bound is far below one standard error.
    worse = compare("a", "b", [False] * 18, [True] * 9 + [False] * 9, **FAST)
    assert not worse.non_inferior


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        paired_difference([True, False], [True], **FAST)


def test_non_inferiority_column_is_opt_in() -> None:
    """It is a claim about one pre-registered comparison, not about every arm."""

    outcomes = [True, False] * 9
    comparisons = [compare("gist", "full", outcomes, list(outcomes), **FAST)]

    assert "Non-inferior" not in render(comparisons, title="t")
    assert "Non-inferior" in render(comparisons, title="t", non_inferiority=True)


def test_predictions_are_scored_against_gold(tmp_path: Path) -> None:
    rows = [
        {"qid": "1", "gold": "A", "full": "A", "gist": "B"},
        {"qid": "2", "gold": "B", "full": "C", "gist": "b"},  # case-insensitive
        {"qid": "3", "gold": "C", "full": "C", "gist": "C"},
    ]
    path = tmp_path / "preds.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))

    outcomes = outcomes_from_predictions(path)

    assert outcomes["full"] == [True, False, True]
    assert outcomes["gist"] == [False, True, True]


# ----------------------------------------------------------------- split ----


def test_split_is_grouped_by_source_video() -> None:
    """No recording may appear on both sides, or the split leaks."""

    manifest = build_manifest()
    assert set(manifest["held_out"]["videos"]).isdisjoint(manifest["dev"]["videos"])


def test_split_partitions_every_case_exactly_once() -> None:
    manifest = build_manifest()
    held = set(manifest["held_out"]["case_ids"])
    dev = set(manifest["dev"]["case_ids"])

    assert held.isdisjoint(dev)
    assert len(held) + len(dev) == manifest["held_out"]["cases"] + manifest["dev"]["cases"]


def test_both_sides_cover_every_query_category() -> None:
    manifest = build_manifest()
    assert set(manifest["held_out"]["categories"]) == set(manifest["dev"]["categories"])


def test_held_out_carries_real_weight_in_speech() -> None:
    """Cross-modal arbitration is the central claim; the held-out split has to
    be able to test it, not just touch the category."""

    manifest = build_manifest()
    assert manifest["held_out"]["categories"]["speech_semantic"] >= 5


def test_split_is_deterministic() -> None:
    assert build_manifest()["held_out_fingerprint"] == build_manifest()["held_out_fingerprint"]


def test_frozen_manifest_matches_what_the_code_produces() -> None:
    """Catches a split that was re-cut without being deliberately re-frozen."""

    frozen = json.loads(Path("data/eval/splits/held-out.json").read_text())
    assert frozen["held_out_fingerprint"] == build_manifest()["held_out_fingerprint"]


def test_source_video_recovers_the_recording() -> None:
    case = {"compression_path": ".gist/runs/paul-graham-y-combinator/some-query/compression.json"}
    assert source_video(case) == "paul-graham-y-combinator"
    assert source_video({}) == "unknown"
