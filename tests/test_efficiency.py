import pytest

from gist.eval.efficiency import (
    PROFILES,
    build_measured_report,
    build_report,
    render_markdown,
    render_measured_markdown,
    transformer_encoder_gflops,
)
from gist.eval.profiling import StageMeasurement


def test_transformer_flops_scale_with_depth_and_tokens():
    base = transformer_encoder_gflops(num_tokens=256, dim=768, depth=12)
    deeper = transformer_encoder_gflops(num_tokens=256, dim=768, depth=24)
    more_tokens = transformer_encoder_gflops(num_tokens=512, dim=768, depth=12)
    assert deeper == 2 * base  # linear in depth
    assert more_tokens > 2 * base  # super-linear (quadratic attention term)
    assert base > 0


def test_gist_saving_is_count_driven_and_large():
    profile = PROFILES["qwen2.5-omni-7b"]
    report = build_report(
        {
            "full baseline": (64, 120),
            "uniform-K": (8, 8),
            "gist-K": (8, 4),
        },
        profile,
        baseline="full baseline",
    )
    costs = {c.condition: c for c in report.conditions}
    assert costs["full baseline"].system_gflops_saving_pct == 0.0
    # Encoding 8 of 64 frames + 4 of 120 windows -> large system saving.
    assert costs["gist-K"].system_gflops_saving_pct > 80.0
    assert costs["gist-K"].system_gflops < costs["full baseline"].system_gflops
    # downstream tokens also collapse
    assert costs["gist-K"].downstream_tokens < costs["full baseline"].downstream_tokens


def test_report_renders_markdown_table():
    report = build_report({"full baseline": (64, 120), "gist-K": (8, 4)},
                          PROFILES["qwen2.5-omni-7b"], baseline="full baseline")
    md = render_markdown(report)
    assert "System GFLOPs" in md
    assert "gist-K" in md
    assert "GFLOPs saving" in md


# --- measured reports -------------------------------------------------------


def _row(condition, wall, activation=None, reserved=None, oom=False, stage="answer"):
    return StageMeasurement(
        stage=stage,
        wall_seconds=wall,
        activation_gb=activation,
        peak_reserved_gb=reserved,
        oom=oom,
        error="CUDA out of memory" if oom else None,
        extra={"condition": condition},
    )


def test_measured_report_compares_against_the_named_baseline():
    stages = [
        # first row of each condition is warmup and is dropped
        _row("full", 9.0, 4.0), _row("full", 4.0, 4.0), _row("full", 4.0, 4.0),
        _row("gist", 9.0, 1.0), _row("gist", 2.0, 1.0), _row("gist", 2.0, 1.0),
    ]
    report = build_measured_report(stages, baseline="full")
    by_name = {c.condition: c for c in report.conditions}
    assert by_name["full"].wall_seconds_median == 4.0
    assert by_name["gist"].wall_seconds_median == 2.0
    assert by_name["gist"].speedup_vs_baseline == 2.0
    assert by_name["gist"].activation_saving_pct == 75.0
    assert report.warmup_dropped == 1


def test_warmup_row_would_otherwise_swamp_a_short_run():
    """The first CUDA call pays for context creation; keeping it inverts results."""
    stages = [_row("full", 1.0), _row("full", 1.0), _row("gist", 30.0), _row("gist", 1.0)]
    kept = build_measured_report(stages, baseline="full", drop_warmup=0)
    dropped = build_measured_report(stages, baseline="full", drop_warmup=1)
    gist_kept = {c.condition: c for c in kept.conditions}["gist"]
    gist_dropped = {c.condition: c for c in dropped.conditions}["gist"]
    assert gist_kept.wall_seconds_median == 15.5  # warmup dominates
    assert gist_dropped.wall_seconds_median == 1.0


def test_oom_rows_are_counted_but_never_averaged_into_timings():
    stages = [
        _row("full", 1.0, 2.0), _row("full", 1.0, 2.0), _row("full", 1.0, 2.0),
        _row("dense", 0.4, oom=True), _row("dense", 0.4, oom=True),
    ]
    report = build_measured_report(stages, baseline="full")
    dense = {c.condition: c for c in report.conditions}["dense"]
    assert dense.n == 2
    assert dense.n_oom == 2
    assert dense.n_used == 0
    assert dense.wall_seconds_median is None  # a failed forward is not a latency
    assert dense.speedup_vs_baseline is None


def test_measured_markdown_calls_out_an_oom_condition_as_a_result():
    stages = [
        _row("full", 1.0, 2.0), _row("full", 1.0, 2.0),
        _row("dense", 0.4, oom=True), _row("dense", 0.4, oom=True),
    ]
    md = render_measured_markdown(build_measured_report(stages, baseline="full"))
    assert "Out of memory" in md
    assert "dense" in md
    assert "did not fit" in md


def test_measured_report_only_aggregates_the_requested_stage():
    stages = [
        _row("gist", 5.0, stage="select"), _row("gist", 5.0, stage="select"),
        _row("gist", 1.0, stage="answer"), _row("gist", 1.0, stage="answer"),
    ]
    report = build_measured_report(stages, baseline="gist", stage="answer", drop_warmup=0)
    assert {c.condition: c for c in report.conditions}["gist"].n == 2


def test_unknown_baseline_is_rejected_rather_than_silently_skipped():
    with pytest.raises(ValueError, match="baseline"):
        build_measured_report([_row("gist", 1.0)], baseline="full")


def test_a_zero_baseline_median_still_produces_a_speedup_column():
    """Regression: guarding on truthiness dropped a legitimate median of 0.0."""
    stages = [
        _row("full", 0.0), _row("full", 0.0), _row("full", 0.0),
        _row("gist", 0.0), _row("gist", 2.0), _row("gist", 2.0),
    ]
    report = build_measured_report(stages, baseline="full")
    by_name = {c.condition: c for c in report.conditions}
    assert by_name["full"].wall_seconds_median == 0.0
    # gist has a non-zero divisor, so the ratio is computable and is reported
    assert by_name["gist"].speedup_vs_baseline == 0.0
    # full divides by its own zero median, which stays undefined rather than raising
    assert by_name["full"].speedup_vs_baseline is None


def test_n_counts_every_row_while_n_used_counts_what_fed_the_statistics():
    stages = [_row("gist", 1.0), _row("gist", 2.0), _row("gist", 2.0)]
    gist = build_measured_report(stages, baseline="gist").conditions[0]
    assert gist.n == 3
    assert gist.n_used == 2
