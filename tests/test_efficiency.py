from gist.eval.efficiency import (
    PROFILES,
    build_report,
    render_markdown,
    transformer_encoder_gflops,
)


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
