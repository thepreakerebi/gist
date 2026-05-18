from gist.vision.spatial import build_query_spatial_mask, estimate_spatial_tokens


def test_build_query_spatial_mask_returns_expected_shape() -> None:
    mask = build_query_spatial_mask(
        evidence_id="frame-1",
        query="show the person",
        grid_size=4,
        retention_ratio=0.25,
    )

    assert mask.total_patches == 16
    assert mask.retained_patches == 4
    assert mask.retained_patch_indexes == sorted(mask.retained_patch_indexes)


def test_query_spatial_mask_is_deterministic() -> None:
    first = build_query_spatial_mask("frame-1", "show the person", grid_size=4)
    second = build_query_spatial_mask("frame-1", "show the person", grid_size=4)

    assert first.retained_patch_indexes == second.retained_patch_indexes


def test_estimate_spatial_tokens_reports_reduction() -> None:
    baseline, retained, reduction = estimate_spatial_tokens(
        selected_visual_count=2,
        grid_size=10,
        retention_ratio=0.4,
    )

    assert baseline == 200
    assert retained == 80
    assert reduction == 60
