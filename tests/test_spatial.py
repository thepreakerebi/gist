from gist.vision.spatial import (
    build_query_spatial_mask,
    estimate_spatial_tokens,
    write_spatial_mask_overlay,
    write_spatial_mask_preview,
)


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
    assert mask.saliency_strategy == "center_object"


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


def test_query_spatial_mask_prefers_text_band_for_ocr_evidence() -> None:
    mask = build_query_spatial_mask(
        evidence_id="frame-ocr",
        query="what text is shown on screen",
        evidence_text="on-screen text near 4.00 seconds: GIST TOKEN SAVER",
        grid_size=5,
        retention_ratio=0.2,
    )

    retained_rows = {index // mask.grid_size for index in mask.retained_patch_indexes}

    assert mask.saliency_strategy == "text_band"
    assert 2 in retained_rows
    assert retained_rows <= {1, 2, 3}


def test_query_spatial_mask_prefers_center_for_visual_object_queries() -> None:
    mask = build_query_spatial_mask(
        evidence_id="frame-hand",
        query="show the robot hand",
        evidence_text="visual frame sampled at 5.00 seconds",
        grid_size=5,
        retention_ratio=0.2,
    )

    assert mask.saliency_strategy == "center_object"
    assert 12 in mask.retained_patch_indexes


def test_write_spatial_mask_preview_writes_svg(tmp_path) -> None:
    mask = build_query_spatial_mask(
        evidence_id="frame-ocr",
        query="what text is shown",
        evidence_text="on-screen text near 4.00 seconds: GIST",
        grid_size=3,
        retention_ratio=0.33,
    )

    path = write_spatial_mask_preview(mask, tmp_path / "mask.svg", cell_size=10)

    svg = path.read_text()
    assert path.exists()
    assert "<svg" in svg
    assert "text_band" in svg
    assert svg.count("<rect") == 10


def test_write_spatial_mask_overlay_writes_svg_with_image_href(tmp_path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake-image")
    mask = build_query_spatial_mask(
        evidence_id="frame-hand",
        query="show the robot hand",
        evidence_text="visual frame sampled at 4.00 seconds",
        grid_size=3,
        retention_ratio=0.33,
    )

    path = write_spatial_mask_overlay(mask, image_path, tmp_path / "overlay.svg", size=90)

    svg = path.read_text()
    assert path.exists()
    assert image_path.resolve().as_uri() in svg
    assert "retained spatial patches" in svg
    assert svg.count("<rect") == mask.retained_patches + 1
