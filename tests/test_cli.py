from pathlib import Path

from gist.cli import _attach_spatial_masks, _clear_previous_clips
from gist.core.presets import CompressionPreset
from gist.core.schemas import CompressionMetrics, CompressionResponse, Modality, SelectedCandidate
from gist.core.token_estimation import TokenEstimatorProfile


def test_clear_previous_clips_removes_stale_mp4_files_only(tmp_path: Path) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    stale = clips / "old.mp4"
    keep = clips / "notes.txt"
    stale.write_bytes(b"old")
    keep.write_text("keep")

    _clear_previous_clips(clips)

    assert stale.exists() is False
    assert keep.exists() is True


def test_attach_spatial_masks_writes_masks_for_visual_evidence(tmp_path: Path) -> None:
    compression = CompressionResponse(
        video_id="demo",
        query="show robot hand",
        preset=CompressionPreset.BALANCED,
        selected=[
            SelectedCandidate(
                id="visual-1",
                modality=Modality.VISUAL,
                timestamp_seconds=4,
                text="visual frame sampled at 4.00 seconds",
                asset_path=tmp_path / "frame.jpg",
                selection_rank=1,
                relevance_score=0.5,
                normalized_score=1,
                mmr_score=0.7,
                source_score_type="test",
                reason="selected",
            ),
            SelectedCandidate(
                id="audio-1",
                modality=Modality.AUDIO,
                timestamp_seconds=8,
                text="robot hand mentioned",
                selection_rank=2,
                relevance_score=0.4,
                normalized_score=0.8,
                mmr_score=0.6,
                source_score_type="test",
                reason="selected",
            ),
        ],
        metrics=CompressionMetrics(
            input_candidates=10,
            selected_candidates=2,
            visual_selected=1,
            audio_selected=1,
            estimated_candidate_reduction_ratio=0.2,
            estimated_candidate_reduction_percent=80,
            dropped_candidates=8,
            budget_preset_used=CompressionPreset.BALANCED,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )
    (tmp_path / "frame.jpg").write_bytes(b"fake-image")

    with_masks = _attach_spatial_masks(
        compression=compression,
        output_dir=tmp_path / "spatial",
        grid_size=4,
        retention_ratio=0.25,
    )

    assert with_masks.selected[0].spatial_mask_path is not None
    assert with_masks.selected[0].spatial_mask_path.exists()
    assert with_masks.selected[0].spatial_mask_preview_path is not None
    assert with_masks.selected[0].spatial_mask_preview_path.exists()
    assert with_masks.selected[0].spatial_mask_overlay_path is not None
    assert with_masks.selected[0].spatial_mask_overlay_path.exists()
    assert with_masks.selected[1].spatial_mask_path is None
    assert with_masks.selected[1].spatial_mask_preview_path is None
    assert with_masks.selected[1].spatial_mask_overlay_path is None
    assert with_masks.metrics.estimated_spatial_visual_tokens == 16
    assert with_masks.metrics.estimated_retained_spatial_visual_tokens == 4
    assert with_masks.metrics.estimated_spatial_token_reduction_percent == 75
