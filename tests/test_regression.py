import json
from pathlib import Path

import pytest

from gist.core.presets import CompressionPreset
from gist.core.schemas import (
    CompressionMetrics,
    CompressionResponse,
    Modality,
    SelectedCandidate,
)
from gist.core.token_estimation import TokenEstimatorProfile
from gist.eval.regression import (
    RegressionCase,
    TimeRange,
    evaluate_case,
    render_regression_markdown,
    run_regression_cases,
)


def test_evaluate_case_passes_when_evidence_ranges_overlap_expected_ranges(
    tmp_path: Path,
) -> None:
    compression_path = _write_compression(
        tmp_path,
        selected=[
            _item("a-1", timestamp=385, clip_start=370, clip_end=400),
            _item("a-2", timestamp=465, clip_start=450, clip_end=480),
        ],
        answer="Builders use AI for research and code generation.",
        token_reduction=99.5,
    )
    case = RegressionCase(
        id="yc-tokenmaxxing",
        compression_path=compression_path,
        expected_evidence_ranges=[
            TimeRange(start_seconds=378, end_seconds=392),
            TimeRange(start_seconds=458, end_seconds=472),
        ],
        min_timestamp_hit_rate=1.0,
        min_token_reduction_percent=95.0,
        max_selected_evidence=3,
        required_answer_terms=["research", "code"],
    )

    result = evaluate_case(case)

    assert result.passed is True
    assert result.timestamp_hit_rate == 1.0
    assert result.selected_evidence == 2
    assert result.audio_evidence == 2


def test_evaluate_case_reports_failures_for_missing_timestamp_and_answer_terms(
    tmp_path: Path,
) -> None:
    compression_path = _write_compression(
        tmp_path,
        selected=[_item("a-1", timestamp=100, clip_start=90, clip_end=110)],
        answer="Builders use AI for research.",
        token_reduction=80.0,
    )
    case = RegressionCase(
        id="bad-case",
        compression_path=compression_path,
        expected_evidence_ranges=[TimeRange(start_seconds=370, end_seconds=400)],
        min_timestamp_hit_rate=1.0,
        min_token_reduction_percent=95.0,
        required_answer_terms=["code"],
    )

    result = evaluate_case(case)

    assert result.passed is False
    assert result.timestamp_hit_rate == 0.0
    assert any("timestamp hit rate" in failure for failure in result.failures)
    assert any("token reduction" in failure for failure in result.failures)
    assert any("answer missing required terms" in failure for failure in result.failures)


def test_run_regression_cases_and_markdown_summary(tmp_path: Path) -> None:
    compression_path = _write_compression(
        tmp_path,
        selected=[_item("a-1", timestamp=10, clip_start=0, clip_end=20)],
        answer="Pricing is explained.",
        token_reduction=99.0,
    )
    report = run_regression_cases(
        [
            RegressionCase(
                id="pricing",
                compression_path=compression_path,
                expected_evidence_ranges=[TimeRange(start_seconds=5, end_seconds=15)],
            )
        ]
    )

    markdown = render_regression_markdown(report)

    assert report.passed is True
    assert "| pricing | pass |" in markdown


def test_evaluate_case_enforces_visual_evidence_floor(tmp_path: Path) -> None:
    compression_path = _write_compression(
        tmp_path,
        selected=[_item("a-1", timestamp=10, clip_start=0, clip_end=20)],
        answer="A robot hand appears on screen.",
        token_reduction=99.0,
    )
    case = RegressionCase(
        id="visual-case",
        compression_path=compression_path,
        expected_evidence_ranges=[TimeRange(start_seconds=5, end_seconds=15)],
        min_visual_evidence=1,
    )

    result = evaluate_case(case)

    assert result.passed is False
    assert any("visual evidence" in failure for failure in result.failures)


def test_evaluate_case_counts_video_grounded_audio_as_visual_evidence(
    tmp_path: Path,
) -> None:
    compression_path = _write_compression(
        tmp_path,
        selected=[
            _item(
                "video:audio:11+video:visual:48",
                timestamp=345,
                clip_start=330,
                clip_end=360,
            )
        ],
        answer="She asks why he will not admit he is freaked out by her robot hand.",
        token_reduction=99.0,
    )
    case = RegressionCase(
        id="mixed-av-video-grounded",
        compression_path=compression_path,
        expected_evidence_ranges=[TimeRange(start_seconds=330, end_seconds=360)],
        min_visual_evidence=1,
        min_audio_evidence=1,
    )

    result = evaluate_case(case)

    assert result.passed is True
    assert result.visual_evidence == 1
    assert result.audio_evidence == 1


def test_evaluate_case_enforces_spatial_artifacts_for_visual_evidence(
    tmp_path: Path,
) -> None:
    mask_path = tmp_path / "mask.json"
    preview_path = tmp_path / "mask.svg"
    overlay_path = tmp_path / "overlay.svg"
    for path in [mask_path, preview_path, overlay_path]:
        path.write_text("artifact")
    compression_path = _write_compression(
        tmp_path,
        selected=[
            _item(
                "v-1",
                timestamp=10,
                clip_start=0,
                clip_end=20,
                modality=Modality.VISUAL,
                spatial_mask_path=mask_path,
                spatial_mask_preview_path=preview_path,
                spatial_mask_overlay_path=overlay_path,
            )
        ],
        answer="A robot hand appears on screen.",
        token_reduction=99.0,
        spatial_reduction=75.0,
    )
    case = RegressionCase(
        id="spatial-case",
        compression_path=compression_path,
        expected_evidence_ranges=[TimeRange(start_seconds=5, end_seconds=15)],
        min_visual_evidence=1,
        min_spatial_token_reduction_percent=70,
        require_spatial_masks=True,
        require_spatial_previews=True,
        require_spatial_overlays=True,
    )

    result = evaluate_case(case)

    assert result.passed is True
    assert result.spatial_token_reduction_percent == 75


def test_evaluate_case_reports_missing_spatial_artifacts(tmp_path: Path) -> None:
    compression_path = _write_compression(
        tmp_path,
        selected=[
            _item(
                "v-1",
                timestamp=10,
                clip_start=0,
                clip_end=20,
                modality=Modality.VISUAL,
            )
        ],
        answer="A robot hand appears on screen.",
        token_reduction=99.0,
        spatial_reduction=0.0,
    )
    case = RegressionCase(
        id="bad-spatial-case",
        compression_path=compression_path,
        expected_evidence_ranges=[TimeRange(start_seconds=5, end_seconds=15)],
        min_spatial_token_reduction_percent=50,
        require_spatial_masks=True,
        require_spatial_previews=True,
        require_spatial_overlays=True,
    )

    result = evaluate_case(case)

    assert result.passed is False
    assert any("spatial token reduction" in failure for failure in result.failures)
    assert any("spatial mask missing" in failure for failure in result.failures)
    assert any("spatial preview missing" in failure for failure in result.failures)
    assert any("spatial overlay missing" in failure for failure in result.failures)


def test_time_range_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance_seconds"):
        TimeRange(start_seconds=0, end_seconds=1).overlaps(
            TimeRange(start_seconds=2, end_seconds=3),
            tolerance_seconds=-1,
        )


def _write_compression(
    tmp_path: Path,
    selected: list[SelectedCandidate],
    answer: str,
    token_reduction: float,
    spatial_reduction: float = 0.0,
) -> Path:
    compression = CompressionResponse(
        video_id="video",
        query="How do builders use AI?",
        answer=answer,
        preset=CompressionPreset.BALANCED,
        selected=selected,
        metrics=CompressionMetrics(
            input_candidates=100,
            selected_candidates=len(selected),
            visual_selected=sum(item.modality == Modality.VISUAL for item in selected),
            audio_selected=sum(item.modality == Modality.AUDIO for item in selected),
            estimated_candidate_reduction_ratio=len(selected) / 100,
            estimated_candidate_reduction_percent=100 - len(selected),
            dropped_candidates=100 - len(selected),
            budget_preset_used=CompressionPreset.BALANCED,
            estimated_baseline_tokens=3200,
            estimated_compressed_tokens=32 * len(selected),
            estimated_saved_tokens=3200 - (32 * len(selected)),
            estimated_token_reduction_ratio=1 - (token_reduction / 100),
            estimated_token_reduction_percent=token_reduction,
            estimated_spatial_token_reduction_percent=spatial_reduction,
            token_estimator=TokenEstimatorProfile.GENERIC,
        ),
    )
    path = tmp_path / "compression.json"
    path.write_text(json.dumps({"compression": compression.model_dump(mode="json")}))
    return path


def _item(
    id_: str,
    timestamp: float,
    clip_start: float,
    clip_end: float,
    modality: Modality = Modality.AUDIO,
    spatial_mask_path: Path | None = None,
    spatial_mask_preview_path: Path | None = None,
    spatial_mask_overlay_path: Path | None = None,
) -> SelectedCandidate:
    return SelectedCandidate(
        id=id_,
        modality=modality,
        timestamp_seconds=timestamp,
        text="evidence",
        clip_start_seconds=clip_start,
        clip_end_seconds=clip_end,
        spatial_mask_path=spatial_mask_path,
        spatial_mask_preview_path=spatial_mask_preview_path,
        spatial_mask_overlay_path=spatial_mask_overlay_path,
        selection_rank=1,
        relevance_score=1,
        normalized_score=1,
        mmr_score=1,
        source_score_type="test",
        reason="test",
    )
