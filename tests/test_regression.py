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
            visual_selected=0,
            audio_selected=len(selected),
            estimated_candidate_reduction_ratio=len(selected) / 100,
            estimated_candidate_reduction_percent=100 - len(selected),
            dropped_candidates=100 - len(selected),
            budget_preset_used=CompressionPreset.BALANCED,
            estimated_baseline_tokens=3200,
            estimated_compressed_tokens=32 * len(selected),
            estimated_saved_tokens=3200 - (32 * len(selected)),
            estimated_token_reduction_ratio=1 - (token_reduction / 100),
            estimated_token_reduction_percent=token_reduction,
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
) -> SelectedCandidate:
    return SelectedCandidate(
        id=id_,
        modality=Modality.AUDIO,
        timestamp_seconds=timestamp,
        text="evidence",
        clip_start_seconds=clip_start,
        clip_end_seconds=clip_end,
        selection_rank=1,
        relevance_score=1,
        normalized_score=1,
        mmr_score=1,
        source_score_type="test",
        reason="test",
    )
