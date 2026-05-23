import json
from pathlib import Path

from gist.core.presets import CompressionPreset
from gist.core.schemas import CompressionMetrics, CompressionResponse, Modality, SelectedCandidate
from gist.eval.quality import (
    QualityCase,
    evaluate_quality_case,
    render_quality_markdown,
    run_quality_cases,
)


def test_evaluate_quality_case_passes_when_answer_and_evidence_align(tmp_path: Path) -> None:
    compression_path = _write_compression(
        tmp_path,
        answer="Builders use AI for research and writing articles.",
        selected=[
            _item(
                "a-1",
                timestamp=385,
                clip_start=375,
                clip_end=395,
                text="They use AI for deep research before writing articles.",
            ),
            _item(
                "a-2",
                timestamp=1720,
                clip_start=1710,
                clip_end=1730,
                text="Builders can do the work of many engineers.",
            ),
        ],
        token_reduction=99.4,
    )
    case = QualityCase(
        id="yc",
        compression_path=compression_path,
        expected_answer_terms=["research", "articles"],
        expected_evidence_terms=["research", "builders"],
        relevant_timestamps=[385, 1720],
        min_answer_term_recall=1.0,
        min_evidence_term_coverage=1.0,
        min_evidence_relevance_rate=1.0,
        min_timestamp_hit_rate=1.0,
        min_token_reduction_percent=99.0,
        max_selected_evidence=2,
    )

    result = evaluate_quality_case(case)

    assert result.passed is True
    assert result.answer_term_recall == 1.0
    assert result.evidence_relevance_rate == 1.0


def test_evaluate_quality_case_reports_quality_failures(tmp_path: Path) -> None:
    compression_path = _write_compression(
        tmp_path,
        answer="The video discusses startups.",
        selected=[
            _item(
                "a-1",
                timestamp=100,
                clip_start=90,
                clip_end=110,
                text="Unrelated introduction.",
            )
        ],
        token_reduction=60.0,
    )
    case = QualityCase(
        id="bad",
        compression_path=compression_path,
        expected_answer_terms=["research"],
        expected_evidence_terms=["research"],
        relevant_timestamps=[385],
        min_answer_term_recall=1.0,
        min_evidence_term_coverage=1.0,
        min_evidence_relevance_rate=1.0,
        min_timestamp_hit_rate=1.0,
        min_token_reduction_percent=99.0,
    )

    result = evaluate_quality_case(case)

    assert result.passed is False
    assert any("answer term recall" in failure for failure in result.failures)
    assert any("evidence term coverage" in failure for failure in result.failures)
    assert any("evidence relevance rate" in failure for failure in result.failures)
    assert any("timestamp hit rate" in failure for failure in result.failures)
    assert any("token reduction" in failure for failure in result.failures)


def test_run_quality_cases_and_markdown_summary(tmp_path: Path) -> None:
    compression_path = _write_compression(
        tmp_path,
        answer="The slide says GIST TOKEN SAVER.",
        selected=[
            _item(
                "v-1",
                timestamp=3,
                clip_start=0,
                clip_end=6,
                text="GIST TOKEN SAVER",
                modality=Modality.VISUAL,
            )
        ],
        token_reduction=95.0,
    )
    report = run_quality_cases(
        [
            QualityCase(
                id="slide",
                compression_path=compression_path,
                expected_answer_terms=["GIST", "TOKEN", "SAVER"],
                expected_evidence_terms=["GIST", "TOKEN", "SAVER"],
                relevant_timestamps=[3],
                min_answer_term_recall=1.0,
                min_evidence_term_coverage=1.0,
                min_evidence_relevance_rate=1.0,
                min_timestamp_hit_rate=1.0,
            )
        ],
        output_root=tmp_path / "quality",
    )

    markdown = render_quality_markdown(report)

    assert report.passed is True
    assert report.summary.pass_rate == 1.0
    assert "| slide | pass |" in markdown


def _write_compression(
    tmp_path: Path,
    answer: str,
    selected: list[SelectedCandidate],
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
            visual_selected=sum(item.modality == Modality.VISUAL for item in selected),
            audio_selected=sum(item.modality == Modality.AUDIO for item in selected),
            estimated_candidate_reduction_ratio=len(selected) / 100,
            estimated_candidate_reduction_percent=100 - len(selected),
            dropped_candidates=100 - len(selected),
            budget_preset_used=CompressionPreset.BALANCED,
            estimated_token_reduction_percent=token_reduction,
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
    text: str,
    modality: Modality = Modality.AUDIO,
) -> SelectedCandidate:
    return SelectedCandidate(
        id=id_,
        modality=modality,
        timestamp_seconds=timestamp,
        text=text,
        clip_start_seconds=clip_start,
        clip_end_seconds=clip_end,
        selection_rank=1,
        relevance_score=1,
        normalized_score=1,
        mmr_score=1,
        source_score_type="test",
        reason="test",
    )
