import json
from pathlib import Path

from gist.eval.acceptance import (
    AcceptanceGates,
    evaluate_acceptance,
    main,
    render_acceptance_html,
    render_acceptance_markdown,
)
from gist.eval.quality import QualityReport, QualityResult, QualitySummary


def test_acceptance_passes_when_quality_meets_gates() -> None:
    report = evaluate_acceptance(
        _quality_report(pass_rate=1.0, failures=0),
        AcceptanceGates(
            min_cases=1,
            min_pass_rate=1.0,
            min_avg_answer_term_recall=0.8,
            min_avg_evidence_relevance_rate=0.8,
            min_avg_timestamp_hit_rate=0.8,
            min_avg_grounded_evidence_rate=0.8,
            min_avg_token_reduction_percent=90.0,
            max_failure_count=0,
        ),
    )

    assert report.passed is True
    assert all(result.passed for result in report.gate_results)
    assert "Gist Acceptance Report" in render_acceptance_markdown(report)
    assert "Gist Acceptance Report" in render_acceptance_html(report)
    assert any(result.name == "avg_grounded_evidence_rate" for result in report.gate_results)


def test_acceptance_fails_when_gate_is_not_met() -> None:
    report = evaluate_acceptance(
        _quality_report(pass_rate=0.5, failures=1),
        AcceptanceGates(min_pass_rate=0.9, max_failure_count=0),
    )

    assert report.passed is False
    assert any(result.name == "pass_rate" and not result.passed for result in report.gate_results)
    assert any(
        result.name == "failure_count" and not result.passed
        for result in report.gate_results
    )


def test_acceptance_cli_writes_reports(tmp_path: Path) -> None:
    compression_path = tmp_path / "compression.json"
    compression_path.write_text(json.dumps({"compression": _compression_payload()}))
    dataset = tmp_path / "acceptance.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "case-a",
                "compression_path": str(compression_path),
                "expected_answer_terms": ["pricing"],
                "expected_evidence_terms": ["pricing"],
                "relevant_ranges": [{"start_seconds": 10, "end_seconds": 20}],
                "timestamp_tolerance_seconds": 0,
                "min_answer_term_recall": 1,
                "min_evidence_term_coverage": 1,
                "min_evidence_relevance_rate": 1,
                "min_timestamp_hit_rate": 1,
                "min_token_reduction_percent": 90,
                "max_selected_evidence": 1,
            }
        )
        + "\n"
    )
    output = tmp_path / "acceptance.json"
    markdown = tmp_path / "acceptance.md"
    html = tmp_path / "acceptance.html"

    exit_code = main(
        [
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--markdown-output",
            str(markdown),
            "--html-output",
            str(html),
            "--min-cases",
            "1",
            "--min-pass-rate",
            "1",
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text())["passed"] is True
    assert markdown.exists()
    assert html.exists()


def test_acceptance_cli_drafts_case_from_compression(tmp_path: Path) -> None:
    compression_path = tmp_path / "compression.json"
    compression_path.write_text(json.dumps({"compression": _compression_payload()}))
    output = tmp_path / "draft.jsonl"

    exit_code = main(
        [
            "--draft-case-from",
            str(compression_path),
            "--case-id",
            "draft-a",
            "--draft-output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text())
    assert payload["id"] == "draft-a"
    assert payload["compression_path"] == str(compression_path)
    assert payload["expected_answer_terms"]
    assert "video123abc" not in payload["expected_answer_terms"]
    assert payload["relevant_ranges"]


def test_acceptance_cli_drafts_cases_from_root(tmp_path: Path) -> None:
    first = tmp_path / "runs" / "video-a" / "query-a" / "compression.json"
    second = tmp_path / "runs" / "video-b" / "query-b" / "compression.json"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text(json.dumps({"compression": _compression_payload()}))
    second.write_text(json.dumps({"compression": _compression_payload()}))
    output = tmp_path / "drafts.jsonl"

    exit_code = main(
        [
            "--draft-cases-from-root",
            str(tmp_path / "runs"),
            "--draft-max-cases",
            "1",
            "--draft-output",
            str(output),
        ]
    )

    assert exit_code == 0
    lines = output.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["compression_path"].endswith("compression.json")


def _quality_report(pass_rate: float, failures: int) -> QualityReport:
    cases = 2
    passed = int(pass_rate * cases)
    results = [
        QualityResult(
            id=f"case-{index}",
            passed=index >= failures,
            query="query",
            answer="answer",
            answer_term_recall=0.9,
            evidence_term_coverage=0.9,
            evidence_relevance_rate=0.9,
            timestamp_hit_rate=0.9,
            grounded_evidence_rate=0.9,
            token_reduction_percent=95.0,
            selected_evidence=1,
            visual_evidence=0,
            audio_evidence=1,
        )
        for index in range(cases)
    ]
    return QualityReport(
        passed=failures == 0,
        summary=QualitySummary(
            cases=cases,
            passed=passed,
            pass_rate=pass_rate,
            avg_answer_term_recall=0.9,
            avg_evidence_term_coverage=0.9,
            avg_evidence_relevance_rate=0.9,
            avg_timestamp_hit_rate=0.9,
            avg_grounded_evidence_rate=0.9,
            avg_token_reduction_percent=95.0,
        ),
        results=results,
    )


def _compression_payload() -> dict:
    return {
        "video_id": "video",
        "query": "Find pricing objections.",
        "answer": "Pricing is too expensive for video123abc.",
        "preset": "balanced",
        "selected": [
            {
                "id": "audio-1",
                "modality": "audio",
                "timestamp_seconds": 15,
                "text": "The customer says pricing is too expensive.",
                "clip_start_seconds": 10,
                "clip_end_seconds": 20,
                "selection_rank": 1,
                "relevance_score": 1,
                "normalized_score": 1,
                "mmr_score": 1,
                "source_score_type": "test",
                "reason": "test",
            }
        ],
        "metrics": {
            "input_candidates": 100,
            "selected_candidates": 1,
            "visual_selected": 0,
            "audio_selected": 1,
            "estimated_candidate_reduction_ratio": 0.01,
            "estimated_candidate_reduction_percent": 99,
            "dropped_candidates": 99,
            "budget_preset_used": "balanced",
            "estimated_token_reduction_percent": 95,
        },
    }
