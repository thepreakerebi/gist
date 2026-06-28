import json
from pathlib import Path

from gist.core.query_intent import QueryIntent
from gist.eval.long_video_suite import (
    LongVideoSuiteGates,
    audit_long_video_artifacts,
    evaluate_long_video_suite,
    main,
    render_long_video_suite_html,
    render_long_video_suite_markdown,
)
from gist.eval.quality import QualityCase


def test_long_video_suite_passes_complete_coverage(tmp_path: Path) -> None:
    cases = _cases(tmp_path)
    report = evaluate_long_video_suite(
        cases,
        LongVideoSuiteGates(
            min_cases=5,
            min_distinct_videos=5,
            min_distinct_domains=3,
            min_cases_per_category=1,
        ),
    )

    assert report.passed
    assert report.case_count == 5
    assert report.long_video_case_count == 5
    assert len(report.video_counts) == 5
    assert report.health.avg_token_reduction_percent == 95
    assert report.health.transcript_metadata_rate == 1
    assert report.health.answered_rate == 1
    assert "Passed: yes" in render_long_video_suite_markdown(report)
    assert "Run Health" in render_long_video_suite_markdown(report)
    assert "<strong>Status:</strong> pass" in render_long_video_suite_html(report)


def test_long_video_suite_reports_coverage_and_metadata_gaps(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "short.json", "short", 120)
    case = QualityCase(
        id="incomplete",
        compression_path=artifact,
        expected_answer_terms=["answer"],
        expected_evidence_terms=["evidence"],
    )

    report = evaluate_long_video_suite(
        [case],
        LongVideoSuiteGates(min_cases=2, min_cases_per_category=1),
    )

    assert not report.passed
    assert any("query_category" in failure for failure in report.metadata_failures)
    assert any("duration" in failure for failure in report.metadata_failures)


def test_long_video_suite_reports_run_health_failures(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "noisy.json", "video", 3700)
    payload = json.loads(artifact.read_text())
    payload["compression"]["answer"] = ""
    payload["compression"]["metrics"]["estimated_token_reduction_percent"] = 40
    payload["compression"]["quality_warnings"] = [
        {"code": "noisy_transcript_evidence", "message": "noisy"}
    ]
    payload["compression"]["transcript_metadata"] = None
    artifact.write_text(json.dumps(payload) + "\n")
    case = QualityCase(
        id="noisy",
        query_category=QueryIntent.GLOBAL_SUMMARY,
        domain="education",
        compression_path=artifact,
        expected_answer_terms=["answer"],
        expected_evidence_terms=["evidence"],
    )

    report = evaluate_long_video_suite(
        [case],
        LongVideoSuiteGates(
            min_cases=1,
            min_distinct_videos=1,
            min_distinct_domains=1,
            min_cases_per_category=1,
            min_avg_token_reduction_percent=90,
            max_noisy_transcript_warning_rate=0,
            min_transcript_metadata_rate=1,
            min_answered_rate=1,
        ),
    )

    assert not report.passed
    failed_gates = {result.name for result in report.gate_results if not result.passed}
    assert "avg_token_reduction_percent" in failed_gates
    assert "noisy_transcript_warning_rate" in failed_gates
    assert "transcript_metadata_rate" in failed_gates
    assert "answered_rate" in failed_gates


def test_audit_long_video_artifacts_classifies_candidates(tmp_path: Path) -> None:
    curated = _write_artifact(tmp_path / "curated" / "compression.json", "curated", 3700)
    _write_artifact(
        tmp_path / "candidate" / "compression.json",
        "candidate-video",
        4200,
    )
    _write_artifact(tmp_path / "short" / "compression.json", "short-video", 120)
    low_token = _write_artifact(
        tmp_path / "low-token" / "compression.json",
        "low-token-video",
        3900,
    )
    payload = json.loads(low_token.read_text())
    payload["compression"]["metrics"]["estimated_token_reduction_percent"] = 80
    low_token.write_text(json.dumps(payload) + "\n")
    noisy_ocr = _write_artifact(
        tmp_path / "noisy-ocr" / "compression.json",
        "noisy-ocr-video",
        3900,
    )
    payload = json.loads(noisy_ocr.read_text())
    payload["compression"]["answer"] = "on-screen text near 3341.69 seconds: oe ORTOP Pa r"
    noisy_ocr.write_text(json.dumps(payload) + "\n")
    cases = [
        QualityCase(
            id="curated",
            query_category=QueryIntent.SPEECH_SEMANTIC,
            domain="education",
            compression_path=curated,
            expected_answer_terms=["answer"],
            expected_evidence_terms=["evidence"],
        )
    ]

    audit = audit_long_video_artifacts(
        root=tmp_path,
        cases=cases,
        gates=LongVideoSuiteGates(
            min_cases=1,
            min_distinct_videos=1,
            min_distinct_domains=1,
            min_cases_per_category=1,
            min_avg_token_reduction_percent=90,
        ),
    )

    by_name = {item.path.parent.name: item for item in audit.items}
    assert audit.artifacts == 5
    assert audit.curated_artifacts == 1
    assert audit.candidate_artifacts == 1
    assert by_name["candidate"].candidate
    assert by_name["curated"].curated
    assert "already curated" in by_name["curated"].reasons
    assert any("duration" in reason for reason in by_name["short"].reasons)
    assert any("token reduction" in reason for reason in by_name["low-token"].reasons)
    assert "answer appears OCR-noisy" in by_name["noisy-ocr"].reasons


def test_long_video_suite_cli_writes_readiness_reports(tmp_path: Path, capsys) -> None:
    cases = _cases(tmp_path)
    dataset = tmp_path / "suite.jsonl"
    dataset.write_text(
        "\n".join(case.model_dump_json(exclude_none=True) for case in cases) + "\n"
    )
    report_dir = tmp_path / "reports"

    exit_code = main(
        [
            "--dataset",
            str(dataset),
            "--output",
            str(report_dir / "suite.json"),
            "--markdown-output",
            str(report_dir / "suite.md"),
            "--html-output",
            str(report_dir / "suite.html"),
            "--min-cases",
            "5",
            "--min-distinct-videos",
            "5",
            "--min-distinct-domains",
            "3",
            "--min-cases-per-category",
            "1",
            "--min-transcript-metadata-rate",
            "1",
        ]
    )

    assert exit_code == 0
    assert "passed=yes" in capsys.readouterr().out
    assert (report_dir / "suite.json").exists()
    assert (report_dir / "suite.md").exists()
    assert (report_dir / "suite.html").exists()


def test_long_video_suite_cli_writes_artifact_audit(tmp_path: Path, capsys) -> None:
    cases = _cases(tmp_path)
    dataset = tmp_path / "suite.jsonl"
    dataset.write_text(
        "\n".join(case.model_dump_json(exclude_none=True) for case in cases) + "\n"
    )
    _write_artifact(tmp_path / "runs" / "new-case" / "compression.json", "new-video", 3900)
    audit_path = tmp_path / "reports" / "audit.json"

    exit_code = main(
        [
            "--dataset",
            str(dataset),
            "--audit-root",
            str(tmp_path / "runs"),
            "--audit-output",
            str(audit_path),
            "--min-cases",
            "5",
            "--min-distinct-videos",
            "5",
            "--min-distinct-domains",
            "3",
            "--min-cases-per-category",
            "1",
            "--min-transcript-metadata-rate",
            "1",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "audit_candidates=1" in output
    assert audit_path.exists()
    audit = json.loads(audit_path.read_text())
    assert audit["candidate_artifacts"] == 1


def _cases(tmp_path: Path) -> list[QualityCase]:
    categories = [
        QueryIntent.SPEECH_SEMANTIC,
        QueryIntent.VISUAL_OBJECT_ACTION,
        QueryIntent.TEMPORAL_BEFORE_AFTER,
        QueryIntent.GLOBAL_SUMMARY,
        QueryIntent.MIXED_AV,
    ]
    return [
        QualityCase(
            id=f"case-{index}",
            query_category=category,
            domain=("education", "film", "healthcare")[index % 3],
            compression_path=_write_artifact(
                tmp_path / f"case-{index}.json",
                f"video-{index}",
                3700,
            ),
            expected_answer_terms=["answer"],
            expected_evidence_terms=["evidence"],
        )
        for index, category in enumerate(categories)
    ]


def _write_artifact(path: Path, video_id: str, duration_seconds: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "ingestion": {
                    "metadata": {"duration_seconds": duration_seconds},
                },
                "compression": {
                    "video_id": video_id,
                    "query": "query",
                    "answer": "This answer covers the expected evidence.",
                    "selected": [
                        {
                            "id": "a-1",
                            "modality": "audio",
                            "timestamp_seconds": 10,
                            "text": "evidence answer",
                            "selection_rank": 1,
                            "relevance_score": 1,
                            "normalized_score": 1,
                            "mmr_score": 1,
                            "source_score_type": "test",
                            "reason": "test",
                        }
                    ],
                    "metrics": {
                        "selected_candidates": 1,
                        "estimated_token_reduction_percent": 95,
                    },
                    "quality_warnings": [],
                    "transcript_metadata": {
                        "quality": "fast",
                        "model_size": "tiny",
                        "device": "cpu",
                        "compute_type": "int8",
                        "beam_size": 1,
                    },
                },
            }
        )
        + "\n"
    )
    return path
