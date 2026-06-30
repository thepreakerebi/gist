import json
from pathlib import Path
from types import SimpleNamespace

import gist.eval.long_video_suite as long_video_suite
from gist.core.presets import CompressionPreset
from gist.core.query_intent import QueryIntent
from gist.core.schemas import CompressionMetrics, CompressionResponse, Modality, SelectedCandidate
from gist.eval.long_video_suite import (
    LongVideoQueryProposal,
    LongVideoSuiteGates,
    append_reviewed_long_video_quality_draft,
    audit_long_video_artifacts,
    build_long_video_curation_queue,
    build_long_video_metadata_refresh_queue,
    curate_long_video_query_proposal,
    evaluate_long_video_suite,
    main,
    render_long_video_curation_queue_markdown,
    render_long_video_metadata_refresh_queue_markdown,
    review_long_video_quality_draft,
    render_long_video_suite_html,
    render_long_video_suite_markdown,
)
from gist.eval.quality import QualityCase
from gist.media.models import IngestedVideo, VideoMetadata


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
    assert report.expansion_plan.needed_cases == 1
    assert report.expansion_plan.needed_long_video_cases == 2
    assert report.expansion_plan.needed_by_category[QueryIntent.SPEECH_SEMANTIC.value] == 1


def test_long_video_suite_expansion_plan_prioritizes_missing_categories(
    tmp_path: Path,
) -> None:
    speech_case = QualityCase(
        id="speech",
        query_category=QueryIntent.SPEECH_SEMANTIC,
        domain="education",
        compression_path=_write_artifact(
            tmp_path / "speech" / "compression.json",
            "speech-video",
            3700,
        ),
        expected_answer_terms=["answer"],
        expected_evidence_terms=["evidence"],
    )

    report = evaluate_long_video_suite(
        [speech_case],
        LongVideoSuiteGates(
            min_cases=6,
            min_distinct_videos=2,
            min_distinct_domains=2,
            min_cases_per_category=2,
        ),
    )
    markdown = render_long_video_suite_markdown(report)
    html = render_long_video_suite_html(report)

    assert report.expansion_plan.needed_cases == 5
    assert report.expansion_plan.needed_distinct_videos == 1
    assert report.expansion_plan.needed_distinct_domains == 1
    assert report.expansion_plan.needed_by_category == {
        QueryIntent.SPEECH_SEMANTIC.value: 1,
        QueryIntent.VISUAL_OBJECT_ACTION.value: 2,
        QueryIntent.TEMPORAL_BEFORE_AFTER.value: 2,
        QueryIntent.GLOBAL_SUMMARY.value: 2,
        QueryIntent.MIXED_AV.value: 2,
    }
    assert report.expansion_plan.query_proposals
    assert report.expansion_plan.query_proposals[0].query_category == QueryIntent.SPEECH_SEMANTIC
    assert report.expansion_plan.query_proposals[0].video_id == "speech-video"
    assert "Expansion Plan" in markdown
    assert "Add 2 verified `mixed_av` case(s)." in markdown
    assert "Query Proposals" in markdown
    assert "Missing Query Categories" in html
    assert "Proposed Query" in html


def test_long_video_curation_queue_reports_next_actions(tmp_path: Path) -> None:
    speech_case = QualityCase(
        id="speech",
        query_category=QueryIntent.SPEECH_SEMANTIC,
        domain="education",
        compression_path=_write_artifact(
            tmp_path / "speech" / "compression.json",
            "speech-video",
            3700,
        ),
        expected_answer_terms=["answer"],
        expected_evidence_terms=["evidence"],
    )
    report = evaluate_long_video_suite(
        [speech_case],
        LongVideoSuiteGates(
            min_cases=6,
            min_distinct_videos=2,
            min_distinct_domains=2,
            min_cases_per_category=2,
        ),
    )

    queue = build_long_video_curation_queue(
        report=report,
        dataset_path=tmp_path / "suite.jsonl",
    )
    markdown = render_long_video_curation_queue_markdown(queue)

    assert queue.needed_cases == 5
    assert queue.needed_by_category[QueryIntent.MIXED_AV.value] == 2
    assert queue.items
    assert queue.items[0].proposal_index == 0
    assert "--curate-proposal-index 0" in queue.items[0].command
    assert "Gist Long-Video Curation Queue" in markdown
    assert "Run proposal `0` first" in markdown


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
    unreliable = _write_artifact(
        tmp_path / "unreliable" / "compression.json",
        "unreliable-video",
        3900,
    )
    payload = json.loads(unreliable.read_text())
    payload["compression"]["answer"] = (
        "I could not derive a reliable answer from the selected evidence."
    )
    unreliable.write_text(json.dumps(payload) + "\n")
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
    assert audit.artifacts == 6
    assert audit.curated_artifacts == 1
    assert audit.candidate_artifacts == 1
    assert by_name["candidate"].candidate
    assert by_name["curated"].curated
    assert "already curated" in by_name["curated"].reasons
    assert any("duration" in reason for reason in by_name["short"].reasons)
    assert any("token reduction" in reason for reason in by_name["low-token"].reasons)
    assert "answer appears OCR-noisy" in by_name["noisy-ocr"].reasons
    assert "unreliable generated answer" in by_name["unreliable"].reasons


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


def test_long_video_suite_cli_writes_curation_queue(tmp_path: Path, capsys) -> None:
    cases = _cases(tmp_path)
    dataset = tmp_path / "suite.jsonl"
    dataset.write_text(
        "\n".join(case.model_dump_json(exclude_none=True) for case in cases) + "\n"
    )
    queue_path = tmp_path / "reports" / "queue.json"
    queue_markdown_path = tmp_path / "reports" / "queue.md"

    exit_code = main(
        [
            "--dataset",
            str(dataset),
            "--queue-output",
            str(queue_path),
            "--queue-markdown-output",
            str(queue_markdown_path),
            "--min-cases",
            "7",
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
    assert exit_code == 1
    assert "queue_items=" in output
    queue = json.loads(queue_path.read_text())
    assert queue["needed_cases"] == 2
    assert queue["items"]
    assert "Curation Queue" in queue_markdown_path.read_text()


def test_long_video_metadata_refresh_queue_reports_missing_metadata(tmp_path: Path) -> None:
    video_path = tmp_path / "source video.mp4"
    video_path.write_text("video")
    missing_metadata = _write_artifact(
        tmp_path / "missing" / "compression.json",
        "video-a",
        3700,
        source_path=video_path,
    )
    payload = json.loads(missing_metadata.read_text())
    payload["compression"]["query"] = "What does the speaker explain?"
    payload["compression"]["transcript_metadata"] = None
    missing_metadata.write_text(json.dumps(payload) + "\n")
    ready_metadata = _write_artifact(
        tmp_path / "ready" / "compression.json",
        "video-b",
        3700,
        source_path=video_path,
    )
    cases = [
        QualityCase(
            id="missing",
            query_category=QueryIntent.SPEECH_SEMANTIC,
            domain="education",
            compression_path=missing_metadata,
            expected_answer_terms=["speaker"],
            expected_evidence_terms=["speaker"],
        ),
        QualityCase(
            id="ready",
            query_category=QueryIntent.SPEECH_SEMANTIC,
            domain="education",
            compression_path=ready_metadata,
            expected_answer_terms=["speaker"],
            expected_evidence_terms=["speaker"],
        ),
    ]

    queue = build_long_video_metadata_refresh_queue(cases)
    markdown = render_long_video_metadata_refresh_queue_markdown(queue)

    assert queue.refresh_needed == 1
    assert queue.items[0].case_id == "missing"
    assert "source video.mp4" in queue.items[0].command
    assert "--audio-scorer whisper" in queue.items[0].command
    assert "--transcript-quality balanced" in queue.items[0].command
    assert "Transcript Metadata Refresh Queue" in markdown


def test_long_video_suite_cli_writes_metadata_refresh_queue(
    tmp_path: Path,
    capsys,
) -> None:
    artifact = _write_artifact(tmp_path / "missing" / "compression.json", "video", 3700)
    payload = json.loads(artifact.read_text())
    payload["compression"]["transcript_metadata"] = None
    artifact.write_text(json.dumps(payload) + "\n")
    dataset = tmp_path / "suite.jsonl"
    dataset.write_text(
        QualityCase(
            id="missing",
            query_category=QueryIntent.SPEECH_SEMANTIC,
            domain="education",
            compression_path=artifact,
            expected_answer_terms=["answer"],
            expected_evidence_terms=["evidence"],
        ).model_dump_json(exclude_none=True)
        + "\n"
    )
    refresh_path = tmp_path / "reports" / "metadata-refresh.json"
    refresh_markdown_path = tmp_path / "reports" / "metadata-refresh.md"

    exit_code = main(
        [
            "--dataset",
            str(dataset),
            "--metadata-refresh-output",
            str(refresh_path),
            "--metadata-refresh-markdown-output",
            str(refresh_markdown_path),
            "--metadata-refresh-quality",
            "accurate",
            "--min-cases",
            "1",
            "--min-distinct-videos",
            "1",
            "--min-distinct-domains",
            "1",
            "--min-cases-per-category",
            "1",
            "--min-transcript-metadata-rate",
            "0",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "metadata_refresh_items=1" in output
    refresh = json.loads(refresh_path.read_text())
    assert refresh["refresh_needed"] == 1
    assert refresh["target_transcript_quality"] == "accurate"
    assert "Metadata Refresh Queue" in refresh_markdown_path.read_text()


def test_curate_long_video_query_proposal_writes_review_bundle(tmp_path: Path) -> None:
    video_path = tmp_path / "source.mp4"
    video_path.write_text("video")
    source_artifact = _write_artifact(
        tmp_path / "runs" / "source-video" / "existing" / "compression.json",
        "source-video",
        3700,
        source_path=video_path,
    )
    proposal = LongVideoQueryProposal(
        video_id="source-video",
        domain="education",
        query_category=QueryIntent.MIXED_AV,
        query="What does the speaker say while the slide is shown?",
        rationale="missing mixed AV coverage",
        source_artifact=source_artifact,
    )

    result = curate_long_video_query_proposal(
        proposal=proposal,
        output_root=tmp_path / "curation",
        pipeline=FakePipeline(),
    )

    assert result.video_path == video_path
    assert result.compression_path.exists()
    assert result.html_report_path.exists()
    assert result.draft_case_path.exists()
    assert result.review_json_path.exists()
    assert result.review_markdown_path.exists()
    draft = json.loads(result.draft_case_path.read_text())
    assert draft["query_category"] == "mixed_av"
    assert draft["domain"] == "education"
    assert draft["compression_path"] == str(result.compression_path)
    review = json.loads(result.review_json_path.read_text())
    assert not review["ready_for_dataset"]
    assert any("Human review" in warning for warning in review["warnings"])
    assert "Checklist" in result.review_markdown_path.read_text()


def test_long_video_suite_cli_curate_proposal_returns_success_when_gates_fail(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    cases = _cases(tmp_path)
    dataset = tmp_path / "suite.jsonl"
    dataset.write_text(
        "\n".join(case.model_dump_json(exclude_none=True) for case in cases) + "\n"
    )

    def fake_curate(**kwargs):
        return SimpleNamespace(
            compression_path=tmp_path / "compression.json",
            html_report_path=tmp_path / "report.html",
            draft_case_path=tmp_path / "quality-case.draft.jsonl",
            review_markdown_path=tmp_path / "curation-review.md",
        )

    monkeypatch.setattr(long_video_suite, "curate_long_video_query_proposal", fake_curate)

    exit_code = main(
        [
            "--dataset",
            str(dataset),
            "--curate-proposal-index",
            "0",
            "--min-cases",
            "30",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "curation_draft=" in output
    assert "curation_review=" in output
    assert "passed=no" in output


def test_review_long_video_quality_draft_rejects_unreviewed_draft(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "compression.json", "video", 3700)
    draft_path = tmp_path / "quality-case.draft.jsonl"
    draft_path.write_text(
        QualityCase(
            id="draft",
            query_category=QueryIntent.MIXED_AV,
            domain="education",
            compression_path=artifact,
            expected_answer_terms=["visual", "near"],
            expected_evidence_terms=["seconds", "text"],
            relevant_ranges=[],
            min_answer_term_recall=0.75,
            min_evidence_relevance_rate=0.8,
            min_timestamp_hit_rate=0.75,
            min_grounded_evidence_rate=0,
            min_token_reduction_percent=90,
            max_selected_evidence=2,
        ).model_dump_json(exclude_none=True)
        + "\n"
    )

    review = review_long_video_quality_draft(draft_path)

    assert not review.ready_for_dataset
    assert any("Expected terms look noisy" in warning for warning in review.warnings)
    assert any("At least one relevant timestamp" in warning for warning in review.warnings)
    assert any("mixed_av cases should require" in warning for warning in review.warnings)


def test_review_long_video_quality_draft_accepts_reviewed_case(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "compression.json", "video", 3700)
    draft_path = tmp_path / "quality-case.draft.jsonl"
    draft_path.write_text(
        QualityCase(
            id="reviewed",
            query_category=QueryIntent.MIXED_AV,
            domain="education",
            compression_path=artifact,
            expected_answer_terms=["robotics", "control"],
            expected_evidence_terms=["robotics", "control"],
            relevant_ranges=[{"start_seconds": 120.0, "end_seconds": 150.0}],
            min_answer_term_recall=0.75,
            min_evidence_relevance_rate=0.8,
            min_timestamp_hit_rate=0.75,
            min_grounded_evidence_rate=0.5,
            min_token_reduction_percent=90,
            max_selected_evidence=2,
            min_visual_evidence=1,
            min_audio_evidence=1,
        ).model_dump_json(exclude_none=True)
        + "\n"
    )

    review = review_long_video_quality_draft(draft_path)

    assert review.ready_for_dataset
    assert not review.warnings


def test_long_video_suite_cli_reviews_draft_without_dataset(tmp_path: Path, capsys) -> None:
    artifact = _write_artifact(tmp_path / "compression.json", "video", 3700)
    draft_path = tmp_path / "quality-case.draft.jsonl"
    draft_path.write_text(
        QualityCase(
            id="reviewed",
            query_category=QueryIntent.SPEECH_SEMANTIC,
            domain="education",
            compression_path=artifact,
            expected_answer_terms=["founders"],
            expected_evidence_terms=["founders"],
            relevant_ranges=[{"start_seconds": 120.0, "end_seconds": 150.0}],
            min_answer_term_recall=0.75,
            min_evidence_relevance_rate=0.8,
            min_timestamp_hit_rate=0.75,
            min_grounded_evidence_rate=0.5,
            min_token_reduction_percent=90,
            max_selected_evidence=2,
        ).model_dump_json(exclude_none=True)
        + "\n"
    )

    exit_code = main(["--review-draft", str(draft_path)])

    assert exit_code == 0
    assert "ready_for_dataset=yes" in capsys.readouterr().out


def test_append_reviewed_long_video_quality_draft_appends_ready_case(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "compression.json", "video", 3700)
    draft_path = tmp_path / "quality-case.draft.jsonl"
    dataset_path = tmp_path / "long-video-quality.jsonl"
    draft_path.write_text(
        QualityCase(
            id="reviewed",
            query_category=QueryIntent.SPEECH_SEMANTIC,
            domain="education",
            compression_path=artifact,
            expected_answer_terms=["founders"],
            expected_evidence_terms=["founders"],
            relevant_ranges=[{"start_seconds": 120.0, "end_seconds": 150.0}],
            min_answer_term_recall=0.75,
            min_evidence_relevance_rate=0.8,
            min_timestamp_hit_rate=0.75,
            min_grounded_evidence_rate=0.5,
            min_token_reduction_percent=90,
            max_selected_evidence=2,
        ).model_dump_json(exclude_none=True)
        + "\n"
    )

    result = append_reviewed_long_video_quality_draft(draft_path, dataset_path)

    assert result.appended
    assert result.review.ready_for_dataset
    assert [case.id for case in load_cases(dataset_path)] == ["reviewed"]


def test_append_reviewed_long_video_quality_draft_rejects_duplicate_id(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "compression.json", "video", 3700)
    draft_path = tmp_path / "quality-case.draft.jsonl"
    dataset_path = tmp_path / "long-video-quality.jsonl"
    case = QualityCase(
        id="reviewed",
        query_category=QueryIntent.SPEECH_SEMANTIC,
        domain="education",
        compression_path=artifact,
        expected_answer_terms=["founders"],
        expected_evidence_terms=["founders"],
        relevant_ranges=[{"start_seconds": 120.0, "end_seconds": 150.0}],
        min_answer_term_recall=0.75,
        min_evidence_relevance_rate=0.8,
        min_timestamp_hit_rate=0.75,
        min_grounded_evidence_rate=0.5,
        min_token_reduction_percent=90,
        max_selected_evidence=2,
    )
    draft_path.write_text(case.model_dump_json(exclude_none=True) + "\n")
    dataset_path.write_text(case.model_dump_json(exclude_none=True) + "\n")

    result = append_reviewed_long_video_quality_draft(draft_path, dataset_path)

    assert not result.appended
    assert not result.review.ready_for_dataset
    assert "case id already exists" in result.review.warnings[0]
    assert len(load_cases(dataset_path)) == 1


def test_long_video_suite_cli_appends_reviewed_draft(tmp_path: Path, capsys) -> None:
    artifact = _write_artifact(tmp_path / "compression.json", "video", 3700)
    draft_path = tmp_path / "quality-case.draft.jsonl"
    dataset_path = tmp_path / "long-video-quality.jsonl"
    draft_path.write_text(
        QualityCase(
            id="reviewed",
            query_category=QueryIntent.SPEECH_SEMANTIC,
            domain="education",
            compression_path=artifact,
            expected_answer_terms=["founders"],
            expected_evidence_terms=["founders"],
            relevant_ranges=[{"start_seconds": 120.0, "end_seconds": 150.0}],
            min_answer_term_recall=0.75,
            min_evidence_relevance_rate=0.8,
            min_timestamp_hit_rate=0.75,
            min_grounded_evidence_rate=0.5,
            min_token_reduction_percent=90,
            max_selected_evidence=2,
        ).model_dump_json(exclude_none=True)
        + "\n"
    )

    exit_code = main(
        [
            "--review-draft",
            str(draft_path),
            "--append-draft-to",
            str(dataset_path),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "appended=yes" in output
    assert [case.id for case in load_cases(dataset_path)] == ["reviewed"]


def load_cases(path: Path) -> list[QualityCase]:
    return [
        QualityCase.model_validate(json.loads(line))
        for line in path.read_text().splitlines()
    ]


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


class FakePipeline:
    def run(self, **kwargs):
        video_path = kwargs["video_path"]
        query = kwargs["query"]
        ingestion = IngestedVideo(
            video_id="source-video",
            source_path=video_path,
            metadata=VideoMetadata(duration_seconds=3700, has_audio=True),
            frames=[],
            audio_windows=[],
        )
        compression = CompressionResponse(
            video_id="source-video",
            query=query,
            answer="The speaker explains the visual evidence clearly.",
            preset=CompressionPreset.BALANCED,
            query_intent=QueryIntent.MIXED_AV,
            selected=[
                SelectedCandidate(
                    id="a-1",
                    modality=Modality.AUDIO,
                    timestamp_seconds=100.0,
                    text="speaker explains visual evidence",
                    selection_rank=1,
                    relevance_score=1,
                    normalized_score=1,
                    mmr_score=1,
                    source_score_type="test",
                    reason="test",
                )
            ],
            metrics=CompressionMetrics(
                input_candidates=10,
                selected_candidates=1,
                visual_selected=0,
                audio_selected=1,
                estimated_candidate_reduction_ratio=0.9,
                estimated_candidate_reduction_percent=90,
                dropped_candidates=9,
                budget_preset_used=CompressionPreset.BALANCED,
                estimated_token_reduction_percent=95,
            ),
        )
        return ingestion, compression


def _write_artifact(
    path: Path,
    video_id: str,
    duration_seconds: float,
    source_path: Path | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "ingestion": {
                    "source_path": str(source_path or "source.mp4"),
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
