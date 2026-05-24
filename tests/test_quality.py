import json
from pathlib import Path

from gist.core.presets import CompressionPreset
from gist.core.schemas import CompressionMetrics, CompressionResponse, Modality, SelectedCandidate
from gist.eval.quality import (
    QualityCase,
    QualityExtractionOptions,
    check_quality_dataset,
    draft_quality_case,
    draft_quality_cases_from_root,
    evaluate_quality_case,
    main,
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
    assert result.failure_categories == [
        "answer_grounding",
        "compression_budget",
        "evidence_retrieval",
        "temporal_localization",
    ]
    assert (
        result.recommendation
        == "Improve query-aware retrieval before changing answer generation."
    )
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
    assert report.summary.failure_categories == {}
    assert "| slide | pass |" in markdown


def test_quality_cases_write_structured_extraction_artifacts(tmp_path: Path) -> None:
    compression_path = _write_compression(
        tmp_path,
        answer="The buyer says pricing is too expensive.",
        selected=[
            _item(
                "a-1",
                timestamp=30,
                clip_start=25,
                clip_end=45,
                text="The buyer says pricing is too expensive.",
            )
        ],
        token_reduction=98.0,
    )
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "name": "sales_feedback",
                "labels": ["pricing objection"],
            }
        )
    )

    report = run_quality_cases(
        [
            QualityCase(
                id="sales-call",
                compression_path=compression_path,
                expected_answer_terms=["pricing"],
                expected_evidence_terms=["pricing"],
                relevant_timestamps=[30],
            )
        ],
        output_root=tmp_path / "quality",
        extraction_options=QualityExtractionOptions(schema_path=schema_path),
    )

    artifact = report.results[0].extraction
    assert artifact is not None
    assert artifact.items == 1
    assert artifact.json_path.exists()
    assert artifact.markdown_path.exists()
    assert artifact.html_path.exists()
    assert artifact.csv_path.exists()
    assert "pricing objection" in artifact.markdown_path.read_text()
    assert "sales_feedback" in render_quality_markdown(report)


def test_check_quality_dataset_reports_warnings(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"
    case = QualityCase(id="unchecked", compression_path=missing_path)

    check = check_quality_dataset([case])

    assert check.cases == 1
    assert check.replay_cases == 1
    assert any("compression_path does not exist" in warning for warning in check.warnings)
    assert any("expected_answer_terms is empty" in warning for warning in check.warnings)


def test_quality_cli_check_only_returns_nonzero_for_warnings(tmp_path: Path) -> None:
    dataset = tmp_path / "quality.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "unchecked",
                "compression_path": str(tmp_path / "missing.json"),
            }
        )
        + "\n"
    )

    exit_code = main(["--dataset", str(dataset), "--check-only"])

    assert exit_code == 1


def test_draft_quality_case_from_existing_compression(tmp_path: Path) -> None:
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
            )
        ],
        token_reduction=99.4,
    )

    draft = draft_quality_case(compression_path, case_id="yc-draft")

    assert draft.case.id == "yc-draft"
    assert draft.case.compression_path == compression_path
    assert "research" in draft.case.expected_answer_terms
    assert "research" in draft.case.expected_evidence_terms
    assert draft.case.relevant_ranges[0].start_seconds == 375
    assert draft.case.relevant_ranges[0].end_seconds == 395
    assert draft.case.max_selected_evidence == 1
    assert draft.notes


def test_quality_cli_drafts_case_without_dataset(tmp_path: Path, capsys) -> None:
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

    exit_code = main(["--draft-case-from", str(compression_path), "--case-id", "slide-draft"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"id":"slide-draft"' in captured.out
    assert "expected_answer_terms" in captured.out
    assert "Review expected_answer_terms" in captured.err


def test_draft_quality_cases_from_root_finds_compressions(tmp_path: Path) -> None:
    first_path = tmp_path / "runs" / "video-a" / "query-a" / "compression.json"
    second_path = tmp_path / "runs" / "video-b" / "query-b" / "compression.json"
    _write_compression_at(
        first_path,
        answer="Builders use AI for research.",
        selected=[
            _item(
                "a-1",
                timestamp=10,
                clip_start=5,
                clip_end=15,
                text="AI research workflow.",
            )
        ],
        token_reduction=99.0,
    )
    _write_compression_at(
        second_path,
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

    drafts = draft_quality_cases_from_root(tmp_path / "runs")

    assert [draft.case.id for draft in drafts] == ["video-a-query-a", "video-b-query-b"]
    assert all(draft.case.compression_path is not None for draft in drafts)


def test_quality_cli_writes_batch_drafts(tmp_path: Path) -> None:
    compression_path = tmp_path / "runs" / "video-a" / "query-a" / "compression.json"
    output_path = tmp_path / "drafts.jsonl"
    _write_compression_at(
        compression_path,
        answer="Builders use AI for research.",
        selected=[
            _item(
                "a-1",
                timestamp=10,
                clip_start=5,
                clip_end=15,
                text="AI research workflow.",
            )
        ],
        token_reduction=99.0,
    )

    exit_code = main(
        [
            "--draft-cases-from-root",
            str(tmp_path / "runs"),
            "--draft-output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    lines = output_path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["id"] == "video-a-query-a"


def test_quality_cli_can_write_extraction_artifacts(tmp_path: Path) -> None:
    compression_path = _write_compression(
        tmp_path,
        answer="The buyer says pricing is too expensive.",
        selected=[
            _item(
                "a-1",
                timestamp=30,
                clip_start=25,
                clip_end=45,
                text="The buyer says pricing is too expensive.",
            )
        ],
        token_reduction=98.0,
    )
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "name": "sales_feedback",
                "labels": ["pricing objection"],
            }
        )
    )
    dataset_path = tmp_path / "quality.jsonl"
    output_path = tmp_path / "quality.json"
    output_root = tmp_path / "quality-root"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "sales-call",
                "compression_path": str(compression_path),
                "expected_answer_terms": ["pricing"],
                "expected_evidence_terms": ["pricing"],
                "relevant_timestamps": [30],
            }
        )
        + "\n"
    )

    exit_code = main(
        [
            "--dataset",
            str(dataset_path),
            "--output",
            str(output_path),
            "--output-root",
            str(output_root),
            "--extraction-schema",
            str(schema_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text())
    artifact = payload["results"][0]["extraction"]
    assert artifact["items"] == 1
    assert Path(artifact["json_path"]).exists()
    assert Path(artifact["csv_path"]).exists()
    assert (output_root / "sales-call" / "extraction" / "extraction.html").exists()
    assert (output_root / "sales-call" / "extraction" / "extraction.csv").exists()


def test_quality_cli_can_use_builtin_extraction_schema_name(tmp_path: Path) -> None:
    compression_path = _write_compression(
        tmp_path,
        answer="The buyer says pricing is too expensive.",
        selected=[
            _item(
                "a-1",
                timestamp=30,
                clip_start=25,
                clip_end=45,
                text="The buyer says pricing is too expensive.",
            )
        ],
        token_reduction=98.0,
    )
    dataset_path = tmp_path / "quality.jsonl"
    output_path = tmp_path / "quality.json"
    output_root = tmp_path / "quality-root"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "sales-call",
                "compression_path": str(compression_path),
                "expected_answer_terms": ["pricing"],
                "expected_evidence_terms": ["pricing"],
                "relevant_timestamps": [30],
            }
        )
        + "\n"
    )

    exit_code = main(
        [
            "--dataset",
            str(dataset_path),
            "--output",
            str(output_path),
            "--output-root",
            str(output_root),
            "--extraction-schema-name",
            "customer_objections",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text())
    artifact = payload["results"][0]["extraction"]
    assert artifact["schema_name"] == "customer_objections"
    assert artifact["items"] == 1


def _write_compression(
    tmp_path: Path,
    answer: str,
    selected: list[SelectedCandidate],
    token_reduction: float,
) -> Path:
    path = tmp_path / "compression.json"
    _write_compression_at(path, answer=answer, selected=selected, token_reduction=token_reduction)
    return path


def _write_compression_at(
    path: Path,
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
    path.parent.mkdir(parents=True, exist_ok=True)
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
