import json
from pathlib import Path

from gist.eval.extraction import (
    ExtractionEvalCase,
    ExpectedExtractionItem,
    evaluate_extraction_case,
    main,
    render_extraction_eval_markdown,
    run_extraction_eval_cases,
)
from gist.eval.regression import TimeRange
from gist.gateway.structured import ExtractedItem, StructuredExtractionResponse


def test_evaluate_extraction_case_passes_for_label_time_and_support(
    tmp_path: Path,
) -> None:
    extraction_path = _write_extraction(
        tmp_path,
        [
            ExtractedItem(
                label="pricing objection",
                description="The buyer says pricing is too expensive.",
                timestamp_start_seconds=30,
                timestamp_end_seconds=45,
                evidence_id="a-1",
                evidence_rank=1,
                confidence=0.9,
                support_text="pricing is too expensive",
            )
        ],
    )
    case = ExtractionEvalCase(
        id="sales",
        extraction_path=extraction_path,
        expected_items=[
            ExpectedExtractionItem(
                label="pricing objection",
                support_terms=["pricing", "expensive"],
                time_range=TimeRange(start_seconds=28, end_seconds=46),
            )
        ],
        min_label_recall=1.0,
        min_timestamp_hit_rate=1.0,
        min_support_term_recall=1.0,
    )

    result = evaluate_extraction_case(case)

    assert result.passed is True
    assert result.label_recall == 1.0
    assert result.timestamp_hit_rate == 1.0
    assert result.support_term_recall == 1.0


def test_evaluate_extraction_case_reports_failures(tmp_path: Path) -> None:
    extraction_path = _write_extraction(
        tmp_path,
        [
            ExtractedItem(
                label="feature request",
                description="The buyer asks for export support.",
                timestamp_start_seconds=90,
                timestamp_end_seconds=110,
                evidence_id="a-1",
                evidence_rank=1,
                confidence=0.9,
                support_text="export support",
            )
        ],
    )
    case = ExtractionEvalCase(
        id="bad-sales",
        extraction_path=extraction_path,
        expected_items=[
            ExpectedExtractionItem(
                label="pricing objection",
                support_terms=["pricing", "expensive"],
                time_range=TimeRange(start_seconds=28, end_seconds=46),
            )
        ],
        max_items=0 + 1,
    )

    result = evaluate_extraction_case(case)

    assert result.passed is False
    assert any("label recall" in failure for failure in result.failures)
    assert any("timestamp hit rate" in failure for failure in result.failures)
    assert any("support term recall" in failure for failure in result.failures)


def test_run_extraction_eval_cases_and_markdown(tmp_path: Path) -> None:
    extraction_path = _write_extraction(
        tmp_path,
        [
            ExtractedItem(
                label="positive reaction",
                description="The user says the product works great.",
                timestamp_start_seconds=10,
                timestamp_end_seconds=15,
                evidence_id="a-1",
                evidence_rank=1,
                confidence=0.9,
                support_text="works great",
            )
        ],
    )

    report = run_extraction_eval_cases(
        [
            ExtractionEvalCase(
                id="positive",
                extraction_path=extraction_path,
                expected_items=[
                    ExpectedExtractionItem(
                        label="positive reaction",
                        support_terms=["great"],
                    )
                ],
            )
        ]
    )
    markdown = render_extraction_eval_markdown(report)

    assert report.passed is True
    assert "| positive | pass |" in markdown


def test_extraction_eval_cli_writes_outputs(tmp_path: Path) -> None:
    extraction_path = _write_extraction(
        tmp_path,
        [
            ExtractedItem(
                label="product mention",
                description="The speaker mentions Gist.",
                timestamp_start_seconds=10,
                timestamp_end_seconds=15,
                evidence_id="a-1",
                evidence_rank=1,
                confidence=0.9,
                support_text="mentions Gist",
            )
        ],
    )
    dataset_path = tmp_path / "dataset.jsonl"
    output_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "product",
                "extraction_path": str(extraction_path),
                "expected_items": [
                    {
                        "label": "product mention",
                        "support_terms": ["Gist"],
                    }
                ],
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
            "--markdown-output",
            str(markdown_path),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert markdown_path.exists()


def _write_extraction(tmp_path: Path, items: list[ExtractedItem]) -> Path:
    response = StructuredExtractionResponse(
        schema_name="sales_feedback",
        query="Find feedback.",
        item_type="feedback",
        items=items,
        prompt="prompt",
        provider="test",
    )
    path = tmp_path / "extraction.json"
    response.write_json(path)
    return path
