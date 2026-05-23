import argparse
import json
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

from gist.eval.regression import TimeRange
from gist.gateway.structured import StructuredExtractionResponse


class ExpectedExtractionItem(BaseModel):
    label: Annotated[str, Field(min_length=1)]
    support_terms: list[str] = Field(default_factory=list)
    time_range: TimeRange | None = None


class ExtractionEvalCase(BaseModel):
    id: Annotated[str, Field(min_length=1)]
    extraction_path: Path
    expected_items: list[ExpectedExtractionItem] = Field(default_factory=list)
    timestamp_tolerance_seconds: Annotated[float, Field(ge=0)] = 5.0
    min_label_recall: Annotated[float, Field(ge=0, le=1)] = 1.0
    min_timestamp_hit_rate: Annotated[float, Field(ge=0, le=1)] = 1.0
    min_support_term_recall: Annotated[float, Field(ge=0, le=1)] = 0.75
    max_items: Annotated[int, Field(gt=0)] | None = None


class ExtractionEvalResult(BaseModel):
    id: str
    passed: bool
    extracted_items: int
    label_recall: float
    timestamp_hit_rate: float
    support_term_recall: float
    failures: list[str] = Field(default_factory=list)


class ExtractionEvalReport(BaseModel):
    passed: bool
    cases: int
    results: list[ExtractionEvalResult]

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")


def load_extraction_eval_cases(path: Path) -> list[ExtractionEvalCase]:
    cases: list[ExtractionEvalCase] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
        cases.append(ExtractionEvalCase.model_validate(payload))
    return cases


def run_extraction_eval_cases(
    cases: list[ExtractionEvalCase],
) -> ExtractionEvalReport:
    results = [evaluate_extraction_case(case) for case in cases]
    return ExtractionEvalReport(
        passed=all(result.passed for result in results),
        cases=len(results),
        results=results,
    )


def evaluate_extraction_case(case: ExtractionEvalCase) -> ExtractionEvalResult:
    extraction = _load_extraction(case.extraction_path)
    label_recall = _label_recall(case.expected_items, extraction)
    timestamp_hit_rate = _timestamp_hit_rate(
        expected_items=case.expected_items,
        extraction=extraction,
        tolerance_seconds=case.timestamp_tolerance_seconds,
    )
    support_recall = _support_term_recall(case.expected_items, extraction)
    failures = _failures(
        case=case,
        extraction=extraction,
        label_recall=label_recall,
        timestamp_hit_rate=timestamp_hit_rate,
        support_recall=support_recall,
    )
    return ExtractionEvalResult(
        id=case.id,
        passed=not failures,
        extracted_items=len(extraction.items),
        label_recall=label_recall,
        timestamp_hit_rate=timestamp_hit_rate,
        support_term_recall=support_recall,
        failures=failures,
    )


def render_extraction_eval_markdown(report: ExtractionEvalReport) -> str:
    lines = [
        "# Gist Structured Extraction Evaluation",
        "",
        f"- Cases: {report.cases}",
        f"- Passed: {'yes' if report.passed else 'no'}",
        "",
        "| Case | Status | Items | Label Recall | Timestamp Hit | Support Recall | Failures |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in report.results:
        failures = "; ".join(result.failures)
        lines.append(
            f"| {result.id} | {'pass' if result.passed else 'fail'} | "
            f"{result.extracted_items} | "
            f"{result.label_recall:.2f} | "
            f"{result.timestamp_hit_rate:.2f} | "
            f"{result.support_term_recall:.2f} | {failures} |"
        )
    return "\n".join(lines).strip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate timestamped structured extraction outputs."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv)

    report = run_extraction_eval_cases(load_extraction_eval_cases(args.dataset))
    if args.output is not None:
        report.write_json(args.output)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_extraction_eval_markdown(report))

    print(f"cases={report.cases}")
    print(f"passed={'yes' if report.passed else 'no'}")
    for result in report.results:
        status = "pass" if result.passed else "fail"
        print(
            f"{result.id}: {status}, items={result.extracted_items}, "
            f"label_recall={result.label_recall:.2f}, "
            f"timestamp_hit={result.timestamp_hit_rate:.2f}, "
            f"support_recall={result.support_term_recall:.2f}"
        )
        for failure in result.failures:
            print(f"  - {failure}")
    return 0 if report.passed else 1


def _load_extraction(path: Path) -> StructuredExtractionResponse:
    return StructuredExtractionResponse.model_validate(json.loads(path.read_text()))


def _label_recall(
    expected_items: list[ExpectedExtractionItem],
    extraction: StructuredExtractionResponse,
) -> float:
    if not expected_items:
        return 1.0
    extracted_labels = {item.label.lower() for item in extraction.items}
    hits = sum(1 for item in expected_items if item.label.lower() in extracted_labels)
    return hits / len(expected_items)


def _timestamp_hit_rate(
    expected_items: list[ExpectedExtractionItem],
    extraction: StructuredExtractionResponse,
    tolerance_seconds: float,
) -> float:
    ranged_items = [item for item in expected_items if item.time_range is not None]
    if not ranged_items:
        return 1.0
    hits = 0
    for expected in ranged_items:
        assert expected.time_range is not None
        if any(
            _item_range(item).overlaps(
                expected.time_range,
                tolerance_seconds=tolerance_seconds,
            )
            for item in extraction.items
            if item.label.lower() == expected.label.lower()
        ):
            hits += 1
    return hits / len(ranged_items)


def _support_term_recall(
    expected_items: list[ExpectedExtractionItem],
    extraction: StructuredExtractionResponse,
) -> float:
    expected_terms = [
        term.lower()
        for item in expected_items
        for term in item.support_terms
        if term.strip()
    ]
    if not expected_terms:
        return 1.0
    extracted_text = " ".join(
        f"{item.description} {item.support_text}" for item in extraction.items
    ).lower()
    hits = sum(1 for term in expected_terms if term in extracted_text)
    return hits / len(expected_terms)


def _failures(
    case: ExtractionEvalCase,
    extraction: StructuredExtractionResponse,
    label_recall: float,
    timestamp_hit_rate: float,
    support_recall: float,
) -> list[str]:
    failures: list[str] = []
    if label_recall < case.min_label_recall:
        failures.append(
            f"label recall {label_recall:.2f} below required {case.min_label_recall:.2f}"
        )
    if timestamp_hit_rate < case.min_timestamp_hit_rate:
        failures.append(
            "timestamp hit rate "
            f"{timestamp_hit_rate:.2f} below required {case.min_timestamp_hit_rate:.2f}"
        )
    if support_recall < case.min_support_term_recall:
        failures.append(
            "support term recall "
            f"{support_recall:.2f} below required {case.min_support_term_recall:.2f}"
        )
    if case.max_items is not None and len(extraction.items) > case.max_items:
        failures.append(f"extracted items {len(extraction.items)} exceeds limit {case.max_items}")
    return failures


def _item_range(item) -> TimeRange:
    return TimeRange(
        start_seconds=min(item.timestamp_start_seconds, item.timestamp_end_seconds),
        end_seconds=max(item.timestamp_start_seconds, item.timestamp_end_seconds),
    )


if __name__ == "__main__":
    raise SystemExit(main())
