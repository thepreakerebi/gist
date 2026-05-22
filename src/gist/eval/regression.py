import argparse
import json
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

from gist.core.schemas import CompressionResponse


class TimeRange(BaseModel):
    start_seconds: Annotated[float, Field(ge=0)]
    end_seconds: Annotated[float, Field(ge=0)]

    def overlaps(self, other: "TimeRange", tolerance_seconds: float = 0.0) -> bool:
        if tolerance_seconds < 0:
            raise ValueError("tolerance_seconds must be non-negative")
        return (
            self.start_seconds <= other.end_seconds + tolerance_seconds
            and other.start_seconds <= self.end_seconds + tolerance_seconds
        )


class RegressionCase(BaseModel):
    id: Annotated[str, Field(min_length=1)]
    compression_path: Path
    expected_evidence_ranges: list[TimeRange] = Field(default_factory=list)
    timestamp_tolerance_seconds: Annotated[float, Field(ge=0)] = 5.0
    min_timestamp_hit_rate: Annotated[float, Field(ge=0, le=1)] = 1.0
    min_token_reduction_percent: Annotated[float, Field(ge=0, le=100)] = 0.0
    max_selected_evidence: Annotated[int, Field(gt=0)] | None = None
    required_answer_terms: list[str] = Field(default_factory=list)


class RegressionResult(BaseModel):
    id: str
    passed: bool
    timestamp_hit_rate: float
    token_reduction_percent: float
    selected_evidence: int
    failures: list[str] = Field(default_factory=list)


class RegressionReport(BaseModel):
    passed: bool
    cases: int
    results: list[RegressionResult]

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")


def load_regression_cases(path: Path) -> list[RegressionCase]:
    cases: list[RegressionCase] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
        cases.append(RegressionCase.model_validate(payload))
    return cases


def run_regression_cases(cases: list[RegressionCase]) -> RegressionReport:
    results = [evaluate_case(case) for case in cases]
    return RegressionReport(
        passed=all(result.passed for result in results),
        cases=len(results),
        results=results,
    )


def evaluate_case(case: RegressionCase) -> RegressionResult:
    compression = _load_compression(case.compression_path)
    selected_ranges = _selected_evidence_ranges(compression)
    hit_rate = _range_hit_rate(
        selected_ranges=selected_ranges,
        expected_ranges=case.expected_evidence_ranges,
        tolerance_seconds=case.timestamp_tolerance_seconds,
    )
    token_reduction = compression.metrics.estimated_token_reduction_percent
    failures: list[str] = []

    if hit_rate < case.min_timestamp_hit_rate:
        failures.append(
            "timestamp hit rate "
            f"{hit_rate:.2f} is below required {case.min_timestamp_hit_rate:.2f}"
        )
    if token_reduction < case.min_token_reduction_percent:
        failures.append(
            "token reduction "
            f"{token_reduction:.2f}% is below required {case.min_token_reduction_percent:.2f}%"
        )
    if (
        case.max_selected_evidence is not None
        and compression.metrics.selected_candidates > case.max_selected_evidence
    ):
        failures.append(
            "selected evidence "
            f"{compression.metrics.selected_candidates} exceeds limit {case.max_selected_evidence}"
        )

    answer = (compression.answer or "").lower()
    missing_terms = [
        term for term in case.required_answer_terms if term.lower() not in answer
    ]
    if missing_terms:
        failures.append(f"answer missing required terms: {', '.join(missing_terms)}")

    return RegressionResult(
        id=case.id,
        passed=not failures,
        timestamp_hit_rate=hit_rate,
        token_reduction_percent=token_reduction,
        selected_evidence=compression.metrics.selected_candidates,
        failures=failures,
    )


def render_regression_markdown(report: RegressionReport) -> str:
    lines = [
        "# Gist Regression Report",
        "",
        f"- Cases: {report.cases}",
        f"- Passed: {'yes' if report.passed else 'no'}",
        "",
        "| Case | Status | Timestamp Hit Rate | Token Reduction | Selected | Failures |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for result in report.results:
        failures = "; ".join(result.failures) if result.failures else ""
        lines.append(
            f"| {result.id} | {'pass' if result.passed else 'fail'} | "
            f"{result.timestamp_hit_rate:.2f} | "
            f"{result.token_reduction_percent:.2f}% | "
            f"{result.selected_evidence} | {failures} |"
        )
    return "\n".join(lines).strip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run local Gist regression checks against generated compression reports."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv)

    report = run_regression_cases(load_regression_cases(args.dataset))
    if args.output is not None:
        report.write_json(args.output)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_regression_markdown(report))

    print(f"cases={report.cases}")
    print(f"passed={'yes' if report.passed else 'no'}")
    for result in report.results:
        status = "pass" if result.passed else "fail"
        print(
            f"{result.id}: {status}, hit_rate={result.timestamp_hit_rate:.2f}, "
            f"token_reduction={result.token_reduction_percent:.2f}%, "
            f"selected={result.selected_evidence}"
        )
        for failure in result.failures:
            print(f"  - {failure}")
    return 0 if report.passed else 1


def _load_compression(path: Path) -> CompressionResponse:
    payload = json.loads(path.read_text())
    compression_payload = payload.get("compression", payload)
    return CompressionResponse.model_validate(compression_payload)


def _selected_evidence_ranges(compression: CompressionResponse) -> list[TimeRange]:
    ranges: list[TimeRange] = []
    for item in compression.selected:
        start_seconds = (
            item.clip_start_seconds
            if item.clip_start_seconds is not None
            else item.scene_start_seconds
        )
        end_seconds = (
            item.clip_end_seconds
            if item.clip_end_seconds is not None
            else item.scene_end_seconds
        )
        if start_seconds is None or end_seconds is None:
            start_seconds = item.timestamp_seconds
            end_seconds = item.timestamp_seconds
        ranges.append(
            TimeRange(
                start_seconds=min(start_seconds, end_seconds),
                end_seconds=max(start_seconds, end_seconds),
            )
        )
    return ranges


def _range_hit_rate(
    selected_ranges: list[TimeRange],
    expected_ranges: list[TimeRange],
    tolerance_seconds: float,
) -> float:
    if not expected_ranges:
        return 1.0
    hits = sum(
        1
        for expected in expected_ranges
        if any(
            selected.overlaps(expected, tolerance_seconds=tolerance_seconds)
            for selected in selected_ranges
        )
    )
    return hits / len(expected_ranges)


if __name__ == "__main__":
    raise SystemExit(main())
