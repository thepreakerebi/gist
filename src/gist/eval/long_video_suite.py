import argparse
import json
from collections import Counter
from html import escape
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

from gist.core.query_intent import QueryIntent
from gist.eval.quality import (
    QualityCase,
    QualityReport,
    load_quality_cases,
    render_quality_html,
    render_quality_markdown,
    run_quality_cases,
)


REQUIRED_QUERY_CATEGORIES = (
    QueryIntent.SPEECH_SEMANTIC,
    QueryIntent.VISUAL_OBJECT_ACTION,
    QueryIntent.TEMPORAL_BEFORE_AFTER,
    QueryIntent.GLOBAL_SUMMARY,
    QueryIntent.MIXED_AV,
)


class LongVideoSuiteGates(BaseModel):
    min_cases: Annotated[int, Field(ge=1)] = 30
    min_distinct_videos: Annotated[int, Field(ge=1)] = 5
    min_distinct_domains: Annotated[int, Field(ge=1)] = 3
    min_cases_per_category: Annotated[int, Field(ge=1)] = 3
    minimum_duration_seconds: Annotated[float, Field(gt=0)] = 3600.0
    min_quality_pass_rate: Annotated[float, Field(ge=0, le=1)] = 0.8


class LongVideoSuiteGateResult(BaseModel):
    name: str
    passed: bool
    actual: float
    required: float
    message: str


class LongVideoSuiteReport(BaseModel):
    passed: bool
    gates: LongVideoSuiteGates
    gate_results: list[LongVideoSuiteGateResult]
    case_count: int
    long_video_case_count: int
    category_counts: dict[str, int]
    domain_counts: dict[str, int]
    video_counts: dict[str, int]
    missing_artifacts: list[Path] = Field(default_factory=list)
    metadata_failures: list[str] = Field(default_factory=list)
    quality: QualityReport | None = None

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")


def evaluate_long_video_suite(
    cases: list[QualityCase],
    gates: LongVideoSuiteGates,
    quality: QualityReport | None = None,
) -> LongVideoSuiteReport:
    category_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    video_counts: Counter[str] = Counter()
    missing_artifacts: list[Path] = []
    metadata_failures: list[str] = []
    long_case_count = 0

    for case in cases:
        if case.query_category is None:
            metadata_failures.append(f"{case.id}: query_category is required")
        else:
            category_counts[case.query_category.value] += 1
        if not case.domain or not case.domain.strip():
            metadata_failures.append(f"{case.id}: domain is required")
        else:
            domain_counts[case.domain.strip().lower()] += 1

        artifact = case.compression_path
        if artifact is None or not artifact.exists():
            if artifact is not None:
                missing_artifacts.append(artifact)
            else:
                metadata_failures.append(
                    f"{case.id}: compression_path is required for curated suite readiness"
                )
            continue
        duration_seconds, video_id = _artifact_identity(artifact)
        video_counts[video_id] += 1
        if duration_seconds >= gates.minimum_duration_seconds:
            long_case_count += 1
        else:
            metadata_failures.append(
                f"{case.id}: duration {duration_seconds:.2f}s is below "
                f"{gates.minimum_duration_seconds:.2f}s"
            )

    gate_results = [
        _at_least("cases", len(cases), gates.min_cases),
        _at_least("long_video_cases", long_case_count, gates.min_cases),
        _at_least("distinct_videos", len(video_counts), gates.min_distinct_videos),
        _at_least("distinct_domains", len(domain_counts), gates.min_distinct_domains),
    ]
    gate_results.extend(
        _at_least(
            f"category_{category.value}",
            category_counts[category.value],
            gates.min_cases_per_category,
        )
        for category in REQUIRED_QUERY_CATEGORIES
    )
    gate_results.append(
        _at_most("missing_artifacts", len(missing_artifacts), 0)
    )
    gate_results.append(
        _at_most("metadata_failures", len(metadata_failures), 0)
    )
    if quality is not None:
        gate_results.append(
            _at_least(
                "quality_pass_rate",
                quality.summary.pass_rate,
                gates.min_quality_pass_rate,
            )
        )

    return LongVideoSuiteReport(
        passed=all(result.passed for result in gate_results),
        gates=gates,
        gate_results=gate_results,
        case_count=len(cases),
        long_video_case_count=long_case_count,
        category_counts=dict(sorted(category_counts.items())),
        domain_counts=dict(sorted(domain_counts.items())),
        video_counts=dict(sorted(video_counts.items())),
        missing_artifacts=missing_artifacts,
        metadata_failures=metadata_failures,
        quality=quality,
    )


def render_long_video_suite_markdown(report: LongVideoSuiteReport) -> str:
    gate_rows = "\n".join(
        f"| {result.name} | {'pass' if result.passed else 'fail'} | "
        f"{result.actual:.2f} | {result.required:.2f} |"
        for result in report.gate_results
    )
    category_rows = "\n".join(
        f"| {category.value} | {report.category_counts.get(category.value, 0)} | "
        f"{report.gates.min_cases_per_category} |"
        for category in REQUIRED_QUERY_CATEGORIES
    )
    quality_detail = (
        f"\n## Quality Detail\n\n{render_quality_markdown(report.quality)}"
        if report.quality is not None
        else ""
    )
    failures = [
        *(f"Missing artifact: `{path}`" for path in report.missing_artifacts),
        *report.metadata_failures,
    ]
    failure_lines = "\n".join(f"- {failure}" for failure in failures) or "- None"
    return f"""# Gist Long-Video Suite Readiness

- Passed: {"yes" if report.passed else "no"}
- Cases: {report.case_count}
- Valid long-video cases: {report.long_video_case_count}
- Distinct videos: {len(report.video_counts)}
- Distinct domains: {len(report.domain_counts)}

## Gates

| Gate | Status | Actual | Required |
|---|---:|---:|---:|
{gate_rows}

## Query Coverage

| Category | Cases | Required |
|---|---:|---:|
{category_rows}

## Dataset Problems

{failure_lines}
{quality_detail}
"""


def render_long_video_suite_html(report: LongVideoSuiteReport) -> str:
    gate_rows = "\n".join(
        "<tr>"
        f"<td>{escape(result.name)}</td>"
        f"<td>{'pass' if result.passed else 'fail'}</td>"
        f"<td>{result.actual:.2f}</td>"
        f"<td>{result.required:.2f}</td>"
        "</tr>"
        for result in report.gate_results
    )
    category_rows = "\n".join(
        "<tr>"
        f"<td>{escape(category.value)}</td>"
        f"<td>{report.category_counts.get(category.value, 0)}</td>"
        f"<td>{report.gates.min_cases_per_category}</td>"
        "</tr>"
        for category in REQUIRED_QUERY_CATEGORIES
    )
    problems = [
        *(f"Missing artifact: {path}" for path in report.missing_artifacts),
        *report.metadata_failures,
    ]
    problem_items = (
        "".join(f"<li>{escape(problem)}</li>" for problem in problems)
        or "<li>None</li>"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Gist Long-Video Suite Readiness</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 32px; color: #172026; }}
    table {{ border-collapse: collapse; width: min(920px, 100%); margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d7dfdf; padding: 8px 10px; text-align: left; }}
    th {{ background: #edf5f3; }}
  </style>
</head>
<body>
  <h1>Gist Long-Video Suite Readiness</h1>
  <p><strong>Status:</strong> {'pass' if report.passed else 'fail'}</p>
  <h2>Gates</h2>
  <table>
    <tr><th>Gate</th><th>Status</th><th>Actual</th><th>Required</th></tr>
    {gate_rows}
  </table>
  <h2>Query Coverage</h2>
  <table>
    <tr><th>Category</th><th>Cases</th><th>Required</th></tr>
    {category_rows}
  </table>
  <h2>Dataset Problems</h2>
  <ul>{problem_items}</ul>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check coverage and optionally run quality for the curated long-video suite."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--html-output", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path(".gist/long-video-suite"))
    parser.add_argument("--run-quality", action="store_true")
    parser.add_argument("--min-cases", type=int, default=30)
    parser.add_argument("--min-distinct-videos", type=int, default=5)
    parser.add_argument("--min-distinct-domains", type=int, default=3)
    parser.add_argument("--min-cases-per-category", type=int, default=3)
    parser.add_argument("--minimum-duration-seconds", type=float, default=3600.0)
    parser.add_argument("--min-quality-pass-rate", type=float, default=0.8)
    args = parser.parse_args(argv)

    cases = load_quality_cases(args.dataset)
    quality = (
        run_quality_cases(cases, output_root=args.output_root)
        if args.run_quality
        else None
    )
    report = evaluate_long_video_suite(
        cases=cases,
        gates=LongVideoSuiteGates(
            min_cases=args.min_cases,
            min_distinct_videos=args.min_distinct_videos,
            min_distinct_domains=args.min_distinct_domains,
            min_cases_per_category=args.min_cases_per_category,
            minimum_duration_seconds=args.minimum_duration_seconds,
            min_quality_pass_rate=args.min_quality_pass_rate,
        ),
        quality=quality,
    )
    if args.output is not None:
        report.write_json(args.output)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_long_video_suite_markdown(report))
    if args.html_output is not None:
        args.html_output.parent.mkdir(parents=True, exist_ok=True)
        args.html_output.write_text(render_long_video_suite_html(report))

    print(f"passed={'yes' if report.passed else 'no'}")
    print(f"cases={len(cases)}")
    print(f"distinct_videos={len(report.video_counts)}")
    print(f"distinct_domains={len(report.domain_counts)}")
    for category in REQUIRED_QUERY_CATEGORIES:
        print(f"{category.value}={report.category_counts.get(category.value, 0)}")
    for result in report.gate_results:
        if not result.passed:
            print(
                f"  - {result.name}: actual={result.actual:.2f}, "
                f"required={result.required:.2f}"
            )
    return 0 if report.passed else 1


def _artifact_identity(path: Path) -> tuple[float, str]:
    payload = json.loads(path.read_text())
    compression = payload.get("compression", payload)
    ingestion = payload.get("ingestion")
    if ingestion is None:
        raise ValueError(f"{path}: ingestion metadata is required")
    return (
        float(ingestion["metadata"]["duration_seconds"]),
        str(compression["video_id"]),
    )


def _at_least(
    name: str,
    actual: float,
    required: float,
) -> LongVideoSuiteGateResult:
    passed = actual >= required
    return LongVideoSuiteGateResult(
        name=name,
        passed=passed,
        actual=actual,
        required=required,
        message=f"{actual:.2f} >= {required:.2f}" if passed else f"{actual:.2f} < {required:.2f}",
    )


def _at_most(
    name: str,
    actual: float,
    required: float,
) -> LongVideoSuiteGateResult:
    passed = actual <= required
    return LongVideoSuiteGateResult(
        name=name,
        passed=passed,
        actual=actual,
        required=required,
        message=f"{actual:.2f} <= {required:.2f}" if passed else f"{actual:.2f} > {required:.2f}",
    )


if __name__ == "__main__":
    raise SystemExit(main())
