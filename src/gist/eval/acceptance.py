import argparse
from html import escape
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

from gist.eval.quality import (
    QualityReport,
    load_quality_cases,
    render_quality_html,
    render_quality_markdown,
    run_quality_cases,
)


class AcceptanceGates(BaseModel):
    min_cases: Annotated[int, Field(ge=1)] = 1
    min_pass_rate: Annotated[float, Field(ge=0, le=1)] = 0.9
    min_avg_answer_term_recall: Annotated[float, Field(ge=0, le=1)] = 0.8
    min_avg_evidence_relevance_rate: Annotated[float, Field(ge=0, le=1)] = 0.8
    min_avg_timestamp_hit_rate: Annotated[float, Field(ge=0, le=1)] = 0.8
    min_avg_token_reduction_percent: Annotated[float, Field(ge=0, le=100)] = 90.0
    max_failure_count: Annotated[int, Field(ge=0)] = 0


class AcceptanceGateResult(BaseModel):
    name: str
    passed: bool
    actual: float
    required: float
    message: str


class AcceptanceReport(BaseModel):
    passed: bool
    gates: AcceptanceGates
    gate_results: list[AcceptanceGateResult]
    quality: QualityReport

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")


def evaluate_acceptance(quality: QualityReport, gates: AcceptanceGates) -> AcceptanceReport:
    failure_count = quality.summary.cases - quality.summary.passed
    gate_results = [
        _at_least("cases", quality.summary.cases, gates.min_cases),
        _at_least("pass_rate", quality.summary.pass_rate, gates.min_pass_rate),
        _at_least(
            "avg_answer_term_recall",
            quality.summary.avg_answer_term_recall,
            gates.min_avg_answer_term_recall,
        ),
        _at_least(
            "avg_evidence_relevance_rate",
            quality.summary.avg_evidence_relevance_rate,
            gates.min_avg_evidence_relevance_rate,
        ),
        _at_least(
            "avg_timestamp_hit_rate",
            quality.summary.avg_timestamp_hit_rate,
            gates.min_avg_timestamp_hit_rate,
        ),
        _at_least(
            "avg_token_reduction_percent",
            quality.summary.avg_token_reduction_percent,
            gates.min_avg_token_reduction_percent,
        ),
        _at_most("failure_count", failure_count, gates.max_failure_count),
    ]
    return AcceptanceReport(
        passed=all(result.passed for result in gate_results),
        gates=gates,
        gate_results=gate_results,
        quality=quality,
    )


def render_acceptance_markdown(report: AcceptanceReport) -> str:
    gate_rows = "\n".join(
        f"| {result.name} | {'pass' if result.passed else 'fail'} | "
        f"{result.actual:.2f} | {result.required:.2f} | {result.message} |"
        for result in report.gate_results
    )
    return f"""# Gist Acceptance Report

- Passed: {"yes" if report.passed else "no"}
- Cases: {report.quality.summary.cases}
- Quality pass rate: {report.quality.summary.pass_rate:.2%}
- Avg answer recall: {report.quality.summary.avg_answer_term_recall:.2f}
- Avg evidence relevance: {report.quality.summary.avg_evidence_relevance_rate:.2f}
- Avg timestamp hit: {report.quality.summary.avg_timestamp_hit_rate:.2f}
- Avg token reduction: {report.quality.summary.avg_token_reduction_percent:.2f}%

## Gates

| Gate | Status | Actual | Required | Message |
|---|---:|---:|---:|---|
{gate_rows}

## Quality Detail

{render_quality_markdown(report.quality)}
"""


def render_acceptance_html(report: AcceptanceReport) -> str:
    gate_rows = "\n".join(
        "<tr>"
        f"<td>{escape(result.name)}</td>"
        f"<td>{'pass' if result.passed else 'fail'}</td>"
        f"<td>{result.actual:.2f}</td>"
        f"<td>{result.required:.2f}</td>"
        f"<td>{escape(result.message)}</td>"
        "</tr>"
        for result in report.gate_results
    )
    quality_html = render_quality_html(report.quality)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Gist Acceptance Report</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 32px; color: #172026; }}
    h1, h2 {{ color: #0f2f2f; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d7dfdf; padding: 8px 10px; text-align: left; }}
    th {{ background: #edf5f3; }}
    .metric {{ display: inline-block; margin: 0 16px 12px 0; }}
    .status {{ font-size: 24px; font-weight: 800; }}
  </style>
</head>
<body>
  <h1>Gist Acceptance Report</h1>
  <p class="status">{"PASSED" if report.passed else "FAILED"}</p>
  <div class="metric"><strong>Cases:</strong> {report.quality.summary.cases}</div>
  <div class="metric"><strong>Pass rate:</strong> {report.quality.summary.pass_rate:.2%}</div>
  <div class="metric">
    <strong>Avg answer recall:</strong> {report.quality.summary.avg_answer_term_recall:.2f}
  </div>
  <div class="metric">
    <strong>Avg evidence relevance:</strong>
    {report.quality.summary.avg_evidence_relevance_rate:.2f}
  </div>
  <div class="metric">
    <strong>Avg timestamp hit:</strong> {report.quality.summary.avg_timestamp_hit_rate:.2f}
  </div>
  <div class="metric">
    <strong>Avg token reduction:</strong> {report.quality.summary.avg_token_reduction_percent:.2f}%
  </div>
  <h2>Gates</h2>
  <table>
    <thead>
      <tr><th>Gate</th><th>Status</th><th>Actual</th><th>Required</th><th>Message</th></tr>
    </thead>
    <tbody>{gate_rows}</tbody>
  </table>
  <h2>Quality Detail</h2>
  {quality_html}
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Gist acceptance gates against a curated quality dataset."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--html-output", type=Path)
    parser.add_argument("--quality-output", type=Path)
    parser.add_argument("--quality-markdown-output", type=Path)
    parser.add_argument("--quality-html-output", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path(".gist/acceptance"))
    parser.add_argument("--min-cases", type=int, default=1)
    parser.add_argument("--min-pass-rate", type=float, default=0.9)
    parser.add_argument("--min-avg-answer-term-recall", type=float, default=0.8)
    parser.add_argument("--min-avg-evidence-relevance-rate", type=float, default=0.8)
    parser.add_argument("--min-avg-timestamp-hit-rate", type=float, default=0.8)
    parser.add_argument("--min-avg-token-reduction-percent", type=float, default=90.0)
    parser.add_argument("--max-failure-count", type=int, default=0)
    args = parser.parse_args(argv)

    gates = AcceptanceGates(
        min_cases=args.min_cases,
        min_pass_rate=args.min_pass_rate,
        min_avg_answer_term_recall=args.min_avg_answer_term_recall,
        min_avg_evidence_relevance_rate=args.min_avg_evidence_relevance_rate,
        min_avg_timestamp_hit_rate=args.min_avg_timestamp_hit_rate,
        min_avg_token_reduction_percent=args.min_avg_token_reduction_percent,
        max_failure_count=args.max_failure_count,
    )
    quality = run_quality_cases(load_quality_cases(args.dataset), output_root=args.output_root)
    report = evaluate_acceptance(quality, gates)

    if args.output is not None:
        report.write_json(args.output)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_acceptance_markdown(report))
    if args.html_output is not None:
        args.html_output.parent.mkdir(parents=True, exist_ok=True)
        args.html_output.write_text(render_acceptance_html(report))
    if args.quality_output is not None:
        quality.write_json(args.quality_output)
    if args.quality_markdown_output is not None:
        args.quality_markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.quality_markdown_output.write_text(render_quality_markdown(quality))
    if args.quality_html_output is not None:
        args.quality_html_output.parent.mkdir(parents=True, exist_ok=True)
        args.quality_html_output.write_text(render_quality_html(quality))

    print(f"passed={'yes' if report.passed else 'no'}")
    print(f"cases={quality.summary.cases}")
    print(f"quality_pass_rate={quality.summary.pass_rate:.2%}")
    for gate_result in report.gate_results:
        status = "pass" if gate_result.passed else "fail"
        print(
            f"{gate_result.name}: {status}, actual={gate_result.actual:.2f}, "
            f"required={gate_result.required:.2f}"
        )
    return 0 if report.passed else 1


def _at_least(name: str, actual: float, required: float) -> AcceptanceGateResult:
    passed = actual >= required
    return AcceptanceGateResult(
        name=name,
        passed=passed,
        actual=actual,
        required=required,
        message=f"{actual:.2f} >= {required:.2f}" if passed else f"{actual:.2f} < {required:.2f}",
    )


def _at_most(name: str, actual: float, required: float) -> AcceptanceGateResult:
    passed = actual <= required
    return AcceptanceGateResult(
        name=name,
        passed=passed,
        actual=actual,
        required=required,
        message=f"{actual:.2f} <= {required:.2f}" if passed else f"{actual:.2f} > {required:.2f}",
    )


if __name__ == "__main__":
    raise SystemExit(main())
