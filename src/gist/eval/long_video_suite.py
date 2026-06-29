import argparse
import json
import re
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
    min_avg_token_reduction_percent: Annotated[float, Field(ge=0, le=100)] = 90.0
    max_noisy_transcript_warning_rate: Annotated[float, Field(ge=0, le=1)] = 0.25
    min_transcript_metadata_rate: Annotated[float, Field(ge=0, le=1)] = 0.8
    min_answered_rate: Annotated[float, Field(ge=0, le=1)] = 0.9
    max_avg_selected_evidence: Annotated[float, Field(gt=0)] = 8.0


class LongVideoSuiteGateResult(BaseModel):
    name: str
    passed: bool
    actual: float
    required: float
    message: str


class LongVideoHealthSummary(BaseModel):
    artifacts: int = 0
    avg_token_reduction_percent: float = 0.0
    noisy_transcript_warning_rate: float = 0.0
    transcript_metadata_rate: float = 0.0
    answered_rate: float = 0.0
    avg_selected_evidence: float = 0.0
    quality_warning_counts: dict[str, int] = Field(default_factory=dict)
    transcript_quality_counts: dict[str, int] = Field(default_factory=dict)


class LongVideoExpansionPlan(BaseModel):
    needed_cases: int = 0
    needed_long_video_cases: int = 0
    needed_distinct_videos: int = 0
    needed_distinct_domains: int = 0
    needed_by_category: dict[str, int] = Field(default_factory=dict)
    priority_actions: list[str] = Field(default_factory=list)


class LongVideoArtifactAuditItem(BaseModel):
    path: Path
    curated: bool
    candidate: bool
    reasons: list[str] = Field(default_factory=list)
    duration_seconds: float | None = None
    video_id: str | None = None
    query: str | None = None
    query_intent: str | None = None
    answer: str | None = None
    selected_evidence: int = 0
    visual_evidence: int = 0
    audio_evidence: int = 0
    token_reduction_percent: float | None = None
    quality_warnings: list[str] = Field(default_factory=list)


class LongVideoArtifactAuditReport(BaseModel):
    root: Path
    artifacts: int
    curated_artifacts: int
    candidate_artifacts: int
    rejected_artifacts: int
    items: list[LongVideoArtifactAuditItem]

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")


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
    health: LongVideoHealthSummary = Field(default_factory=LongVideoHealthSummary)
    expansion_plan: LongVideoExpansionPlan = Field(default_factory=LongVideoExpansionPlan)
    quality: QualityReport | None = None

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")


def audit_long_video_artifacts(
    root: Path,
    cases: list[QualityCase],
    gates: LongVideoSuiteGates,
) -> LongVideoArtifactAuditReport:
    curated_paths = {
        case.compression_path.resolve()
        for case in cases
        if case.compression_path is not None and case.compression_path.exists()
    }
    items = [
        _audit_artifact(path=path, curated_paths=curated_paths, gates=gates)
        for path in sorted(root.rglob("compression.json"))
    ]
    return LongVideoArtifactAuditReport(
        root=root,
        artifacts=len(items),
        curated_artifacts=sum(item.curated for item in items),
        candidate_artifacts=sum(item.candidate for item in items),
        rejected_artifacts=sum(not item.curated and not item.candidate for item in items),
        items=items,
    )


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
    artifact_payloads: list[dict] = []
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
        payload = _load_artifact(artifact)
        artifact_payloads.append(payload)
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

    health = _health_summary(artifact_payloads)
    if health.artifacts:
        gate_results.extend(
            [
                _at_least(
                    "avg_token_reduction_percent",
                    health.avg_token_reduction_percent,
                    gates.min_avg_token_reduction_percent,
                ),
                _at_most(
                    "noisy_transcript_warning_rate",
                    health.noisy_transcript_warning_rate,
                    gates.max_noisy_transcript_warning_rate,
                ),
                _at_least(
                    "transcript_metadata_rate",
                    health.transcript_metadata_rate,
                    gates.min_transcript_metadata_rate,
                ),
                _at_least("answered_rate", health.answered_rate, gates.min_answered_rate),
                _at_most(
                    "avg_selected_evidence",
                    health.avg_selected_evidence,
                    gates.max_avg_selected_evidence,
                ),
            ]
        )

    expansion_plan = _expansion_plan(
        case_count=len(cases),
        long_case_count=long_case_count,
        category_counts=category_counts,
        domain_counts=domain_counts,
        video_counts=video_counts,
        gates=gates,
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
        health=health,
        expansion_plan=expansion_plan,
        quality=quality,
    )


def _audit_artifact(
    path: Path,
    curated_paths: set[Path],
    gates: LongVideoSuiteGates,
) -> LongVideoArtifactAuditItem:
    reasons: list[str] = []
    curated = path.resolve() in curated_paths
    try:
        payload = _load_artifact(path)
    except (OSError, json.JSONDecodeError) as exc:
        return LongVideoArtifactAuditItem(
            path=path,
            curated=curated,
            candidate=False,
            reasons=[f"artifact could not be read: {exc}"],
        )

    compression = payload.get("compression", payload)
    ingestion = payload.get("ingestion")
    duration_seconds: float | None = None
    if ingestion is None:
        reasons.append("missing ingestion metadata")
    else:
        try:
            duration_seconds = float(ingestion["metadata"]["duration_seconds"])
        except (KeyError, TypeError, ValueError):
            reasons.append("missing duration metadata")

    metrics = compression.get("metrics") or {}
    selected = compression.get("selected") or []
    token_reduction = _optional_float(metrics.get("estimated_token_reduction_percent"))
    if curated:
        reasons.append("already curated")
    if duration_seconds is None or duration_seconds < gates.minimum_duration_seconds:
        actual = 0.0 if duration_seconds is None else duration_seconds
        reasons.append(
            f"duration {actual:.2f}s below {gates.minimum_duration_seconds:.2f}s"
        )
    if not str(compression.get("answer") or "").strip():
        reasons.append("missing answer")
    else:
        reasons.extend(_answer_audit_reasons(str(compression.get("answer"))))
    if not selected:
        reasons.append("no selected evidence")
    if token_reduction is None:
        reasons.append("missing token reduction metric")
    elif token_reduction < gates.min_avg_token_reduction_percent:
        reasons.append(
            f"token reduction {token_reduction:.2f}% below "
            f"{gates.min_avg_token_reduction_percent:.2f}%"
        )

    return LongVideoArtifactAuditItem(
        path=path,
        curated=curated,
        candidate=not curated and not reasons,
        reasons=reasons,
        duration_seconds=duration_seconds,
        video_id=_optional_string(compression.get("video_id")),
        query=_optional_string(compression.get("query")),
        query_intent=_optional_string(compression.get("query_intent")),
        answer=_optional_string(compression.get("answer")),
        selected_evidence=len(selected),
        visual_evidence=int(metrics.get("visual_selected") or 0),
        audio_evidence=int(metrics.get("audio_selected") or 0),
        token_reduction_percent=token_reduction,
        quality_warnings=[
            str(warning.get("code"))
            for warning in compression.get("quality_warnings", [])
            if warning.get("code")
        ],
    )


def _optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _answer_audit_reasons(answer: str) -> list[str]:
    normalized = " ".join(answer.strip().split())
    if not normalized:
        return ["missing answer"]
    if "could not derive a reliable answer" in normalized.lower():
        return ["unreliable generated answer"]
    if _looks_like_noisy_ocr_answer(normalized):
        return ["answer appears OCR-noisy"]
    return []


def _looks_like_noisy_ocr_answer(answer: str) -> bool:
    marker = "on-screen text near"
    if marker not in answer.lower():
        return False
    _, _, ocr_text = answer.partition(":")
    content_terms = [
        term.lower()
        for term in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", ocr_text)
        if term.lower() not in {"near", "seconds", "screen", "text"}
    ]
    if len(content_terms) < 2:
        return True
    short_terms = sum(len(term) <= 3 for term in content_terms)
    return short_terms / len(content_terms) > 0.6


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
    warning_rows = "\n".join(
        f"| {code} | {count} |"
        for code, count in report.health.quality_warning_counts.items()
    ) or "| none | 0 |"
    transcript_rows = "\n".join(
        f"| {quality} | {count} |"
        for quality, count in report.health.transcript_quality_counts.items()
    ) or "| none | 0 |"
    missing_category_rows = "\n".join(
        f"| {category} | {needed} |"
        for category, needed in report.expansion_plan.needed_by_category.items()
    ) or "| none | 0 |"
    priority_lines = "\n".join(
        f"- {action}" for action in report.expansion_plan.priority_actions
    ) or "- No expansion actions required by the configured gates."
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

## Run Health

- Artifacts: {report.health.artifacts}
- Avg token reduction: {report.health.avg_token_reduction_percent:.2f}%
- Noisy transcript warning rate: {report.health.noisy_transcript_warning_rate:.2%}
- Transcript metadata rate: {report.health.transcript_metadata_rate:.2%}
- Answered rate: {report.health.answered_rate:.2%}
- Avg selected evidence: {report.health.avg_selected_evidence:.2f}

### Quality Warnings

| Warning | Count |
|---|---:|
{warning_rows}

### Transcript Quality

| Quality | Count |
|---|---:|
{transcript_rows}

## Expansion Plan

- Additional cases needed: {report.expansion_plan.needed_cases}
- Additional long-video cases needed: {report.expansion_plan.needed_long_video_cases}
- Additional distinct videos needed: {report.expansion_plan.needed_distinct_videos}
- Additional distinct domains needed: {report.expansion_plan.needed_distinct_domains}

### Missing Query Categories

| Category | Additional Cases Needed |
|---|---:|
{missing_category_rows}

### Priority Actions

{priority_lines}

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
    warning_rows = "".join(
        f"<tr><td>{escape(code)}</td><td>{count}</td></tr>"
        for code, count in report.health.quality_warning_counts.items()
    ) or "<tr><td>none</td><td>0</td></tr>"
    transcript_rows = "".join(
        f"<tr><td>{escape(quality)}</td><td>{count}</td></tr>"
        for quality, count in report.health.transcript_quality_counts.items()
    ) or "<tr><td>none</td><td>0</td></tr>"
    missing_category_rows = "".join(
        f"<tr><td>{escape(category)}</td><td>{needed}</td></tr>"
        for category, needed in report.expansion_plan.needed_by_category.items()
    ) or "<tr><td>none</td><td>0</td></tr>"
    priority_items = (
        "".join(
            f"<li>{escape(action)}</li>"
            for action in report.expansion_plan.priority_actions
        )
        or "<li>No expansion actions required by the configured gates.</li>"
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
  <h2>Run Health</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Artifacts</td><td>{report.health.artifacts}</td></tr>
    <tr><td>Avg token reduction</td><td>{report.health.avg_token_reduction_percent:.2f}%</td></tr>
    <tr><td>Noisy transcript warning rate</td><td>
      {report.health.noisy_transcript_warning_rate:.2%}
    </td></tr>
    <tr><td>Transcript metadata rate</td><td>{report.health.transcript_metadata_rate:.2%}</td></tr>
    <tr><td>Answered rate</td><td>{report.health.answered_rate:.2%}</td></tr>
    <tr><td>Avg selected evidence</td><td>{report.health.avg_selected_evidence:.2f}</td></tr>
  </table>
  <h2>Quality Warnings</h2>
  <table>
    <tr><th>Warning</th><th>Count</th></tr>
    {warning_rows}
  </table>
  <h2>Transcript Quality</h2>
  <table>
    <tr><th>Quality</th><th>Count</th></tr>
    {transcript_rows}
  </table>
  <h2>Expansion Plan</h2>
  <table>
    <tr><th>Metric</th><th>Needed</th></tr>
    <tr><td>Additional cases</td><td>{report.expansion_plan.needed_cases}</td></tr>
    <tr><td>Additional long-video cases</td><td>{report.expansion_plan.needed_long_video_cases}</td></tr>
    <tr><td>Additional distinct videos</td><td>{report.expansion_plan.needed_distinct_videos}</td></tr>
    <tr><td>Additional distinct domains</td><td>{report.expansion_plan.needed_distinct_domains}</td></tr>
  </table>
  <h3>Missing Query Categories</h3>
  <table>
    <tr><th>Category</th><th>Additional Cases Needed</th></tr>
    {missing_category_rows}
  </table>
  <h3>Priority Actions</h3>
  <ul>{priority_items}</ul>
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
    parser.add_argument("--audit-root", type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path(".gist/long-video-suite"))
    parser.add_argument("--run-quality", action="store_true")
    parser.add_argument("--min-cases", type=int, default=30)
    parser.add_argument("--min-distinct-videos", type=int, default=5)
    parser.add_argument("--min-distinct-domains", type=int, default=3)
    parser.add_argument("--min-cases-per-category", type=int, default=3)
    parser.add_argument("--minimum-duration-seconds", type=float, default=3600.0)
    parser.add_argument("--min-quality-pass-rate", type=float, default=0.8)
    parser.add_argument("--min-avg-token-reduction-percent", type=float, default=90.0)
    parser.add_argument("--max-noisy-transcript-warning-rate", type=float, default=0.25)
    parser.add_argument("--min-transcript-metadata-rate", type=float, default=0.8)
    parser.add_argument("--min-answered-rate", type=float, default=0.9)
    parser.add_argument("--max-avg-selected-evidence", type=float, default=8.0)
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
            min_avg_token_reduction_percent=args.min_avg_token_reduction_percent,
            max_noisy_transcript_warning_rate=args.max_noisy_transcript_warning_rate,
            min_transcript_metadata_rate=args.min_transcript_metadata_rate,
            min_answered_rate=args.min_answered_rate,
            max_avg_selected_evidence=args.max_avg_selected_evidence,
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
    if args.audit_root is not None or args.audit_output is not None:
        audit = audit_long_video_artifacts(
            root=args.audit_root or Path(".gist/runs"),
            cases=cases,
            gates=report.gates,
        )
        if args.audit_output is not None:
            audit.write_json(args.audit_output)
        print(f"audit_artifacts={audit.artifacts}")
        print(f"audit_candidates={audit.candidate_artifacts}")
        print(f"audit_rejected={audit.rejected_artifacts}")

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
    payload = _load_artifact(path)
    compression = payload.get("compression", payload)
    ingestion = payload.get("ingestion")
    if ingestion is None:
        raise ValueError(f"{path}: ingestion metadata is required")
    return (
        float(ingestion["metadata"]["duration_seconds"]),
        str(compression["video_id"]),
    )


def _load_artifact(path: Path) -> dict:
    return json.loads(path.read_text())


def _health_summary(payloads: list[dict]) -> LongVideoHealthSummary:
    if not payloads:
        return LongVideoHealthSummary()

    warning_counts: Counter[str] = Counter()
    transcript_quality_counts: Counter[str] = Counter()
    token_reductions: list[float] = []
    selected_counts: list[int] = []
    noisy_count = 0
    metadata_count = 0
    answered_count = 0

    for payload in payloads:
        compression = payload.get("compression", payload)
        metrics = compression.get("metrics", {})
        token_reductions.append(float(metrics.get("estimated_token_reduction_percent", 0.0)))
        selected = compression.get("selected") or []
        selected_counts.append(int(metrics.get("selected_candidates", len(selected))))
        answer = str(compression.get("answer") or "").strip()
        if len(answer.split()) >= 3:
            answered_count += 1

        warnings = compression.get("quality_warnings") or []
        warning_codes = [
            str(warning.get("code"))
            for warning in warnings
            if isinstance(warning, dict) and warning.get("code")
        ]
        warning_counts.update(warning_codes)
        if "noisy_transcript_evidence" in warning_codes:
            noisy_count += 1

        transcript_metadata = compression.get("transcript_metadata")
        if isinstance(transcript_metadata, dict) and transcript_metadata:
            metadata_count += 1
            quality = str(transcript_metadata.get("quality") or "unknown")
            transcript_quality_counts[quality] += 1

    total = len(payloads)
    return LongVideoHealthSummary(
        artifacts=total,
        avg_token_reduction_percent=_average(token_reductions),
        noisy_transcript_warning_rate=noisy_count / total,
        transcript_metadata_rate=metadata_count / total,
        answered_rate=answered_count / total,
        avg_selected_evidence=_average(selected_counts),
        quality_warning_counts=dict(sorted(warning_counts.items())),
        transcript_quality_counts=dict(sorted(transcript_quality_counts.items())),
    )


def _expansion_plan(
    *,
    case_count: int,
    long_case_count: int,
    category_counts: Counter[str],
    domain_counts: Counter[str],
    video_counts: Counter[str],
    gates: LongVideoSuiteGates,
) -> LongVideoExpansionPlan:
    needed_by_category = {
        category.value: gates.min_cases_per_category - category_counts[category.value]
        for category in REQUIRED_QUERY_CATEGORIES
        if category_counts[category.value] < gates.min_cases_per_category
    }
    plan = LongVideoExpansionPlan(
        needed_cases=max(0, gates.min_cases - case_count),
        needed_long_video_cases=max(0, gates.min_cases - long_case_count),
        needed_distinct_videos=max(0, gates.min_distinct_videos - len(video_counts)),
        needed_distinct_domains=max(0, gates.min_distinct_domains - len(domain_counts)),
        needed_by_category=needed_by_category,
    )
    plan.priority_actions = _priority_actions(plan)
    return plan


def _priority_actions(plan: LongVideoExpansionPlan) -> list[str]:
    actions: list[str] = []
    if plan.needed_cases:
        actions.append(f"Curate {plan.needed_cases} more verified long-video question cases.")
    if plan.needed_long_video_cases:
        actions.append(
            f"Ensure {plan.needed_long_video_cases} added cases come from videos at least 60 minutes long."
        )
    if plan.needed_distinct_videos:
        actions.append(
            f"Add cases from {plan.needed_distinct_videos} new long-video source(s)."
        )
    if plan.needed_distinct_domains:
        actions.append(
            f"Add at least {plan.needed_distinct_domains} new domain label(s) beyond the current set."
        )
    for category, needed in plan.needed_by_category.items():
        actions.append(f"Add {needed} verified `{category}` case(s).")
    return actions


def _average(values: list[float] | list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


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
