import argparse
import json
import re
import sys
from html import escape
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from gist.core.answering import answer_from_evidence, verify_answer_claims
from gist.core.evidence_pruning import annotate_evidence_support
from gist.core.modes import AudioScoringMode, VisualScoringMode
from gist.core.presets import CompressionPreset
from gist.core.schemas import CompressionResponse, Modality
from gist.eval.regression import TimeRange
from gist.media.longform import ProcessingMode
from gist.pipeline import LocalCompressionPipeline


class QualityCase(BaseModel):
    id: Annotated[str, Field(min_length=1)]
    query: Annotated[str, Field(min_length=1)] | None = None
    compression_path: Path | None = None
    video_path: Path | None = None
    preset: CompressionPreset = CompressionPreset.BALANCED
    processing_mode: ProcessingMode = ProcessingMode.AUTO
    sample_count: Annotated[int, Field(gt=0)] | None = None
    audio_window_seconds: Annotated[float, Field(gt=0)] | None = None
    visual_scorer: VisualScoringMode = VisualScoringMode.CLIP_SCENE
    audio_scorer: AudioScoringMode = AudioScoringMode.BASELINE
    adaptive_budget: bool = True
    decompose_query: bool = True
    visual_ocr: bool = True
    relevant_timestamps: list[float] = Field(default_factory=list)
    relevant_ranges: list[TimeRange] = Field(default_factory=list)
    timestamp_tolerance_seconds: Annotated[float, Field(ge=0)] = 5.0
    expected_answer_terms: list[str] = Field(default_factory=list)
    expected_evidence_terms: list[str] = Field(default_factory=list)
    min_answer_term_recall: Annotated[float, Field(ge=0, le=1)] = 0.0
    min_evidence_term_coverage: Annotated[float, Field(ge=0, le=1)] = 0.0
    min_evidence_relevance_rate: Annotated[float, Field(ge=0, le=1)] = 0.0
    min_timestamp_hit_rate: Annotated[float, Field(ge=0, le=1)] = 0.0
    min_token_reduction_percent: Annotated[float, Field(ge=0, le=100)] = 0.0
    max_selected_evidence: Annotated[int, Field(gt=0)] | None = None
    min_visual_evidence: Annotated[int, Field(ge=0)] = 0
    min_audio_evidence: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def validate_source(self) -> "QualityCase":
        if self.compression_path is None and self.video_path is None:
            raise ValueError("either compression_path or video_path is required")
        if self.video_path is not None and self.query is None:
            raise ValueError("query is required when video_path is used")
        return self


class QualityResult(BaseModel):
    id: str
    passed: bool
    query: str
    answer: str | None
    answer_term_recall: float
    evidence_term_coverage: float
    evidence_relevance_rate: float
    timestamp_hit_rate: float
    token_reduction_percent: float
    selected_evidence: int
    visual_evidence: int
    audio_evidence: int
    failure_categories: list[str] = Field(default_factory=list)
    recommendation: str | None = None
    failures: list[str] = Field(default_factory=list)


class QualitySummary(BaseModel):
    cases: int
    passed: int
    pass_rate: float
    avg_answer_term_recall: float
    avg_evidence_term_coverage: float
    avg_evidence_relevance_rate: float
    avg_timestamp_hit_rate: float
    avg_token_reduction_percent: float
    failure_categories: dict[str, int] = Field(default_factory=dict)


class QualityReport(BaseModel):
    passed: bool
    summary: QualitySummary
    results: list[QualityResult]

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")


class QualityDatasetCheck(BaseModel):
    cases: int
    runnable_cases: int
    replay_cases: int
    warnings: list[str] = Field(default_factory=list)


class QualityCaseDraft(BaseModel):
    case: QualityCase
    notes: list[str] = Field(default_factory=list)


def load_quality_cases(path: Path) -> list[QualityCase]:
    cases: list[QualityCase] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
        cases.append(QualityCase.model_validate(payload))
    return cases


def draft_quality_case(
    compression_path: Path,
    case_id: str | None = None,
    min_token_reduction_percent: float = 90.0,
    timestamp_tolerance_seconds: float = 8.0,
    max_selected_evidence: int | None = None,
) -> QualityCaseDraft:
    compression = _load_compression(compression_path)
    evidence_ranges = _selected_ranges(compression)
    answer_terms = _keyword_terms(compression.answer or "")
    evidence_terms = _keyword_terms(" ".join(item.text for item in compression.selected))
    case = QualityCase(
        id=case_id or _case_id_from_compression_path(compression_path),
        compression_path=compression_path,
        expected_answer_terms=answer_terms[:6],
        expected_evidence_terms=evidence_terms[:8],
        relevant_ranges=evidence_ranges,
        timestamp_tolerance_seconds=timestamp_tolerance_seconds,
        min_answer_term_recall=0.75 if answer_terms else 0.0,
        min_evidence_term_coverage=0.5 if evidence_terms else 0.0,
        min_evidence_relevance_rate=0.8,
        min_timestamp_hit_rate=0.75 if evidence_ranges else 0.0,
        min_token_reduction_percent=min_token_reduction_percent,
        max_selected_evidence=max_selected_evidence or max(len(compression.selected), 1),
        min_visual_evidence=1 if compression.metrics.visual_selected else 0,
        min_audio_evidence=1 if compression.metrics.audio_selected else 0,
    )
    return QualityCaseDraft(case=case, notes=_draft_notes(compression, case))


def draft_quality_cases_from_root(
    root: Path,
    min_token_reduction_percent: float = 90.0,
    timestamp_tolerance_seconds: float = 8.0,
    max_selected_evidence: int | None = None,
) -> list[QualityCaseDraft]:
    return [
        draft_quality_case(
            compression_path=path,
            min_token_reduction_percent=min_token_reduction_percent,
            timestamp_tolerance_seconds=timestamp_tolerance_seconds,
            max_selected_evidence=max_selected_evidence,
        )
        for path in sorted(root.rglob("compression.json"))
    ]


def check_quality_dataset(cases: list[QualityCase]) -> QualityDatasetCheck:
    warnings: list[str] = []
    ids: set[str] = set()
    for case in cases:
        if case.id in ids:
            warnings.append(f"{case.id}: duplicate case id")
        ids.add(case.id)
        warnings.extend(_case_warnings(case))
    return QualityDatasetCheck(
        cases=len(cases),
        runnable_cases=sum(case.video_path is not None for case in cases),
        replay_cases=sum(case.compression_path is not None for case in cases),
        warnings=warnings,
    )


def run_quality_cases(
    cases: list[QualityCase],
    output_root: Path = Path(".gist/quality"),
) -> QualityReport:
    results = [evaluate_quality_case(case, output_root=output_root) for case in cases]
    passed = sum(result.passed for result in results)
    summary = QualitySummary(
        cases=len(results),
        passed=passed,
        pass_rate=0.0 if not results else passed / len(results),
        avg_answer_term_recall=_average(result.answer_term_recall for result in results),
        avg_evidence_term_coverage=_average(result.evidence_term_coverage for result in results),
        avg_evidence_relevance_rate=_average(result.evidence_relevance_rate for result in results),
        avg_timestamp_hit_rate=_average(result.timestamp_hit_rate for result in results),
        avg_token_reduction_percent=_average(result.token_reduction_percent for result in results),
        failure_categories=_category_counts(results),
    )
    return QualityReport(
        passed=all(result.passed for result in results),
        summary=summary,
        results=results,
    )


def evaluate_quality_case(
    case: QualityCase,
    output_root: Path = Path(".gist/quality"),
) -> QualityResult:
    compression = _compression_for_case(case, output_root=output_root)
    answer = compression.answer or ""
    answer_recall = _term_recall(case.expected_answer_terms, answer)
    evidence_text = " ".join(item.text for item in compression.selected)
    evidence_coverage = _term_recall(case.expected_evidence_terms, evidence_text)
    relevant_ranges = _expected_ranges(case)
    selected_ranges = _selected_ranges(compression)
    timestamp_hit_rate = _timestamp_hit_rate(
        selected_ranges=selected_ranges,
        expected_ranges=relevant_ranges,
        tolerance_seconds=case.timestamp_tolerance_seconds,
    )
    relevance_rate = _evidence_relevance_rate(
        compression=compression,
        expected_terms=case.expected_evidence_terms,
        expected_ranges=relevant_ranges,
        tolerance_seconds=case.timestamp_tolerance_seconds,
    )
    token_reduction = compression.metrics.estimated_token_reduction_percent
    failures = _quality_failures(
        case=case,
        compression=compression,
        answer_recall=answer_recall,
        evidence_coverage=evidence_coverage,
        relevance_rate=relevance_rate,
        timestamp_hit_rate=timestamp_hit_rate,
        token_reduction=token_reduction,
    )
    failure_categories = _failure_categories(
        case=case,
        compression=compression,
        answer_recall=answer_recall,
        evidence_coverage=evidence_coverage,
        relevance_rate=relevance_rate,
        timestamp_hit_rate=timestamp_hit_rate,
        token_reduction=token_reduction,
    )
    return QualityResult(
        id=case.id,
        passed=not failures,
        query=compression.query,
        answer=compression.answer,
        answer_term_recall=answer_recall,
        evidence_term_coverage=evidence_coverage,
        evidence_relevance_rate=relevance_rate,
        timestamp_hit_rate=timestamp_hit_rate,
        token_reduction_percent=token_reduction,
        selected_evidence=compression.metrics.selected_candidates,
        visual_evidence=compression.metrics.visual_selected,
        audio_evidence=compression.metrics.audio_selected,
        failure_categories=failure_categories,
        recommendation=_recommendation(failure_categories),
        failures=failures,
    )


def render_quality_markdown(report: QualityReport) -> str:
    lines = [
        "# Gist Local Quality Report",
        "",
        f"- Cases: {report.summary.cases}",
        f"- Passed: {report.summary.passed}",
        f"- Pass rate: {report.summary.pass_rate:.2%}",
        f"- Avg answer term recall: {report.summary.avg_answer_term_recall:.2f}",
        f"- Avg evidence term coverage: {report.summary.avg_evidence_term_coverage:.2f}",
        f"- Avg evidence relevance rate: {report.summary.avg_evidence_relevance_rate:.2f}",
        f"- Avg timestamp hit rate: {report.summary.avg_timestamp_hit_rate:.2f}",
        f"- Avg token reduction: {report.summary.avg_token_reduction_percent:.2f}%",
        "",
        "| Case | Status | Answer Recall | Evidence Coverage | Evidence Relevance | "
        "Timestamp Hit | Token Reduction | Selected | Categories | Recommendation | Failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for result in report.results:
        failures = "; ".join(result.failures)
        lines.append(
            f"| {result.id} | {'pass' if result.passed else 'fail'} | "
            f"{result.answer_term_recall:.2f} | "
            f"{result.evidence_term_coverage:.2f} | "
            f"{result.evidence_relevance_rate:.2f} | "
            f"{result.timestamp_hit_rate:.2f} | "
            f"{result.token_reduction_percent:.2f}% | "
            f"{result.selected_evidence} | "
            f"{', '.join(result.failure_categories)} | "
            f"{result.recommendation or ''} | {failures} |"
        )
    return "\n".join(lines).strip() + "\n"


def render_quality_html(report: QualityReport) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(result.id)}</td>"
        f"<td>{'pass' if result.passed else 'fail'}</td>"
        f"<td>{result.answer_term_recall:.2f}</td>"
        f"<td>{result.evidence_term_coverage:.2f}</td>"
        f"<td>{result.evidence_relevance_rate:.2f}</td>"
        f"<td>{result.timestamp_hit_rate:.2f}</td>"
        f"<td>{result.token_reduction_percent:.2f}%</td>"
        f"<td>{result.selected_evidence}</td>"
        f"<td>{escape(', '.join(result.failure_categories))}</td>"
        f"<td>{escape(result.recommendation or '')}</td>"
        f"<td>{escape('; '.join(result.failures))}</td>"
        "</tr>"
        for result in report.results
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Gist Local Quality Report</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 32px; color: #172026; }}
    h1, h2 {{ color: #0f2f2f; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{
      border: 1px solid #d7dfdf;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #edf5f3; }}
    .metric {{ display: inline-block; margin: 0 16px 12px 0; }}
    .muted {{ color: #5f6f6f; }}
  </style>
</head>
<body>
  <h1>Gist Local Quality Report</h1>
  <p class="muted">
    Curated local checks for answer quality, evidence alignment,
    timestamp localization, and compression.
  </p>
  <h2>Summary</h2>
  <div class="metric"><strong>Cases:</strong> {report.summary.cases}</div>
  <div class="metric"><strong>Passed:</strong> {report.summary.passed}</div>
  <div class="metric"><strong>Pass rate:</strong> {report.summary.pass_rate:.2%}</div>
  <div class="metric">
    <strong>Avg answer recall:</strong> {report.summary.avg_answer_term_recall:.2f}
  </div>
  <div class="metric">
    <strong>Avg evidence coverage:</strong> {report.summary.avg_evidence_term_coverage:.2f}
  </div>
  <div class="metric">
    <strong>Avg evidence relevance:</strong> {report.summary.avg_evidence_relevance_rate:.2f}
  </div>
  <div class="metric">
    <strong>Avg timestamp hit:</strong> {report.summary.avg_timestamp_hit_rate:.2f}
  </div>
  <div class="metric">
    <strong>Avg token reduction:</strong> {report.summary.avg_token_reduction_percent:.2f}%
  </div>
  {_render_failure_category_summary(report.summary.failure_categories)}
  <h2>Cases</h2>
  <table>
    <thead>
      <tr>
        <th>Case</th>
        <th>Status</th>
        <th>Answer Recall</th>
        <th>Evidence Coverage</th>
        <th>Evidence Relevance</th>
        <th>Timestamp Hit</th>
        <th>Token Reduction</th>
        <th>Selected</th>
        <th>Categories</th>
        <th>Recommendation</th>
        <th>Failures</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run curated local Gist quality checks against videos or compression files."
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--html-output", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path(".gist/quality"))
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the dataset shape and warnings without running compression.",
    )
    parser.add_argument(
        "--draft-case-from",
        type=Path,
        help="Print a ready-to-edit JSONL quality case from an existing compression.json.",
    )
    parser.add_argument(
        "--draft-cases-from-root",
        type=Path,
        help="Print ready-to-edit JSONL quality cases for every compression.json under a root.",
    )
    parser.add_argument(
        "--draft-output",
        type=Path,
        help="Write drafted JSONL cases to this file instead of stdout.",
    )
    parser.add_argument("--case-id", help="Override the drafted case id.")
    parser.add_argument(
        "--draft-min-token-reduction-percent",
        type=float,
        default=90.0,
    )
    parser.add_argument(
        "--draft-timestamp-tolerance-seconds",
        type=float,
        default=8.0,
    )
    parser.add_argument("--draft-max-selected-evidence", type=int)
    args = parser.parse_args(argv)

    if args.draft_case_from is not None and args.draft_cases_from_root is not None:
        raise SystemExit("Use either --draft-case-from or --draft-cases-from-root, not both")
    if args.case_id and args.draft_cases_from_root is not None:
        raise SystemExit("--case-id can only be used with --draft-case-from")

    if args.draft_case_from is not None:
        draft = draft_quality_case(
            compression_path=args.draft_case_from,
            case_id=args.case_id,
            min_token_reduction_percent=args.draft_min_token_reduction_percent,
            timestamp_tolerance_seconds=args.draft_timestamp_tolerance_seconds,
            max_selected_evidence=args.draft_max_selected_evidence,
        )
        _write_drafts([draft], output_path=args.draft_output)
        return 0

    if args.draft_cases_from_root is not None:
        drafts = draft_quality_cases_from_root(
            root=args.draft_cases_from_root,
            min_token_reduction_percent=args.draft_min_token_reduction_percent,
            timestamp_tolerance_seconds=args.draft_timestamp_tolerance_seconds,
            max_selected_evidence=args.draft_max_selected_evidence,
        )
        _write_drafts(drafts, output_path=args.draft_output)
        return 0

    if args.dataset is None:
        raise SystemExit("--dataset is required unless --draft-case-from is used")

    cases = load_quality_cases(args.dataset)
    if args.check_only:
        check = check_quality_dataset(cases)
        print(f"cases={check.cases}")
        print(f"runnable_cases={check.runnable_cases}")
        print(f"replay_cases={check.replay_cases}")
        print(f"warnings={len(check.warnings)}")
        for warning in check.warnings:
            print(f"  - {warning}")
        return 0 if not check.warnings else 1

    report = run_quality_cases(cases, output_root=args.output_root)
    if args.output is not None:
        report.write_json(args.output)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_quality_markdown(report))
    if args.html_output is not None:
        args.html_output.parent.mkdir(parents=True, exist_ok=True)
        args.html_output.write_text(render_quality_html(report))

    print(f"cases={report.summary.cases}")
    print(f"passed={report.summary.passed}")
    print(f"pass_rate={report.summary.pass_rate:.2%}")
    for result in report.results:
        status = "pass" if result.passed else "fail"
        print(
            f"{result.id}: {status}, answer_recall={result.answer_term_recall:.2f}, "
            f"evidence_coverage={result.evidence_term_coverage:.2f}, "
            f"evidence_relevance={result.evidence_relevance_rate:.2f}, "
            f"timestamp_hit={result.timestamp_hit_rate:.2f}, "
            f"token_reduction={result.token_reduction_percent:.2f}%"
        )
        for failure in result.failures:
            print(f"  - {failure}")
    return 0 if report.passed else 1


def _compression_for_case(case: QualityCase, output_root: Path) -> CompressionResponse:
    if case.compression_path is not None:
        return _load_compression(case.compression_path)

    assert case.video_path is not None
    assert case.query is not None
    _ingestion, compression = LocalCompressionPipeline(output_root=output_root).run(
        video_path=case.video_path,
        query=case.query,
        preset=case.preset,
        sample_count=case.sample_count,
        audio_window_seconds=case.audio_window_seconds,
        processing_mode=case.processing_mode,
        visual_scorer=case.visual_scorer,
        audio_scorer=case.audio_scorer,
        adaptive_budget=case.adaptive_budget,
        decompose_query=case.decompose_query,
        task_aware_selection=True,
        visual_ocr=case.visual_ocr,
    )
    answer = answer_from_evidence(compression)
    answered = compression.model_copy(update={"answer": verify_answer_claims(answer, compression)})
    return annotate_evidence_support(answered)


def _write_drafts(drafts: list[QualityCaseDraft], output_path: Path | None) -> None:
    lines = [draft.case.model_dump_json(exclude_none=True) for draft in drafts]
    content = "\n".join(lines)
    if content:
        content += "\n"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
    else:
        sys.stdout.write(content)
    for draft in drafts:
        for note in draft.notes:
            print(f"{draft.case.id}: {note}", file=sys.stderr)


def _quality_failures(
    case: QualityCase,
    compression: CompressionResponse,
    answer_recall: float,
    evidence_coverage: float,
    relevance_rate: float,
    timestamp_hit_rate: float,
    token_reduction: float,
) -> list[str]:
    failures: list[str] = []
    if answer_recall < case.min_answer_term_recall:
        failures.append(
            f"answer term recall {answer_recall:.2f} below required "
            f"{case.min_answer_term_recall:.2f}"
        )
    if evidence_coverage < case.min_evidence_term_coverage:
        failures.append(
            f"evidence term coverage {evidence_coverage:.2f} below required "
            f"{case.min_evidence_term_coverage:.2f}"
        )
    if relevance_rate < case.min_evidence_relevance_rate:
        failures.append(
            f"evidence relevance rate {relevance_rate:.2f} below required "
            f"{case.min_evidence_relevance_rate:.2f}"
        )
    if timestamp_hit_rate < case.min_timestamp_hit_rate:
        failures.append(
            f"timestamp hit rate {timestamp_hit_rate:.2f} below required "
            f"{case.min_timestamp_hit_rate:.2f}"
        )
    if token_reduction < case.min_token_reduction_percent:
        failures.append(
            f"token reduction {token_reduction:.2f}% below required "
            f"{case.min_token_reduction_percent:.2f}%"
        )
    if (
        case.max_selected_evidence is not None
        and compression.metrics.selected_candidates > case.max_selected_evidence
    ):
        failures.append(
            f"selected evidence {compression.metrics.selected_candidates} exceeds "
            f"limit {case.max_selected_evidence}"
        )
    if compression.metrics.visual_selected < case.min_visual_evidence:
        failures.append(
            f"visual evidence {compression.metrics.visual_selected} below required "
            f"{case.min_visual_evidence}"
        )
    if compression.metrics.audio_selected < case.min_audio_evidence:
        failures.append(
            f"audio evidence {compression.metrics.audio_selected} below required "
            f"{case.min_audio_evidence}"
        )
    return failures


def _failure_categories(
    case: QualityCase,
    compression: CompressionResponse,
    answer_recall: float,
    evidence_coverage: float,
    relevance_rate: float,
    timestamp_hit_rate: float,
    token_reduction: float,
) -> list[str]:
    categories: list[str] = []
    if answer_recall < case.min_answer_term_recall:
        categories.append("answer_grounding")
    if (
        evidence_coverage < case.min_evidence_term_coverage
        or relevance_rate < case.min_evidence_relevance_rate
    ):
        categories.append("evidence_retrieval")
    if timestamp_hit_rate < case.min_timestamp_hit_rate:
        categories.append("temporal_localization")
    if token_reduction < case.min_token_reduction_percent:
        categories.append("compression_budget")
    if (
        case.max_selected_evidence is not None
        and compression.metrics.selected_candidates > case.max_selected_evidence
    ):
        categories.append("evidence_pruning")
    if (
        compression.metrics.visual_selected < case.min_visual_evidence
        or compression.metrics.audio_selected < case.min_audio_evidence
    ):
        categories.append("modality_balance")
    return sorted(set(categories))


def _recommendation(categories: list[str]) -> str | None:
    if not categories:
        return None
    if "evidence_retrieval" in categories and "temporal_localization" in categories:
        return "Improve query-aware retrieval before changing answer generation."
    if "evidence_retrieval" in categories:
        return "Tune candidate scoring, query decomposition, or scene/audio fusion."
    if "temporal_localization" in categories:
        return "Adjust clip span, transcript anchoring, or temporal context expansion."
    if "answer_grounding" in categories:
        return "Improve answer synthesis and citation pruning from selected evidence."
    if "compression_budget" in categories:
        return "Tighten token estimation or reduce selected context budget."
    if "evidence_pruning" in categories:
        return "Make answer-support pruning more selective."
    if "modality_balance" in categories:
        return "Review router modality allocation for this query intent."
    return "Inspect selected evidence and add a targeted regression case."


def _case_warnings(case: QualityCase) -> list[str]:
    warnings: list[str] = []
    if case.compression_path is not None and not case.compression_path.exists():
        warnings.append(f"{case.id}: compression_path does not exist: {case.compression_path}")
    if case.video_path is not None and not case.video_path.exists():
        warnings.append(f"{case.id}: video_path does not exist: {case.video_path}")
    if case.compression_path is not None and case.video_path is not None:
        warnings.append(
            f"{case.id}: both compression_path and video_path are set; "
            "compression_path replay will be used"
        )
    if not case.expected_answer_terms:
        warnings.append(f"{case.id}: expected_answer_terms is empty")
    if not case.expected_evidence_terms:
        warnings.append(f"{case.id}: expected_evidence_terms is empty")
    if not case.relevant_timestamps and not case.relevant_ranges:
        warnings.append(f"{case.id}: no relevant timestamps or ranges")
    if case.min_answer_term_recall == 0:
        warnings.append(f"{case.id}: min_answer_term_recall is not enforcing quality")
    if case.min_evidence_relevance_rate == 0:
        warnings.append(f"{case.id}: min_evidence_relevance_rate is not enforcing quality")
    if case.min_token_reduction_percent == 0:
        warnings.append(f"{case.id}: min_token_reduction_percent is not enforcing compression")
    return warnings


def _load_compression(path: Path) -> CompressionResponse:
    payload = json.loads(path.read_text())
    compression_payload = payload.get("compression", payload)
    return CompressionResponse.model_validate(compression_payload)


def _case_id_from_compression_path(path: Path) -> str:
    parts = [part for part in path.parts if part not in {".", "compression.json"}]
    if len(parts) >= 2:
        return _safe_case_id(f"{parts[-2]}-{parts[-1]}")
    return _safe_case_id(path.parent.name or path.stem)


def _safe_case_id(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return normalized or "quality-case"


def _keyword_terms(text: str, limit: int = 16) -> list[str]:
    counts: dict[str, int] = {}
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9'-]{2,}", text.lower()):
        normalized = token.strip("'")
        normalized = _TERM_NORMALIZATIONS.get(normalized, normalized)
        if normalized in _STOPWORDS:
            continue
        counts[normalized] = counts.get(normalized, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [term for term, _count in ranked[:limit]]


def _draft_notes(compression: CompressionResponse, case: QualityCase) -> list[str]:
    notes = [
        "Review expected_answer_terms and expected_evidence_terms before committing.",
        "Review relevant_ranges against the HTML report/video player.",
    ]
    if not case.expected_answer_terms:
        notes.append("No answer terms inferred; add expected_answer_terms manually.")
    if not case.expected_evidence_terms:
        notes.append("No evidence terms inferred; add expected_evidence_terms manually.")
    if compression.metrics.estimated_token_reduction_percent < case.min_token_reduction_percent:
        notes.append(
            "Current token reduction is below the drafted threshold; lower the "
            "threshold only if this is intentional."
        )
    return notes


def _selected_ranges(compression: CompressionResponse) -> list[TimeRange]:
    ranges: list[TimeRange] = []
    for item in compression.selected:
        start_seconds = item.clip_start_seconds or item.scene_start_seconds
        end_seconds = item.clip_end_seconds or item.scene_end_seconds
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


def _expected_ranges(case: QualityCase) -> list[TimeRange]:
    ranges = list(case.relevant_ranges)
    ranges.extend(
        TimeRange(start_seconds=timestamp, end_seconds=timestamp)
        for timestamp in case.relevant_timestamps
    )
    return ranges


def _timestamp_hit_rate(
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


def _evidence_relevance_rate(
    compression: CompressionResponse,
    expected_terms: list[str],
    expected_ranges: list[TimeRange],
    tolerance_seconds: float,
) -> float:
    if not compression.selected:
        return 0.0
    if not expected_terms and not expected_ranges:
        return 1.0

    relevant = 0
    for item in compression.selected:
        text_match = _contains_any_term(item.text, expected_terms)
        item_range = _selected_ranges(
            compression.model_copy(update={"selected": [item]})
        )[0]
        time_match = any(
            item_range.overlaps(expected, tolerance_seconds=tolerance_seconds)
            for expected in expected_ranges
        )
        if text_match or time_match:
            relevant += 1
    return relevant / len(compression.selected)


def _term_recall(expected_terms: list[str], text: str) -> float:
    normalized_terms = [term.strip().lower() for term in expected_terms if term.strip()]
    if not normalized_terms:
        return 1.0
    normalized_text = text.lower()
    hits = sum(1 for term in normalized_terms if term in normalized_text)
    return hits / len(normalized_terms)


def _contains_any_term(text: str, expected_terms: list[str]) -> bool:
    normalized_text = text.lower()
    return any(
        term.strip().lower() in normalized_text
        for term in expected_terms
        if term.strip()
    )


def _average(values) -> float:
    resolved = list(values)
    if not resolved:
        return 0.0
    return sum(resolved) / len(resolved)


def _category_counts(results: list[QualityResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for category in result.failure_categories:
            counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def _render_failure_category_summary(category_counts: dict[str, int]) -> str:
    if not category_counts:
        return "<p class=\"muted\">No failure categories.</p>"
    items = "".join(
        f"<li>{escape(category)}: {count}</li>"
        for category, count in category_counts.items()
    )
    return f"<h2>Failure Categories</h2><ul>{items}</ul>"


_STOPWORDS = {
    "all",
    "and",
    "any",
    "about",
    "after",
    "again",
    "also",
    "answer",
    "are",
    "but",
    "because",
    "being",
    "clip",
    "does",
    "during",
    "evidence",
    "for",
    "from",
    "has",
    "have",
    "here",
    "his",
    "how",
    "its",
    "into",
    "like",
    "not",
    "our",
    "out",
    "she",
    "than",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "transcript",
    "was",
    "what",
    "when",
    "who",
    "where",
    "which",
    "why",
    "with",
    "would",
    "you",
    "your",
}


_TERM_NORMALIZATIONS = {
    "it's": "its",
    "that's": "that",
    "there's": "there",
    "they're": "they",
    "you're": "you",
}


if __name__ == "__main__":
    raise SystemExit(main())
