"""Ablation runner for the long-video quality suite.

Compares full Gist audio-visual retrieval against three baselines while holding
the candidate pool fixed, so the only thing that varies is the selection input:

- ``full_gist``       -- the real compressor over visual + audio candidates.
- ``visual_only``     -- the real compressor with audio candidates removed.
- ``transcript_only`` -- the real compressor with visual candidates removed.
- ``uniform``         -- query-agnostic uniform temporal sampling at the same
                          budget the full Gist run selected for that case.

Every mode is scored through :func:`gist.eval.quality.evaluate_quality_case`,
so the metrics and gate thresholds are identical to the committed quality suite.
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path

from pydantic import BaseModel

from gist.core.answering import answer_from_evidence, verify_answer_claims
from gist.core.compressor import GistCompressor
from gist.core.evidence_pruning import (
    annotate_evidence_support,
    consolidate_redundant_evidence,
    prune_evidence_to_answer,
    prune_evidence_to_answer_citations,
    prune_weakly_grounded_evidence,
)
from gist.core.modes import AudioScoringMode, VisualScoringMode
from gist.core.quality_gate import apply_quality_gate
from gist.core.presets import CompressionPreset
from gist.core.schemas import (
    Candidate,
    CompressionMetrics,
    CompressionRequest,
    CompressionResponse,
)
from gist.core.token_estimation import TokenEstimatorProfile, estimate_tokens
from gist.eval.baselines import _score_topk_select, _uniform_select
from gist.eval.quality import (
    QualityCase,
    QualityResult,
    evaluate_quality_case,
    load_quality_cases,
)
from gist.audio.whisper import TranscriptQuality
from gist.media.longform import ProcessingMode
from gist.pipeline import LocalCompressionPipeline, _with_raw_reduction_metrics

MODES = ("full_gist", "visual_only", "transcript_only", "score_topk", "uniform")

MODE_LABELS = {
    "full_gist": "Full Gist (audio+visual)",
    "visual_only": "Visual-only retrieval",
    "transcript_only": "Transcript-only retrieval",
    "score_topk": "Score top-k (relevance only)",
    "uniform": "Uniform sampling",
}


class ResolvedCaseConfig(BaseModel):
    """Pipeline configuration recovered from a committed compression artifact."""

    video_path: Path
    query: str
    duration_seconds: float
    preset: CompressionPreset
    audio_scorer: AudioScoringMode
    visual_scorer: VisualScoringMode
    visual_ocr: bool
    processing_mode: ProcessingMode
    transcript_quality: str | None = None
    whisper_model_size: str | None = None
    whisper_device: str | None = None
    whisper_compute_type: str | None = None
    whisper_beam_size: int | None = None


class ModeOutcome(BaseModel):
    mode: str
    passed: bool
    answer_term_recall: float
    evidence_term_coverage: float
    evidence_relevance_rate: float
    timestamp_hit_rate: float
    grounded_evidence_rate: float
    token_reduction_percent: float
    selected_evidence: int
    visual_evidence: int
    audio_evidence: int
    answer: str | None = None
    failures: list[str]


class AblationCaseResult(BaseModel):
    case_id: str
    query_category: str | None = None
    domain: str | None = None
    outcomes: dict[str, ModeOutcome]


class ModeSummary(BaseModel):
    mode: str
    label: str
    cases: int
    passed: int
    pass_rate: float
    avg_answer_term_recall: float
    avg_evidence_term_coverage: float
    avg_evidence_relevance_rate: float
    avg_timestamp_hit_rate: float
    avg_grounded_evidence_rate: float
    avg_token_reduction_percent: float
    avg_selected_evidence: float


class AblationReport(BaseModel):
    cases: int
    summaries: dict[str, ModeSummary]
    results: list[AblationCaseResult]

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))


def resolve_case_config(
    case: QualityCase,
    default_visual_scorer: VisualScoringMode = VisualScoringMode.CLIP_SCENE,
) -> ResolvedCaseConfig:
    """Recover the pipeline config a committed artifact was produced with."""

    if case.compression_path is None:
        raise ValueError(f"{case.id}: ablation requires a compression_path")
    payload = json.loads(case.compression_path.read_text())
    compression = payload.get("compression", payload)
    ingestion = payload.get("ingestion", {})
    source_path = ingestion.get("source_path")
    if not source_path:
        raise ValueError(f"{case.id}: artifact has no ingestion.source_path")
    metadata = ingestion.get("metadata", {})
    duration_seconds = float(metadata.get("duration_seconds") or 0.0)
    if duration_seconds <= 0:
        raise ValueError(f"{case.id}: artifact has no positive duration")

    transcript = compression.get("transcript_metadata")
    audio_used = compression.get("audio_scorer_used")
    if transcript or audio_used == AudioScoringMode.WHISPER.value:
        audio_scorer = AudioScoringMode.WHISPER
    else:
        audio_scorer = AudioScoringMode(audio_used or AudioScoringMode.BASELINE.value)

    transcript = transcript or {}
    return ResolvedCaseConfig(
        video_path=Path(source_path),
        query=compression.get("query") or (case.query or ""),
        duration_seconds=duration_seconds,
        preset=CompressionPreset(compression.get("preset", CompressionPreset.BALANCED.value)),
        audio_scorer=audio_scorer,
        visual_scorer=default_visual_scorer,
        visual_ocr=case.visual_ocr,
        processing_mode=ProcessingMode.AUTO,
        transcript_quality=transcript.get("quality"),
        whisper_model_size=transcript.get("model_size"),
        whisper_device=transcript.get("device"),
        whisper_compute_type=transcript.get("compute_type"),
        whisper_beam_size=transcript.get("beam_size"),
    )


def _extractive_answer(response: CompressionResponse) -> CompressionResponse:
    """Reproduce the CLI extractive answer step: answer, verify, annotate grounding."""

    answer = answer_from_evidence(response)
    answered = response.model_copy(update={"answer": verify_answer_claims(answer, response)})
    return annotate_evidence_support(answered)


def _finalize_gist(response: CompressionResponse) -> CompressionResponse:
    """Replicate ``cli._finalize_compression`` (extractive path, no clips/masks).

    This is the exact evidence-pruning chain the committed quality artifacts went
    through, so ``full_gist`` reproduces them and the single-modality ablations get
    the same treatment.
    """

    def ids(compression: CompressionResponse) -> list[str]:
        return [item.id for item in compression.selected]

    response = _extractive_answer(response)

    before = ids(response)
    response = prune_evidence_to_answer(response)
    if ids(response) != before:
        response = _extractive_answer(response)

    before = ids(response)
    response = prune_evidence_to_answer_citations(response)
    if ids(response) != before:
        response = _extractive_answer(response)
        response = prune_evidence_to_answer_citations(response)

    before = ids(response)
    response = consolidate_redundant_evidence(response)
    if ids(response) != before:
        response = _extractive_answer(response)
        response = prune_evidence_to_answer_citations(response)

    before = ids(response)
    response = prune_weakly_grounded_evidence(response)
    if ids(response) != before:
        response = _extractive_answer(response)

    return apply_quality_gate(response)


def _compress_mode(
    compressor: GistCompressor,
    request_template: CompressionRequest,
    visual: list[Candidate],
    audio: list[Candidate],
    raw_candidate_count: int,
    raw_visual_count: int,
    raw_audio_count: int,
) -> CompressionResponse:
    request = request_template.model_copy(
        update={"visual_candidates": visual, "audio_candidates": audio}
    )
    response = compressor.compress(request)
    response = _with_raw_reduction_metrics(
        compression=response,
        raw_candidate_count=raw_candidate_count,
        raw_visual_count=raw_visual_count,
        raw_audio_count=raw_audio_count,
    )
    return _finalize_gist(response)


def _uniform_mode(
    request_template: CompressionRequest,
    visual: list[Candidate],
    audio: list[Candidate],
    budget: int,
    raw_candidate_count: int,
    raw_visual_count: int,
    raw_audio_count: int,
) -> CompressionResponse:
    selected = _uniform_select(visual, audio, max(budget, 1))
    input_count = len(visual) + len(audio)
    selected_count = len(selected)
    reduction_ratio = 1.0 if input_count == 0 else selected_count / input_count
    token_estimate = estimate_tokens(
        input_visual_candidates=len(visual),
        input_audio_candidates=len(audio),
        selected_modalities=[item.modality for item in selected],
        profile=request_template.token_estimator,
    )
    metrics = CompressionMetrics(
        input_candidates=input_count,
        selected_candidates=selected_count,
        visual_selected=sum(item.modality.value == "visual" for item in selected),
        audio_selected=sum(item.modality.value == "audio" for item in selected),
        estimated_candidate_reduction_ratio=reduction_ratio,
        estimated_candidate_reduction_percent=(1.0 - reduction_ratio) * 100,
        dropped_candidates=max(input_count - selected_count, 0),
        budget_mode="uniform",
        budget_preset_used=request_template.preset,
        estimated_baseline_tokens=token_estimate.baseline_tokens,
        estimated_compressed_tokens=token_estimate.compressed_tokens,
        estimated_saved_tokens=token_estimate.saved_tokens,
        estimated_token_reduction_ratio=token_estimate.reduction_ratio,
        estimated_token_reduction_percent=token_estimate.reduction_percent,
        token_estimator=token_estimate.profile,
    )
    response = CompressionResponse(
        video_id=request_template.video_id,
        query=request_template.query,
        preset=request_template.preset,
        selected=selected,
        metrics=metrics,
    )
    response = _with_raw_reduction_metrics(
        compression=response,
        raw_candidate_count=raw_candidate_count,
        raw_visual_count=raw_visual_count,
        raw_audio_count=raw_audio_count,
    )
    # Uniform is a naive baseline: extract an answer for scoring, but do NOT apply
    # Gist's query-aware evidence pruning — that would leak Gist's selection smarts.
    return _extractive_answer(response)


def _score_topk_mode(
    request_template: CompressionRequest,
    visual: list[Candidate],
    audio: list[Candidate],
    budget: int,
    raw_candidate_count: int,
    raw_visual_count: int,
    raw_audio_count: int,
) -> CompressionResponse:
    """Query-aware ranking baseline: greedy top-k by candidate saliency, no
    diversity, scene structure, cross-modal fusion, or answer pruning."""

    selected = _score_topk_select(visual, audio, max(budget, 1))
    input_count = len(visual) + len(audio)
    selected_count = len(selected)
    reduction_ratio = 1.0 if input_count == 0 else selected_count / input_count
    token_estimate = estimate_tokens(
        input_visual_candidates=len(visual),
        input_audio_candidates=len(audio),
        selected_modalities=[item.modality for item in selected],
        profile=request_template.token_estimator,
    )
    metrics = CompressionMetrics(
        input_candidates=input_count,
        selected_candidates=selected_count,
        visual_selected=sum(item.modality.value == "visual" for item in selected),
        audio_selected=sum(item.modality.value == "audio" for item in selected),
        estimated_candidate_reduction_ratio=reduction_ratio,
        estimated_candidate_reduction_percent=(1.0 - reduction_ratio) * 100,
        dropped_candidates=max(input_count - selected_count, 0),
        budget_mode="score_topk",
        budget_preset_used=request_template.preset,
        estimated_baseline_tokens=token_estimate.baseline_tokens,
        estimated_compressed_tokens=token_estimate.compressed_tokens,
        estimated_saved_tokens=token_estimate.saved_tokens,
        estimated_token_reduction_ratio=token_estimate.reduction_ratio,
        estimated_token_reduction_percent=token_estimate.reduction_percent,
        token_estimator=token_estimate.profile,
    )
    response = CompressionResponse(
        video_id=request_template.video_id,
        query=request_template.query,
        preset=request_template.preset,
        selected=selected,
        metrics=metrics,
    )
    response = _with_raw_reduction_metrics(
        compression=response,
        raw_candidate_count=raw_candidate_count,
        raw_visual_count=raw_visual_count,
        raw_audio_count=raw_audio_count,
    )
    return _extractive_answer(response)


def _mode_outcome(mode: str, result: QualityResult) -> ModeOutcome:
    return ModeOutcome(
        mode=mode,
        passed=result.passed,
        answer_term_recall=result.answer_term_recall,
        evidence_term_coverage=result.evidence_term_coverage,
        evidence_relevance_rate=result.evidence_relevance_rate,
        timestamp_hit_rate=result.timestamp_hit_rate,
        grounded_evidence_rate=result.grounded_evidence_rate,
        token_reduction_percent=result.token_reduction_percent,
        selected_evidence=result.selected_evidence,
        visual_evidence=result.visual_evidence,
        audio_evidence=result.audio_evidence,
        answer=result.answer,
        failures=result.failures,
    )


def run_ablation_case(
    case: QualityCase,
    pipeline: LocalCompressionPipeline,
    compressor: GistCompressor,
    output_root: Path,
    default_visual_scorer: VisualScoringMode = VisualScoringMode.CLIP_SCENE,
) -> AblationCaseResult:
    config = resolve_case_config(case, default_visual_scorer=default_visual_scorer)
    ingested, candidates, raw_candidate_count = pipeline.prepare_candidates(
        video_path=config.video_path,
        query=config.query,
        sample_count=None,
        audio_window_seconds=None,
        processing_mode=config.processing_mode,
        visual_scorer=config.visual_scorer,
        audio_scorer=config.audio_scorer,
        visual_ocr=config.visual_ocr,
        # Match the transcript quality the artifact was produced with, otherwise
        # prepare_candidates defaults to "balanced" and re-transcribes the whole
        # video with a heavier Whisper model (cache miss + wrong transcript).
        transcript_quality=(
            TranscriptQuality(config.transcript_quality)
            if config.transcript_quality
            else TranscriptQuality.BALANCED
        ),
        whisper_model_size=config.whisper_model_size,
        whisper_device=config.whisper_device,
        whisper_compute_type=config.whisper_compute_type,
        whisper_beam_size=config.whisper_beam_size,
    )
    request_template = CompressionRequest(
        video_id=ingested.video_id,
        query=config.query,
        duration_seconds=ingested.metadata.duration_seconds,
        preset=config.preset,
        adaptive_budget=case.adaptive_budget,
        decompose_query=case.decompose_query,
        token_estimator=TokenEstimatorProfile.GENERIC,
        task_aware_selection=True,
    )
    raw_visual_count = len(ingested.frames)
    raw_audio_count = len(ingested.audio_windows)

    responses: dict[str, CompressionResponse] = {}
    responses["full_gist"] = _compress_mode(
        compressor,
        request_template,
        candidates.visual,
        candidates.audio,
        raw_candidate_count,
        raw_visual_count,
        raw_audio_count,
    )
    responses["visual_only"] = _compress_mode(
        compressor,
        request_template,
        candidates.visual,
        [],
        raw_candidate_count,
        raw_visual_count,
        raw_audio_count,
    )
    responses["transcript_only"] = _compress_mode(
        compressor,
        request_template,
        [],
        candidates.audio,
        raw_candidate_count,
        raw_visual_count,
        raw_audio_count,
    )
    shared_budget = max(responses["full_gist"].metrics.selected_candidates, 1)
    responses["score_topk"] = _score_topk_mode(
        request_template,
        candidates.visual,
        candidates.audio,
        shared_budget,
        raw_candidate_count,
        raw_visual_count,
        raw_audio_count,
    )
    responses["uniform"] = _uniform_mode(
        request_template,
        candidates.visual,
        candidates.audio,
        shared_budget,
        raw_candidate_count,
        raw_visual_count,
        raw_audio_count,
    )

    outcomes: dict[str, ModeOutcome] = {}
    for mode in MODES:
        mode_path = output_root / _safe(case.id) / mode / "compression.json"
        mode_path.parent.mkdir(parents=True, exist_ok=True)
        mode_path.write_text(
            json.dumps({"compression": responses[mode].model_dump(mode="json")}, indent=2)
        )
        mode_case = case.model_copy(
            update={"compression_path": mode_path, "video_path": None, "query": None}
        )
        result = evaluate_quality_case(mode_case, output_root=output_root / _safe(case.id) / mode)
        outcomes[mode] = _mode_outcome(mode, result)

    return AblationCaseResult(
        case_id=case.id,
        query_category=case.query_category.value if case.query_category else None,
        domain=case.domain,
        outcomes=outcomes,
    )


def _summarize(mode: str, results: list[AblationCaseResult]) -> ModeSummary:
    outcomes = [result.outcomes[mode] for result in results if mode in result.outcomes]
    count = len(outcomes)
    passed = sum(outcome.passed for outcome in outcomes)

    def avg(attr: str) -> float:
        if not outcomes:
            return 0.0
        return sum(getattr(outcome, attr) for outcome in outcomes) / count

    return ModeSummary(
        mode=mode,
        label=MODE_LABELS[mode],
        cases=count,
        passed=passed,
        pass_rate=0.0 if not count else passed / count,
        avg_answer_term_recall=avg("answer_term_recall"),
        avg_evidence_term_coverage=avg("evidence_term_coverage"),
        avg_evidence_relevance_rate=avg("evidence_relevance_rate"),
        avg_timestamp_hit_rate=avg("timestamp_hit_rate"),
        avg_grounded_evidence_rate=avg("grounded_evidence_rate"),
        avg_token_reduction_percent=avg("token_reduction_percent"),
        avg_selected_evidence=avg("selected_evidence"),
    )


def run_ablation_suite(
    cases: list[QualityCase],
    output_root: Path = Path(".gist/ablation"),
    default_visual_scorer: VisualScoringMode = VisualScoringMode.CLIP_SCENE,
    progress=None,
) -> AblationReport:
    pipeline = LocalCompressionPipeline(output_root=output_root)
    compressor = GistCompressor()
    results: list[AblationCaseResult] = []
    for index, case in enumerate(cases, start=1):
        if progress is not None:
            progress(f"[{index}/{len(cases)}] {case.id}")
        results.append(
            run_ablation_case(
                case,
                pipeline=pipeline,
                compressor=compressor,
                output_root=output_root,
                default_visual_scorer=default_visual_scorer,
            )
        )
    summaries = {mode: _summarize(mode, results) for mode in MODES}
    return AblationReport(cases=len(results), summaries=summaries, results=results)


def render_ablation_markdown(report: AblationReport) -> str:
    lines = [
        "# Gist Ablation Report",
        "",
        f"- Cases: {report.cases}",
        "- Candidate pool held fixed per case; only the selection input varies.",
        "- Uniform sampling uses the same budget the full Gist run selected.",
        "",
        "## How to read this",
        "",
        "All four modes run the current code over an identical per-case candidate "
        "pool and are scored against the same reference terms and timestamp ranges. "
        "The **continuous averages below are the comparison of record**: they show, "
        "at matched token reduction, which selection strategy best recovers the "
        "answer-bearing moment. The reference terms/ranges were authored from the "
        "full-Gist artifacts, so they are if anything biased in full Gist's favour; "
        "the baselines are judged against the same targets.",
        "",
        "The per-case pass/fail table is secondary. Its gates were pinned to the "
        "exact evidence the committed suite selected, so a mode that picks a "
        "different-but-valid moment (e.g. one of several identical recurring slides) "
        "can score high on the averages yet miss a strict gate.",
        "",
        "## Mode comparison",
        "",
        "| Mode | Pass rate | Answer recall | Evidence coverage | Evidence relevance "
        "| Timestamp hit | Grounded | Token reduction | Avg selected |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in MODES:
        summary = report.summaries[mode]
        lines.append(
            f"| {summary.label} | {summary.pass_rate:.0%} "
            f"({summary.passed}/{summary.cases}) "
            f"| {summary.avg_answer_term_recall:.2f} "
            f"| {summary.avg_evidence_term_coverage:.2f} "
            f"| {summary.avg_evidence_relevance_rate:.2f} "
            f"| {summary.avg_timestamp_hit_rate:.2f} "
            f"| {summary.avg_grounded_evidence_rate:.2f} "
            f"| {summary.avg_token_reduction_percent:.2f}% "
            f"| {summary.avg_selected_evidence:.2f} |"
        )
    lines.extend(["", "## Per-case pass/fail", "", "| Case | Category | " + " | ".join(
        MODE_LABELS[mode] for mode in MODES
    ) + " |", "| :--- | :--- | " + " | ".join(":---:" for _ in MODES) + " |"])
    for result in report.results:
        cells = " | ".join(
            ("pass" if result.outcomes[mode].passed else "fail") for mode in MODES
        )
        lines.append(f"| {result.case_id} | {result.query_category or ''} | {cells} |")
    return "\n".join(lines) + "\n"


def render_ablation_html(report: AblationReport) -> str:
    header = "".join(f"<th>{escape(MODE_LABELS[mode])}</th>" for mode in MODES)
    summary_rows = "".join(
        (
            "<tr>"
            f"<td>{escape(report.summaries[mode].label)}</td>"
            f"<td>{report.summaries[mode].pass_rate:.0%} "
            f"({report.summaries[mode].passed}/{report.summaries[mode].cases})</td>"
            f"<td>{report.summaries[mode].avg_answer_term_recall:.2f}</td>"
            f"<td>{report.summaries[mode].avg_evidence_term_coverage:.2f}</td>"
            f"<td>{report.summaries[mode].avg_evidence_relevance_rate:.2f}</td>"
            f"<td>{report.summaries[mode].avg_timestamp_hit_rate:.2f}</td>"
            f"<td>{report.summaries[mode].avg_grounded_evidence_rate:.2f}</td>"
            f"<td>{report.summaries[mode].avg_token_reduction_percent:.2f}%</td>"
            f"<td>{report.summaries[mode].avg_selected_evidence:.2f}</td>"
            "</tr>"
        )
        for mode in MODES
    )
    case_rows = "".join(
        (
            "<tr>"
            f"<td>{escape(result.case_id)}</td>"
            f"<td>{escape(result.query_category or '')}</td>"
            + "".join(
                f"<td class='{('pass' if result.outcomes[mode].passed else 'fail')}'>"
                f"{('pass' if result.outcomes[mode].passed else 'fail')}</td>"
                for mode in MODES
            )
            + "</tr>"
        )
        for result in report.results
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Gist Ablation Report</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 32px; color: #172026; }}
    h1, h2 {{ color: #0f2f2f; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d7dfdf; padding: 8px 10px; text-align: left; }}
    th {{ background: #edf5f3; }}
    td.pass {{ background: #e7f6ec; }}
    td.fail {{ background: #fbe9e9; }}
  </style>
</head>
<body>
  <h1>Gist Ablation Report</h1>
  <p>Cases: {report.cases}. Candidate pool held fixed per case; only the selection
  input varies. Uniform sampling uses the same budget the full Gist run selected.</p>
  <h2>Mode comparison</h2>
  <table>
    <thead><tr><th>Mode</th><th>Pass rate</th><th>Answer recall</th>
    <th>Evidence coverage</th><th>Evidence relevance</th><th>Timestamp hit</th>
    <th>Grounded</th><th>Token reduction</th><th>Avg selected</th></tr></thead>
    <tbody>{summary_rows}</tbody>
  </table>
  <h2>Per-case pass/fail</h2>
  <table>
    <thead><tr><th>Case</th><th>Category</th>{header}</tr></thead>
    <tbody>{case_rows}</tbody>
  </table>
</body>
</html>
"""


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run selection ablations over the long-video quality suite."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(".gist/ablation"))
    parser.add_argument("--json", type=Path, dest="json_output")
    parser.add_argument("--markdown", type=Path, dest="markdown_output")
    parser.add_argument("--html", type=Path, dest="html_output")
    parser.add_argument("--limit", type=int, help="Only run the first N cases.")
    parser.add_argument("--case-id", action="append", help="Run only these case ids.")
    parser.add_argument(
        "--visual-scorer",
        default=VisualScoringMode.CLIP_SCENE.value,
        choices=[mode.value for mode in VisualScoringMode],
    )
    args = parser.parse_args(argv)

    cases = load_quality_cases(args.dataset)
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case.id in wanted]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("no cases selected")

    report = run_ablation_suite(
        cases,
        output_root=args.output_root,
        default_visual_scorer=VisualScoringMode(args.visual_scorer),
        progress=lambda message: print(message, flush=True),
    )

    if args.json_output is not None:
        report.write_json(args.json_output)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_ablation_markdown(report))
    if args.html_output is not None:
        args.html_output.parent.mkdir(parents=True, exist_ok=True)
        args.html_output.write_text(render_ablation_html(report))

    print(f"cases={report.cases}")
    for mode in MODES:
        summary = report.summaries[mode]
        print(
            f"{mode}: pass_rate={summary.pass_rate:.0%} "
            f"({summary.passed}/{summary.cases}), "
            f"answer_recall={summary.avg_answer_term_recall:.2f}, "
            f"evidence_relevance={summary.avg_evidence_relevance_rate:.2f}, "
            f"timestamp_hit={summary.avg_timestamp_hit_rate:.2f}, "
            f"grounded={summary.avg_grounded_evidence_rate:.2f}, "
            f"token_reduction={summary.avg_token_reduction_percent:.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
