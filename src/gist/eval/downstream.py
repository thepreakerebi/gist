"""Downstream end-to-end evaluation for Gist.

Gist's claim is that query-aware audio-visual compression lets a downstream
language model answer questions about a long video *better and cheaper* than
naive context. This harness tests that claim directly.

For each transcript-answerable case it builds three contexts over the same
candidate pool and feeds each to the *same* local LLM answerer:

- ``whole``   -- the full transcript (every audio window). The expensive ceiling.
- ``uniform`` -- evenly-spaced transcript chunks at the Gist evidence budget
                 (query-agnostic). The naive same-budget baseline.
- ``gist``    -- Gist's selected evidence.

It then scores each answer against the expected answer terms and records the
context size fed to the model, so we can compare answer accuracy *and* token
cost across conditions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel

from gist.audio.whisper import FasterWhisperTranscriber
from gist.core.cache import ingestion_cache_key
from gist.core.presets import CompressionPreset
from gist.core.schemas import (
    Candidate,
    CompressionMetrics,
    CompressionResponse,
    Modality,
    SelectedCandidate,
)
from gist.eval.ablation import resolve_case_config
from gist.eval.baselines import _uniform_select
from gist.eval.judge import JudgeError, LlmJudge
from gist.eval.quality import QualityCase, _load_compression, _term_recall, load_quality_cases
from gist.gateway.evidence_package import build_evidence_prompt
from gist.gateway.ollama import DEFAULT_OLLAMA_MODEL, OllamaGatewayError, OllamaTextGateway
from gist.gateway.schemas import GatewayRequest
from gist.pipeline import LocalCompressionPipeline

CONDITIONS = ("whole", "uniform", "gist")

CONDITION_LABELS = {
    "whole": "Whole transcript (all audio)",
    "uniform": "Uniform sampling (Gist budget)",
    "gist": "Gist-compressed evidence",
}

# Categories whose answer lives in speech, so a text-only answerer can be judged.
TRANSCRIPT_CATEGORIES = {"speech_semantic", "mixed_av", "global_summary"}


class CaseConditionResult(BaseModel):
    condition: str
    answer: str
    answer_term_recall: float
    correct: bool
    context_tokens: int
    evidence_items: int
    judge_correct: bool | None = None
    judge_score: float | None = None
    judge_reason: str | None = None


class DownstreamCaseResult(BaseModel):
    case_id: str
    query_category: str | None = None
    domain: str | None = None
    query: str
    conditions: dict[str, CaseConditionResult]


class ConditionSummary(BaseModel):
    condition: str
    label: str
    cases: int
    avg_answer_term_recall: float
    correct_rate: float
    avg_context_tokens: float
    judge_correct_rate: float | None = None


class DownstreamReport(BaseModel):
    cases: int
    answerer: str
    correct_threshold: float
    summaries: dict[str, ConditionSummary]
    results: list[DownstreamCaseResult]

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))


def _approx_tokens(text: str) -> int:
    # Transparent proxy: ~4 characters per token.
    return max(1, len(text) // 4)


def _to_selected(candidate: Candidate, modality: Modality, rank: int) -> SelectedCandidate:
    return SelectedCandidate(
        id=candidate.id,
        modality=modality,
        timestamp_seconds=candidate.timestamp_seconds,
        text=candidate.text,
        segment_id=candidate.segment_id,
        scene_start_seconds=candidate.scene_start_seconds,
        scene_end_seconds=candidate.scene_end_seconds,
        selection_rank=rank,
        relevance_score=candidate.saliency_score or 0.0,
        normalized_score=candidate.saliency_score or 0.0,
        mmr_score=candidate.saliency_score or 0.0,
        source_score_type="downstream",
        reason="downstream context item",
    )


def _response_for(query: str, selected: list[SelectedCandidate]) -> CompressionResponse:
    visual = sum(item.modality == Modality.VISUAL for item in selected)
    audio = sum(item.modality == Modality.AUDIO for item in selected)
    metrics = CompressionMetrics(
        input_candidates=len(selected),
        selected_candidates=len(selected),
        visual_selected=visual,
        audio_selected=audio,
        estimated_candidate_reduction_ratio=1.0,
        estimated_candidate_reduction_percent=0.0,
        dropped_candidates=0,
        budget_preset_used=CompressionPreset.BALANCED,
    )
    return CompressionResponse(
        video_id="downstream",
        query=query,
        preset=CompressionPreset.BALANCED,
        selected=selected,
        metrics=metrics,
    )


def _build_condition_selected(
    condition: str,
    audio: list[Candidate],
    gist_selected: list[SelectedCandidate],
) -> list[SelectedCandidate]:
    if condition == "gist":
        return gist_selected
    budget = max(len(gist_selected), 1) if condition == "uniform" else len(audio)
    # Uniform temporal spacing over audio windows (query-agnostic).
    return _uniform_select(visual_candidates=[], audio_candidates=audio, budget=budget)


def _ingest_only(pipeline: LocalCompressionPipeline, config):
    """Get the ingested video (frames + audio windows) without CLIP candidate
    scoring — the text-QA downstream eval only needs the transcript."""
    key = ingestion_cache_key(
        video_path=config.video_path,
        sample_count=None,
        audio_window_seconds=None,
        processing_mode=config.processing_mode.value,
    )
    ingested = pipeline.cache.get_ingestion(key)
    if ingested is None:
        ingested = pipeline.ingestor.ingest(
            video_path=config.video_path,
            sample_count=None,
            audio_window_seconds=None,
            processing_mode=config.processing_mode,
        )
        pipeline.cache.set_ingestion(key, ingested)
    return ingested


def _full_transcript_selected(ingested, config) -> list[SelectedCandidate]:
    """Transcribe every audio window (cache-backed) for a true whole-transcript."""
    if not ingested.audio_windows:
        return []
    transcriber = FasterWhisperTranscriber(
        model_size=config.whisper_model_size or "base",
        device=config.whisper_device or "cpu",
        compute_type=config.whisper_compute_type or "int8",
        beam_size=config.whisper_beam_size or 1,
        max_windows=None,
    )
    try:
        transcripts = transcriber.transcribe_windows(ingested.audio_windows)
    except Exception:
        return []
    selected: list[SelectedCandidate] = []
    windows = sorted(ingested.audio_windows, key=lambda w: w.start_seconds)
    rank = 1
    for window in windows:
        text = (transcripts.get(window.path) or "").strip()
        if not text:
            continue
        end = window.start_seconds + window.duration_seconds
        selected.append(
            SelectedCandidate(
                id=f"whole-{window.index}",
                modality=Modality.AUDIO,
                timestamp_seconds=window.start_seconds + window.duration_seconds / 2,
                text=text,
                scene_start_seconds=window.start_seconds,
                scene_end_seconds=end,
                selection_rank=rank,
                relevance_score=0.0,
                normalized_score=0.0,
                mmr_score=0.0,
                source_score_type="whole",
                reason="full transcript window",
            )
        )
        rank += 1
    return selected


def _selected_to_candidates(selected: list[SelectedCandidate]) -> list[Candidate]:
    return [
        Candidate(
            id=item.id,
            timestamp_seconds=item.timestamp_seconds,
            text=item.text,
            saliency_score=item.relevance_score,
            scene_start_seconds=item.scene_start_seconds,
            scene_end_seconds=item.scene_end_seconds,
        )
        for item in selected
    ]


def run_downstream_case(
    case: QualityCase,
    pipeline: LocalCompressionPipeline,
    gateway: OllamaTextGateway,
    correct_threshold: float,
    judge: LlmJudge | None = None,
) -> DownstreamCaseResult | None:
    config = resolve_case_config(case)
    ingested = _ingest_only(pipeline, config)

    # "whole" = the full transcript (every audio window). No CLIP needed: the
    # text answerer only reads transcript, so we skip candidate scoring entirely.
    whole_selected = _full_transcript_selected(ingested, config)
    if not whole_selected:
        return None  # Not transcript-answerable; skip.

    gist_compression = _load_compression(case.compression_path)
    gist_selected = list(gist_compression.selected)

    conditions: dict[str, CaseConditionResult] = {}
    for condition in CONDITIONS:
        if condition == "gist":
            selected = gist_selected
        elif condition == "whole":
            selected = whole_selected
        else:  # uniform: evenly spaced over the full transcript at the Gist budget
            budget = max(len(gist_selected), 1)
            selected = _uniform_select(
                visual_candidates=[], audio_candidates=_selected_to_candidates(whole_selected),
                budget=budget,
            )

        response = _response_for(config.query, selected)
        prompt = build_evidence_prompt(response)
        try:
            answer = gateway.answer(GatewayRequest(query=config.query, compression=response)).answer
        except Exception:
            # A single slow/failed LLM call must not crash the whole suite.
            answer = ""
        recall = _term_recall(case.expected_answer_terms, answer)
        judge_correct: bool | None = None
        judge_score: float | None = None
        judge_reason: str | None = None
        if judge is not None:
            try:
                verdict = judge.judge(config.query, case.expected_answer_terms, answer)
                judge_correct, judge_score, judge_reason = (
                    verdict.correct, verdict.score, verdict.reason,
                )
            except JudgeError:
                judge_correct, judge_score, judge_reason = None, None, "judge unavailable"
        conditions[condition] = CaseConditionResult(
            condition=condition,
            answer=answer,
            answer_term_recall=recall,
            correct=recall >= correct_threshold,
            context_tokens=_approx_tokens(prompt),
            evidence_items=len(selected),
            judge_correct=judge_correct,
            judge_score=judge_score,
            judge_reason=judge_reason,
        )

    return DownstreamCaseResult(
        case_id=case.id,
        query_category=case.query_category.value if case.query_category else None,
        domain=case.domain,
        query=config.query,
        conditions=conditions,
    )


def _transcript_quality(value: str | None):
    from gist.audio.whisper import TranscriptQuality

    return TranscriptQuality(value) if value else TranscriptQuality.BALANCED


def _summarize(condition: str, results: list[DownstreamCaseResult]) -> ConditionSummary:
    rows = [r.conditions[condition] for r in results if condition in r.conditions]
    n = len(rows)

    def avg(fn) -> float:
        return 0.0 if not n else sum(fn(r) for r in rows) / n

    judged = [r for r in rows if r.judge_correct is not None]
    judge_rate = (
        None if not judged else sum(1.0 for r in judged if r.judge_correct) / len(judged)
    )
    return ConditionSummary(
        condition=condition,
        label=CONDITION_LABELS[condition],
        cases=n,
        avg_answer_term_recall=avg(lambda r: r.answer_term_recall),
        correct_rate=avg(lambda r: 1.0 if r.correct else 0.0),
        avg_context_tokens=avg(lambda r: r.context_tokens),
        judge_correct_rate=judge_rate,
    )


def run_downstream_suite(
    cases: list[QualityCase],
    output_root: Path = Path(".gist/downstream"),
    model: str = DEFAULT_OLLAMA_MODEL,
    correct_threshold: float = 0.5,
    num_ctx: int = 16384,
    judge_model: str | None = None,
    progress=None,
) -> DownstreamReport:
    pipeline = LocalCompressionPipeline(output_root=output_root)
    # Whole-transcript calls (10k+ tokens) can take minutes on CPU; give headroom.
    gateway = OllamaTextGateway(model=model, num_ctx=num_ctx, timeout_seconds=600.0)
    judge = LlmJudge(model=judge_model) if judge_model else None
    results: list[DownstreamCaseResult] = []
    for index, case in enumerate(cases, start=1):
        if progress is not None:
            progress(f"[{index}/{len(cases)}] {case.id}")
        result = run_downstream_case(case, pipeline, gateway, correct_threshold, judge=judge)
        if result is not None:
            results.append(result)
    summaries = {c: _summarize(c, results) for c in CONDITIONS}
    return DownstreamReport(
        cases=len(results),
        answerer=f"ollama:{model}",
        correct_threshold=correct_threshold,
        summaries=summaries,
        results=results,
    )


def render_downstream_markdown(report: DownstreamReport) -> str:
    lines = [
        "# Gist Downstream QA Evaluation",
        "",
        f"- Cases (transcript-answerable): {report.cases}",
        f"- Answerer: {report.answerer}",
        f"- Correct threshold: answer term recall >= {report.correct_threshold:.2f}",
        "- Same LLM answers three contexts per case; only the context differs.",
        "",
        "| Context | Answer recall | Correct rate | LLM-judge correct | Avg context tokens |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for c in CONDITIONS:
        s = report.summaries[c]
        judge = "n/a" if s.judge_correct_rate is None else f"{s.judge_correct_rate:.0%}"
        lines.append(
            f"| {s.label} | {s.avg_answer_term_recall:.2f} "
            f"| {s.correct_rate:.0%} | {judge} | {s.avg_context_tokens:.0f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Downstream QA evaluation: whole vs uniform vs Gist context."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(".gist/downstream"))
    parser.add_argument("--json", type=Path, dest="json_output")
    parser.add_argument("--markdown", type=Path, dest="markdown_output")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append")
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Do not restrict to transcript-answerable query categories.",
    )
    parser.add_argument("--correct-threshold", type=float, default=0.5)
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=16384,
        help="Ollama context window; must fit the whole-transcript condition.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Enable LLM-judge semantic scoring with this Ollama model (e.g. llama3.2:3b).",
    )
    args = parser.parse_args(argv)

    cases = load_quality_cases(args.dataset)
    if not args.all_categories:
        cases = [
            c
            for c in cases
            if c.query_category is not None
            and c.query_category.value in TRANSCRIPT_CATEGORIES
        ]
    if args.case_id:
        wanted = set(args.case_id)
        cases = [c for c in cases if c.id in wanted]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("no transcript-answerable cases selected")

    report = run_downstream_suite(
        cases,
        output_root=args.output_root,
        model=args.model,
        correct_threshold=args.correct_threshold,
        num_ctx=args.num_ctx,
        judge_model=args.judge_model,
        progress=lambda m: print(m, flush=True),
    )

    if args.json_output is not None:
        report.write_json(args.json_output)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_downstream_markdown(report))

    print(f"cases={report.cases} answerer={report.answerer}")
    for c in CONDITIONS:
        s = report.summaries[c]
        judge = "" if s.judge_correct_rate is None else f"judge_correct={s.judge_correct_rate:.0%}, "
        print(
            f"{c}: answer_recall={s.avg_answer_term_recall:.2f}, "
            f"correct_rate={s.correct_rate:.0%}, {judge}"
            f"avg_context_tokens={s.avg_context_tokens:.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
