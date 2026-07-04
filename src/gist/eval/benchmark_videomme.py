"""External benchmark: Video-MME long subset (with-subtitles / text setting).

Runs real Video-MME long-video multiple-choice questions through the same
whole-vs-uniform-vs-Gist comparison as the internal downstream eval, but scored
against the benchmark's *gold* answers instead of hand-authored terms. Because
the answerer is text-only, this is the standard Video-MME "with subtitles"
setting: the model reads transcript-derived context and picks A/B/C/D.

The "gist" condition is Gist's transcript selection (the compressor over audio
candidates) — faithful for a text answerer and avoids CLIP entirely.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib import error, request

from pydantic import BaseModel

from gist.audio.whisper import FasterWhisperTranscriber
from gist.core.cache import ingestion_cache_key
from gist.core.compressor import GistCompressor
from gist.core.presets import CompressionPreset
from gist.core.schemas import Candidate, CompressionRequest, Modality, SelectedCandidate
from gist.core.scoring import lexical_relevance
from gist.core.token_estimation import TokenEstimatorProfile
from gist.eval.baselines import _uniform_select
from gist.media.longform import ProcessingMode
from gist.pipeline import LocalCompressionPipeline

CONDITIONS = ("whole", "uniform", "gist")
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"


class BenchQuestion(BaseModel):
    question_id: str
    videoID: str
    question: str
    options: list[str]
    answer: str  # gold letter


class BenchConditionResult(BaseModel):
    condition: str
    predicted: str
    correct: bool
    context_tokens: int


class BenchQuestionResult(BaseModel):
    question_id: str
    videoID: str
    gold: str
    conditions: dict[str, BenchConditionResult]


class BenchConditionSummary(BaseModel):
    condition: str
    cases: int
    accuracy: float
    avg_context_tokens: float


class BenchReport(BaseModel):
    benchmark: str = "Video-MME (long, with-subtitles/text)"
    answerer: str
    questions: int
    videos: int
    summaries: dict[str, BenchConditionSummary]
    results: list[BenchQuestionResult]

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _parse_options(raw) -> list[str]:
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw.replace("'", '"'))
    except Exception:
        return [p.strip() for p in str(raw).split("',") if p.strip()]


def load_questions(path: Path, video_ids: set[str] | None) -> list[BenchQuestion]:
    rows = json.loads(path.read_text())
    out: list[BenchQuestion] = []
    for r in rows:
        if video_ids is not None and r["videoID"] not in video_ids:
            continue
        out.append(
            BenchQuestion(
                question_id=r["question_id"],
                videoID=r["videoID"],
                question=r["question"],
                options=_parse_options(r["options"]),
                answer=str(r["answer"]).strip()[:1].upper(),
            )
        )
    return out


def _full_transcript(
    pipeline: LocalCompressionPipeline, video_path: Path, transcribe_model: str = "base"
) -> list[Candidate]:
    key = ingestion_cache_key(
        video_path=video_path, sample_count=None, audio_window_seconds=None,
        processing_mode=ProcessingMode.AUTO.value,
    )
    ingested = pipeline.cache.get_ingestion(key)
    if ingested is None:
        ingested = pipeline.ingestor.ingest(
            video_path=video_path, sample_count=None, audio_window_seconds=None,
            processing_mode=ProcessingMode.AUTO,
        )
        pipeline.cache.set_ingestion(key, ingested)
    transcriber = FasterWhisperTranscriber(model_size=transcribe_model, device="cpu",
                                           compute_type="int8", beam_size=1, max_windows=None)
    transcripts = transcriber.transcribe_windows(ingested.audio_windows)
    cands: list[Candidate] = []
    for w in sorted(ingested.audio_windows, key=lambda x: x.start_seconds):
        text = (transcripts.get(w.path) or "").strip()
        if not text:
            continue
        cands.append(Candidate(
            id=f"w{w.index}", timestamp_seconds=w.start_seconds + w.duration_seconds / 2,
            text=text, scene_start_seconds=w.start_seconds,
            scene_end_seconds=w.start_seconds + w.duration_seconds,
        ))
    return cands


def _cand_to_selected(c: Candidate, rank: int, score: float = 0.0) -> SelectedCandidate:
    return SelectedCandidate(
        id=c.id, modality=Modality.AUDIO, timestamp_seconds=c.timestamp_seconds, text=c.text,
        scene_start_seconds=c.scene_start_seconds, scene_end_seconds=c.scene_end_seconds,
        selection_rank=rank, relevance_score=score, normalized_score=score, mmr_score=score,
        source_score_type="benchmark", reason="benchmark context",
    )


def _gist_select(question: str, transcript: list[Candidate]) -> list[SelectedCandidate]:
    scored = [
        c.model_copy(update={"saliency_score": lexical_relevance(question, c)})
        for c in transcript
    ]
    response = GistCompressor().compress(CompressionRequest(
        video_id="videomme", query=question, duration_seconds=max(
            (c.scene_end_seconds or c.timestamp_seconds for c in transcript), default=1.0),
        preset=CompressionPreset.BALANCED, adaptive_budget=True, decompose_query=True,
        token_estimator=TokenEstimatorProfile.GENERIC, task_aware_selection=True,
        visual_candidates=[], audio_candidates=scored,
    ))
    return list(response.selected)


def _mc_prompt(question: str, options: list[str], context_items: list[SelectedCandidate]) -> str:
    evidence = "\n".join(
        f"- [{c.scene_start_seconds:.0f}s] {c.text}" for c in context_items
    ) or "(no transcript evidence)"
    opts = "\n".join(options)
    return (
        "You answer a multiple-choice question about a video using ONLY the transcript "
        "evidence below. Pick the single best option.\n"
        f"Transcript evidence:\n{evidence}\n\n"
        f"Question: {question}\nOptions:\n{opts}\n"
        "Respond with ONLY the letter (A, B, C, or D) of the best answer."
    )


def _ollama_answer(prompt: str, model: str, num_ctx: int, timeout: float) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.0, "num_ctx": num_ctx}}
    req = request.Request(f"{DEFAULT_OLLAMA_URL}/api/generate",
                          data=json.dumps(payload).encode(), method="POST",
                          headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode()).get("response", "")
    except (error.URLError, TimeoutError, json.JSONDecodeError):
        return ""


def _parse_letter(text: str) -> str:
    m = re.search(r"\b([ABCD])\b", text.upper())
    return m.group(1) if m else ""


def run_benchmark(
    questions: list[BenchQuestion],
    video_dir: Path,
    output_root: Path = Path(".gist/benchmark"),
    model: str = DEFAULT_MODEL,
    num_ctx: int = 16384,
    transcribe_model: str = "base",
    progress=None,
) -> BenchReport:
    pipeline = LocalCompressionPipeline(output_root=output_root)
    by_video: dict[str, list[BenchQuestion]] = {}
    for q in questions:
        by_video.setdefault(q.videoID, []).append(q)

    results: list[BenchQuestionResult] = []
    for vid, qs in by_video.items():
        video_path = video_dir / f"videomme-{vid}.mp4"
        if not video_path.exists():
            if progress:
                progress(f"skip {vid}: video not downloaded")
            continue
        if progress:
            progress(f"transcribing {vid} ({len(qs)} questions)")
        transcript = _full_transcript(pipeline, video_path, transcribe_model)
        whole_selected = [_cand_to_selected(c, i) for i, c in enumerate(transcript, 1)]
        for q in qs:
            if progress:
                progress(f"answering {q.question_id}")
            gist_selected = _gist_select(q.question, transcript)
            budget = max(len(gist_selected), 1)
            uniform_selected = _uniform_select(
                visual_candidates=[],
                audio_candidates=[c for c in transcript], budget=budget,
            )
            conds: dict[str, BenchConditionResult] = {}
            for cond, sel in (("whole", whole_selected), ("uniform", uniform_selected),
                              ("gist", gist_selected)):
                prompt = _mc_prompt(q.question, q.options, sel)
                pred = _parse_letter(_ollama_answer(prompt, model, num_ctx, 600.0))
                conds[cond] = BenchConditionResult(
                    condition=cond, predicted=pred, correct=(pred == q.answer),
                    context_tokens=_approx_tokens(prompt),
                )
            results.append(BenchQuestionResult(
                question_id=q.question_id, videoID=vid, gold=q.answer, conditions=conds))

    summaries = {}
    for cond in CONDITIONS:
        rows = [r.conditions[cond] for r in results if cond in r.conditions]
        n = len(rows)
        summaries[cond] = BenchConditionSummary(
            condition=cond, cases=n,
            accuracy=0.0 if not n else sum(r.correct for r in rows) / n,
            avg_context_tokens=0.0 if not n else sum(r.context_tokens for r in rows) / n,
        )
    return BenchReport(
        answerer=f"ollama:{model}", questions=len(results),
        videos=len({r.videoID for r in results}), summaries=summaries, results=results,
    )


def render_markdown(report: BenchReport) -> str:
    lines = [
        "# Gist on Video-MME (long subset, with-subtitles/text setting)",
        "",
        f"- Answerer: {report.answerer}",
        f"- Questions: {report.questions} across {report.videos} videos",
        "- Real Video-MME multiple-choice questions scored against gold answers.",
        "",
        "| Context | Accuracy | Avg context tokens |",
        "| :--- | ---: | ---: |",
    ]
    labels = {"whole": "Whole transcript", "uniform": "Uniform sampling",
              "gist": "Gist-compressed"}
    for c in CONDITIONS:
        s = report.summaries[c]
        lines.append(f"| {labels[c]} | {s.accuracy:.0%} | {s.avg_context_tokens:.0f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Gist on a Video-MME long-subset slice.")
    parser.add_argument("--questions", type=Path, default=Path(".gist/benchmark/videomme_long.json"))
    parser.add_argument("--video-dir", type=Path, default=Path(".gist/videos/archive"))
    parser.add_argument("--video-id", action="append", help="Restrict to these videoIDs.")
    parser.add_argument("--json", type=Path, dest="json_output")
    parser.add_argument("--markdown", type=Path, dest="markdown_output")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--transcribe-model", default="base")
    args = parser.parse_args(argv)

    video_ids = set(args.video_id) if args.video_id else None
    questions = load_questions(args.questions, video_ids)
    if not questions:
        raise SystemExit("no questions selected")
    report = run_benchmark(
        questions, video_dir=args.video_dir, model=args.model, num_ctx=args.num_ctx,
        transcribe_model=args.transcribe_model,
        progress=lambda m: print(m, flush=True),
    )
    if args.json_output:
        report.write_json(args.json_output)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report))
    print(f"questions={report.questions} videos={report.videos} answerer={report.answerer}")
    for c in CONDITIONS:
        s = report.summaries[c]
        print(f"{c}: accuracy={s.accuracy:.0%} avg_context_tokens={s.avg_context_tokens:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
