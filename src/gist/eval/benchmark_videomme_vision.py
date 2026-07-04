"""External benchmark: Video-MME long subset, VISION setting.

This is the fair test of Gist as a *video* system. A vision-language model
answers real Video-MME multiple-choice questions from FRAMES only (no
transcript), under three frame budgets:

- ``dense``   -- many uniformly-sampled frames (the expensive "look at lots" baseline).
- ``uniform`` -- few uniformly-sampled frames (cheap, query-blind).
- ``gist``    -- few Gist-selected, query-relevant frames (Gist's core: CLIP frame selection).

The question it answers: at a small frame budget, does Gist's query-aware frame
selection beat blind uniform sampling, and approach the dense baseline?
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from urllib import error, request

from pydantic import BaseModel

from gist.core.compressor import GistCompressor
from gist.core.modes import AudioScoringMode, VisualScoringMode
from gist.core.presets import CompressionPreset
from gist.core.schemas import CompressionRequest
from gist.core.token_estimation import TokenEstimatorProfile
from gist.eval.benchmark_videomme import (
    BenchQuestion,
    _parse_letter,
    load_questions,
)
from gist.media.longform import ProcessingMode
from gist.pipeline import LocalCompressionPipeline

CONDITIONS = ("dense", "uniform", "gist")
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_VISION_MODEL = "llava:7b"


class VisionConditionResult(BaseModel):
    condition: str
    predicted: str
    correct: bool
    frames: int


class VisionQuestionResult(BaseModel):
    question_id: str
    videoID: str
    gold: str
    conditions: dict[str, VisionConditionResult]


class VisionConditionSummary(BaseModel):
    condition: str
    cases: int
    accuracy: float
    avg_frames: float


class VisionReport(BaseModel):
    benchmark: str = "Video-MME (long, vision)"
    answerer: str
    questions: int
    videos: int
    dense_budget: int
    small_budget: int
    summaries: dict[str, VisionConditionSummary]
    results: list[VisionQuestionResult]

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))


def _uniform_frame_paths(frame_paths: list[Path], budget: int) -> list[Path]:
    if budget <= 0 or not frame_paths:
        return []
    if len(frame_paths) <= budget:
        return frame_paths
    step = (len(frame_paths) - 1) / (budget - 1) if budget > 1 else 0
    return [frame_paths[round(i * step)] for i in range(budget)]


def _gist_frame_paths(
    pipeline: LocalCompressionPipeline,
    compressor: GistCompressor,
    video_path: Path,
    question: str,
    budget: int,
) -> list[Path]:
    ingested, candidates, _raw = pipeline.prepare_candidates(
        video_path=video_path,
        query=question,
        sample_count=None,
        audio_window_seconds=None,
        processing_mode=ProcessingMode.AUTO,
        visual_scorer=VisualScoringMode.CLIP_SCENE,
        audio_scorer=AudioScoringMode.BASELINE,
        visual_ocr=True,
    )
    response = compressor.compress(
        CompressionRequest(
            video_id=ingested.video_id,
            query=question,
            duration_seconds=ingested.metadata.duration_seconds,
            preset=CompressionPreset.BALANCED,
            adaptive_budget=True,
            decompose_query=True,
            token_estimator=TokenEstimatorProfile.GENERIC,
            task_aware_selection=True,
            visual_candidates=candidates.visual,
            audio_candidates=[],
        )
    )
    paths = [
        item.asset_path
        for item in response.selected
        if item.asset_path is not None and item.asset_path.exists()
    ]
    if len(paths) < budget:
        # Top-up from the highest-saliency visual candidates so Gist gets a fair
        # frame budget comparable to the uniform conditions.
        ranked = sorted(
            candidates.visual,
            key=lambda c: (c.saliency_score or 0.0),
            reverse=True,
        )
        for c in ranked:
            if c.asset_path is not None and c.asset_path.exists() and c.asset_path not in paths:
                paths.append(c.asset_path)
            if len(paths) >= budget:
                break
    return paths[:budget]


def _all_frame_paths(pipeline: LocalCompressionPipeline, video_path: Path) -> list[Path]:
    from gist.core.cache import ingestion_cache_key

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
    return [f.path for f in sorted(ingested.frames, key=lambda x: x.index) if f.path.exists()]


def _mc_vision_prompt(question: str, options: list[str]) -> str:
    opts = "\n".join(options)
    return (
        "You are shown frames sampled from a video. Answer the multiple-choice "
        "question using only what is visible in these frames.\n"
        f"Question: {question}\nOptions:\n{opts}\n"
        "Respond with ONLY the letter (A, B, C, or D) of the best answer."
    )


def _ollama_vision_answer(
    prompt: str, image_paths: list[Path], model: str, timeout: float = 600.0
) -> str:
    images = []
    for p in image_paths:
        try:
            images.append(base64.b64encode(p.read_bytes()).decode("ascii"))
        except OSError:
            continue
    payload = {
        "model": model,
        "prompt": prompt,
        "images": images,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    req = request.Request(
        f"{DEFAULT_OLLAMA_URL}/api/generate",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode()).get("response", "")
    except (error.URLError, TimeoutError, json.JSONDecodeError):
        return ""


def run_vision_benchmark(
    questions: list[BenchQuestion],
    video_dir: Path,
    output_root: Path = Path(".gist/benchmark-vision"),
    model: str = DEFAULT_VISION_MODEL,
    dense_budget: int = 12,
    small_budget: int = 6,
    progress=None,
) -> VisionReport:
    pipeline = LocalCompressionPipeline(output_root=output_root)
    compressor = GistCompressor()
    by_video: dict[str, list[BenchQuestion]] = {}
    for q in questions:
        by_video.setdefault(q.videoID, []).append(q)

    results: list[VisionQuestionResult] = []
    for vid, qs in by_video.items():
        video_path = video_dir / f"videomme-{vid}.mp4"
        if not video_path.exists():
            if progress:
                progress(f"skip {vid}: video not downloaded")
            continue
        all_frames = _all_frame_paths(pipeline, video_path)
        dense_frames = _uniform_frame_paths(all_frames, dense_budget)
        uniform_frames = _uniform_frame_paths(all_frames, small_budget)
        for q in qs:
            if progress:
                progress(f"gist frame selection {q.question_id}")
            gist_frames = _gist_frame_paths(pipeline, compressor, video_path, q.question, small_budget)
            conds: dict[str, VisionConditionResult] = {}
            for cond, frames in (("dense", dense_frames), ("uniform", uniform_frames),
                                 ("gist", gist_frames)):
                if progress:
                    progress(f"answering {q.question_id} [{cond}] ({len(frames)} frames)")
                prompt = _mc_vision_prompt(q.question, q.options)
                pred = _parse_letter(_ollama_vision_answer(prompt, frames, model))
                conds[cond] = VisionConditionResult(
                    condition=cond, predicted=pred, correct=(pred == q.answer), frames=len(frames),
                )
            results.append(VisionQuestionResult(
                question_id=q.question_id, videoID=vid, gold=q.answer, conditions=conds))

    summaries = {}
    for cond in CONDITIONS:
        rows = [r.conditions[cond] for r in results if cond in r.conditions]
        n = len(rows)
        summaries[cond] = VisionConditionSummary(
            condition=cond, cases=n,
            accuracy=0.0 if not n else sum(r.correct for r in rows) / n,
            avg_frames=0.0 if not n else sum(r.frames for r in rows) / n,
        )
    return VisionReport(
        answerer=f"ollama:{model}", questions=len(results),
        videos=len({r.videoID for r in results}),
        dense_budget=dense_budget, small_budget=small_budget,
        summaries=summaries, results=results,
    )


def render_markdown(report: VisionReport) -> str:
    labels = {"dense": f"Dense uniform ({report.dense_budget} frames)",
              "uniform": f"Uniform ({report.small_budget} frames)",
              "gist": f"Gist-selected ({report.small_budget} frames)"}
    lines = [
        "# Gist on Video-MME (long subset, VISION setting)",
        "",
        f"- Answerer: {report.answerer} (vision-language model)",
        f"- Questions: {report.questions} across {report.videos} videos",
        "- Frames only (no transcript); real Video-MME questions scored against gold.",
        "",
        "| Frame budget | Accuracy | Avg frames |",
        "| :--- | ---: | ---: |",
    ]
    for c in CONDITIONS:
        s = report.summaries[c]
        lines.append(f"| {labels[c]} | {s.accuracy:.0%} | {s.avg_frames:.0f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Gist frame selection on Video-MME (vision).")
    parser.add_argument("--questions", type=Path, default=Path(".gist/benchmark/videomme_long.json"))
    parser.add_argument("--video-dir", type=Path, default=Path(".gist/videos/archive"))
    parser.add_argument("--video-id", action="append")
    parser.add_argument("--json", type=Path, dest="json_output")
    parser.add_argument("--markdown", type=Path, dest="markdown_output")
    parser.add_argument("--model", default=DEFAULT_VISION_MODEL)
    parser.add_argument("--dense-budget", type=int, default=12)
    parser.add_argument("--small-budget", type=int, default=6)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    video_ids = set(args.video_id) if args.video_id else None
    questions = load_questions(args.questions, video_ids)
    if args.limit is not None:
        questions = questions[: args.limit]
    if not questions:
        raise SystemExit("no questions selected")
    report = run_vision_benchmark(
        questions, video_dir=args.video_dir, model=args.model,
        dense_budget=args.dense_budget, small_budget=args.small_budget,
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
        print(f"{c}: accuracy={s.accuracy:.0%} avg_frames={s.avg_frames:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
