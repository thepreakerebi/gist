#!/usr/bin/env python3
"""Draft candidate questions for the curated long-video corpus and screen them so
only questions that genuinely need BOTH modalities survive.

Why the screen exists
---------------------
Questions written from a transcript are answerable from the transcript. If the
corpus fills with those, Whisper alone scores full marks, the visual pathway is
never exercised, and the joint budget arbitration that Gist is built around goes
untested. Drafting is therefore only half the job; the screen is the other half.

Each candidate is answered three times by the same model:

  transcript only   speech, no frames
  frames only       frames, no speech
  both              the full evidence

A candidate is kept only when *both* answers correctly and *neither* single
modality does. Anything else is rejected, with the reason recorded so the
rejection can be audited rather than trusted.

This produces drafts for human review. It does not produce ground truth. Every
surviving candidate still needs a person to confirm the answer and the timestamp
before it enters the dataset; `curate_long_video_query_proposal` then grounds it
against a real pipeline run.

Usage
-----
    uv run python scripts/draft_corpus_questions.py \
        --manifest data/eval/long-video-sources.json \
        --output data/eval/corpus-questions.draft.json \
        --per-video 6 --keep 4
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gist.audio.whisper import FasterWhisperTranscriber
from gist.gateway.openai_vision import (
    DEFAULT_MODEL,
    OpenAIVisionGatewayError,
    extract_output_text,
    post_openai_response,
)
from gist.media.ingestion import MediaIngestor
from gist.media.longform import ProcessingMode
from gist.media.models import AudioWindow, ExtractedFrame, IngestedVideo

DRAFT_FRAMES = 16
SCREEN_FRAMES = 12
TIMEOUT_SECONDS = 180.0

# The corpus taxonomy, from data/eval/long-video-quality.jsonl. Not invented here:
# the ablation reports and RQ4 are defined over these exact names.
QUERY_CATEGORIES = [
    "speech_semantic",
    "visual_object_action",
    "mixed_av",
    "temporal_before_after",
    "global_summary",
]

CATEGORY_BRIEFS = {
    "mixed_av": (
        "The answer requires BOTH what is said and what is on screen at the same "
        "moment. The canonical shape is: something the speaker says ABOUT something "
        "simultaneously visible. Neither the transcript nor the frames alone suffice."
    ),
    "speech_semantic": (
        "The answer is carried by what is said, understood rather than quoted verbatim."
    ),
    "visual_object_action": (
        "The answer is carried by an object or action visible on screen."
    ),
    "temporal_before_after": "The answer depends on the order of two events.",
    "global_summary": "The answer requires summarising across a long span.",
}


@dataclass
class Candidate:
    question: str
    options: list[str]
    answer: str
    query: str
    timestamp_seconds: float
    query_category: str
    why_both_modalities: str
    screen: dict[str, Any] = field(default_factory=dict)
    verdict: str = "unscreened"
    reason: str = ""


def api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        legacy = Path(".gist/.openai_key")
        if legacy.exists():
            key = legacy.read_text().strip()
    if not key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "or export the variable."
        )
    return key


def data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def ask(
    *,
    instructions: str,
    text: str,
    frames: list[Path],
    key: str,
    model: str,
    json_output: bool,
) -> str:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
    for frame in frames:
        content.append({"type": "input_image", "image_url": data_url(frame)})
    payload: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": [{"role": "user", "content": content}],
    }
    if json_output:
        payload["text"] = {"format": {"type": "json_object"}}
    response = post_openai_response(payload, api_key=key, timeout_seconds=TIMEOUT_SECONDS)
    return extract_output_text(response)


def frange(start: float, stop: float, step: float) -> list[float]:
    out, value = [], start
    while value < stop:
        out.append(value)
        value += step
    return out


def evenly_spaced(items: list[Any], count: int) -> list[Any]:
    if len(items) <= count:
        return list(items)
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]


def transcript_text(
    windows: list[AudioWindow], transcripts: dict[Path, str], limit_chars: int = 24000
) -> str:
    lines: list[str] = []
    for window in windows:
        said = (transcripts.get(window.path) or "").strip()
        if said:
            minutes, seconds = divmod(int(window.start_seconds), 60)
            lines.append(f"[{minutes:02d}:{seconds:02d}] {said}")
    joined = "\n".join(lines)
    if len(joined) > limit_chars:
        joined = joined[:limit_chars] + "\n[transcript truncated]"
    return joined


def frame_catalogue(frames: list[ExtractedFrame]) -> str:
    stamps = []
    for frame in frames:
        minutes, seconds = divmod(int(frame.timestamp_seconds), 60)
        stamps.append(f"{minutes:02d}:{seconds:02d}")
    return ", ".join(stamps)


DRAFT_INSTRUCTIONS = """You write evaluation questions for a long-video question
answering benchmark. You will be shown a transcript and a set of sampled frames
from one recording.

Write multiple-choice questions that CANNOT be answered from the transcript alone
and CANNOT be answered from the frames alone. Each must require combining
something said with something seen. A question whose answer appears verbatim in
the transcript is useless here, and so is one answerable from a single frame.

Give exactly four options, one correct. Keep questions factual and checkable
against a specific moment. Never ask about anything you are unsure of.

Also give, for each question, a `query`: the same thing asked as a natural
open-ended question a person would type, with no options. This is what the
evaluation corpus stores; the multiple-choice form exists only so the question can
be screened automatically.

Return JSON: {"questions": [{"question": str, "options": [str, str, str, str],
"answer": "A"|"B"|"C"|"D", "query": str, "timestamp_seconds": number,
"query_category": str, "why_both_modalities": str}]}"""

SCREEN_INSTRUCTIONS = """Answer the multiple-choice question from the evidence
provided. If the evidence provided is insufficient to determine the answer, reply
exactly UNANSWERABLE. Do not guess. Reply with a single letter, or UNANSWERABLE,
and nothing else."""


def window_transcript(
    windows: list[AudioWindow],
    transcripts: dict[Path, str],
    start: float,
    end: float,
) -> str:
    lines = []
    for window in windows:
        if not (start <= window.start_seconds < end):
            continue
        said = (transcripts.get(window.path) or "").strip()
        if said:
            minutes, seconds = divmod(int(window.start_seconds), 60)
            lines.append(f"[{minutes:02d}:{seconds:02d}] {said}")
    return "\n".join(lines)


def draft_for_video(
    *,
    video_path: Path,
    title: str,
    ingestion: IngestedVideo,
    transcript: str,
    per_video: int,
    key: str,
    model: str,
    category: str | None = None,
    segment: tuple[float, float] | None = None,
    transcripts: dict[Path, str] | None = None,
) -> list[Candidate]:
    """Draft candidates, optionally confined to one segment of the recording.

    Segment drafting exists because the first run of this tool sampled 16 frames
    across 96 minutes and the model confabulated specifics: five of eight
    candidates were rejected for unreliable ground truth rather than for modality
    failure. Confining a draft to a five or ten minute window lets the same frame
    budget cover it densely, so the model is describing what it can actually see.
    """
    if segment is not None:
        start, end = segment
        pool = [f for f in ingestion.frames if start <= f.timestamp_seconds < end]
        frames = evenly_spaced(pool or ingestion.frames, DRAFT_FRAMES)
        text = (
            window_transcript(ingestion.audio_windows, transcripts or {}, start, end)
            if transcripts
            else transcript
        )
        scope = (
            f"Segment: {start / 60:.1f} to {end / 60:.1f} minutes. Every question must be "
            f"answerable from THIS segment, and its timestamp must fall inside it."
        )
    else:
        frames = evenly_spaced(ingestion.frames, DRAFT_FRAMES)
        text = transcript
        scope = "Spread the questions across the whole recording."

    if category:
        brief = CATEGORY_BRIEFS.get(category, "")
        ask_for = (
            f"Every question must be of category '{category}'. {brief}\n"
            f"Set query_category to '{category}' on every question."
        )
    else:
        ask_for = (
            "Spread them across these categories where the material allows: "
            + ", ".join(QUERY_CATEGORIES)
        )

    prompt = (
        f"Recording: {title}\n"
        f"Duration: {ingestion.metadata.duration_seconds / 60:.1f} minutes\n"
        f"{scope}\n"
        f"Frames supplied, at: {frame_catalogue(frames)}\n\n"
        f"Write {per_video} questions. {ask_for}\n\n"
        f"TRANSCRIPT\n{text}\n\n"
        "Respond with a json object in the schema given."
    )
    raw = ask(
        instructions=DRAFT_INSTRUCTIONS,
        text=prompt,
        frames=[f.path for f in frames],
        key=key,
        model=model,
        json_output=True,
    )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  ! drafting returned non-JSON for {title}", file=sys.stderr)
        return []
    candidates = []
    for item in payload.get("questions", []):
        try:
            candidates.append(
                Candidate(
                    question=str(item["question"]),
                    options=[str(o) for o in item["options"]][:4],
                    answer=str(item["answer"]).strip().upper()[:1],
                    query=str(item.get("query") or item["question"]),
                    timestamp_seconds=float(item.get("timestamp_seconds", 0.0)),
                    query_category=str(item.get("query_category", "")),
                    why_both_modalities=str(item.get("why_both_modalities", "")),
                )
            )
        except (KeyError, ValueError, TypeError):
            continue
    return candidates


def question_block(candidate: Candidate) -> str:
    letters = "ABCD"
    lines = [candidate.question]
    for letter, option in zip(letters, candidate.options, strict=False):
        lines.append(f"{letter}. {option}")
    return "\n".join(lines)


def screen(
    *,
    candidate: Candidate,
    ingestion: IngestedVideo,
    transcript: str,
    key: str,
    model: str,
) -> Candidate:
    window = 180.0
    near = [
        f
        for f in ingestion.frames
        if abs(f.timestamp_seconds - candidate.timestamp_seconds) <= window
    ]
    frames = evenly_spaced(near or ingestion.frames, SCREEN_FRAMES)
    frame_paths = [f.path for f in frames]
    block = question_block(candidate)

    conditions = {
        "transcript_only": (f"TRANSCRIPT\n{transcript}\n\nQUESTION\n{block}", []),
        "frames_only": (f"QUESTION\n{block}", frame_paths),
        "both": (f"TRANSCRIPT\n{transcript}\n\nQUESTION\n{block}", frame_paths),
    }
    results: dict[str, str] = {}
    for name, (text, images) in conditions.items():
        try:
            reply = ask(
                instructions=SCREEN_INSTRUCTIONS,
                text=text,
                frames=images,
                key=key,
                model=model,
                json_output=False,
            ).strip().upper()
        except OpenAIVisionGatewayError as exc:
            results[name] = f"ERROR: {exc}"
            continue
        results[name] = reply[:1] if reply[:1] in "ABCD" else reply[:20]

    candidate.screen = results
    correct = candidate.answer
    if results.get("both") != correct:
        candidate.verdict = "rejected"
        candidate.reason = "both-modality answer was wrong; ground truth is unreliable"
    elif results.get("transcript_only") == correct:
        candidate.verdict = "rejected"
        candidate.reason = "speech alone answers it; does not test visual selection"
    elif results.get("frames_only") == correct:
        candidate.verdict = "rejected"
        candidate.reason = "frames alone answer it; does not test audio selection"
    else:
        candidate.verdict = "kept"
        candidate.reason = "needs both modalities"
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/eval/long-video-sources.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/eval/corpus-questions.draft.json")
    )
    parser.add_argument("--artifact-root", type=Path, default=Path(".gist/corpus-drafting"))
    parser.add_argument("--per-video", type=int, default=6, help="candidates drafted per recording")
    parser.add_argument(
        "--keep", type=int, default=4, help="hard cap on kept questions per recording"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--whisper-max-windows", type=int, default=240)
    parser.add_argument("--only", action="append", help="substring match on title; repeatable")
    parser.add_argument(
        "--video", type=Path, help="draft against one local file, ignoring the manifest"
    )
    parser.add_argument(
        "--title", default=None, help="title to record when --video is used"
    )
    parser.add_argument(
        "--domain", default=None, help="domain to record when --video is used"
    )
    parser.add_argument(
        "--category", choices=QUERY_CATEGORIES, help="draft only this category"
    )
    parser.add_argument(
        "--segment-minutes",
        type=float,
        default=0.0,
        help="draft per segment of this length with dense frames (0 = whole recording)",
    )
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    random.seed(args.seed)
    key = api_key()
    if args.video:
        if not args.video.exists():
            raise SystemExit(f"no such file: {args.video}")
        sources = [
            {
                "video_id": args.video.stem,
                "title": args.title or args.video.stem,
                "domain": args.domain or "",
                "local_video_path": str(args.video),
                "duration_seconds": 0,
            }
        ]
    else:
        sources = json.loads(args.manifest.read_text())
    ingestor = MediaIngestor(output_root=args.artifact_root)
    transcriber = FasterWhisperTranscriber(max_windows=args.whisper_max_windows)

    out: list[dict[str, Any]] = []
    for source in sources:
        title = source.get("title", "untitled")
        if args.only and not any(o.lower() in title.lower() for o in args.only):
            continue
        local = source.get("local_video_path")
        if not local or not Path(local).exists():
            print(f"- skip (no local file): {title}")
            continue
        video_path = Path(local)
        print(f"- {title} ({source.get('duration_seconds', 0) / 60:.1f} min)")

        print("    ingesting")
        ingestion = ingestor.ingest(
            video_path=video_path,
            sample_count=None,
            audio_window_seconds=None,
            processing_mode=ProcessingMode.AUTO,
        )
        print(f"    frames={len(ingestion.frames)} windows={len(ingestion.audio_windows)}")

        print("    transcribing")
        transcripts = transcriber.transcribe_windows(ingestion.audio_windows)
        transcript = transcript_text(ingestion.audio_windows, transcripts)
        if not transcript.strip():
            print("    ! empty transcript, skipping")
            continue

        duration = ingestion.metadata.duration_seconds
        if args.segment_minutes > 0:
            step = args.segment_minutes * 60
            segments = [(s, min(s + step, duration)) for s in frange(0, duration, step)]
        else:
            segments = [None]

        per_segment = max(1, round(args.per_video / len(segments))) if segments else args.per_video
        print(
            f"    drafting {args.per_video} candidates "
            f"over {len(segments)} segment(s)"
            + (f", category={args.category}" if args.category else "")
        )
        candidates = []
        for segment in segments:
            if len(candidates) >= args.per_video:
                break
            candidates.extend(
                draft_for_video(
                    video_path=video_path,
                    title=title,
                    ingestion=ingestion,
                    transcript=transcript,
                    per_video=per_segment,
                    key=key,
                    model=args.model,
                    category=args.category,
                    segment=segment,
                    transcripts=transcripts,
                )
            )
        candidates = candidates[: args.per_video]
        print(f"    screening {len(candidates)}")
        kept = 0
        for candidate in candidates:
            screen(
                candidate=candidate,
                ingestion=ingestion,
                transcript=transcript,
                key=key,
                model=args.model,
            )
            if candidate.verdict == "kept":
                if kept >= args.keep:
                    candidate.verdict = "rejected"
                    candidate.reason = f"per-recording cap of {args.keep} already met"
                else:
                    kept += 1
            mark = "keep" if candidate.verdict == "kept" else "drop"
            print(f"      [{mark}] {candidate.question[:64]} — {candidate.reason}")
            out.append(
                {
                    "video_id": source.get("video_id"),
                    "title": title,
                    "domain": source.get("domain"),
                    "local_video_path": local,
                    "question": candidate.question,
                    "options": candidate.options,
                    "answer": candidate.answer,
                    "query": candidate.query,
                    "timestamp_seconds": candidate.timestamp_seconds,
                    "query_category": candidate.query_category,
                    "why_both_modalities": candidate.why_both_modalities,
                    "screen": candidate.screen,
                    "verdict": candidate.verdict,
                    "reason": candidate.reason,
                    "human_verified": False,
                }
            )
        print(f"    kept {kept}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    kept_total = sum(1 for o in out if o["verdict"] == "kept")
    print(f"\n{kept_total} kept of {len(out)} drafted -> {args.output}")
    print("Every kept question still needs human verification before it enters the dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
