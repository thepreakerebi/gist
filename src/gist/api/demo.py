"""Streaming demo endpoint for the Gist live web app.

Runs the full local pipeline on CPU, streaming scoring progress to the browser
as Server-Sent Events, then answers with a selectable hosted multimodal LLM
(OpenAI or Claude). The hosted answerer is used for demo reliability only; the
paper's measured efficiency/FLOP claims come from Qwen2.5-Omni-7B offline.
"""

import json
import os
import queue
import subprocess
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import Field, model_validator

from gist.api.schemas import LocalVideoCompressionRequest
from gist.gateway.hosted_answerer import answer_with_hosted_llm
from gist.gateway.schemas import GatewayRequest
from gist.pipeline import LocalCompressionPipeline

demo_router = APIRouter(prefix="/v1/demo", tags=["demo"])

# Live-safety guardrail: reject clips longer than this so an arbitrary upload
# cannot blow latency/RAM on the free 16 GB Space.
DEMO_MAX_DURATION_SECONDS = 240.0
_KEY_FILES = {
    "OPENAI_API_KEY": Path(".gist/.openai_key"),
    "ANTHROPIC_API_KEY": Path(".gist/.anthropic_key"),
}


class DemoRunRequest(LocalVideoCompressionRequest):
    """A demo run: either a local video_path or a video_url to download."""

    # Optional here (the base requires it): a URL run supplies video_url instead.
    video_path: Path | None = None  # type: ignore[assignment]
    video_url: str | None = None
    answerer: Literal["openai", "claude", "extractive"] = "openai"
    answerer_model: str | None = None
    max_frames: Annotated[int, Field(gt=0, le=16)] = 8

    @model_validator(mode="after")
    def _require_a_source(self) -> "DemoRunRequest":
        if not self.video_path and not self.video_url:
            raise ValueError("provide either video_path or video_url")
        return self


@demo_router.post("/run")
def demo_run(request: DemoRunRequest) -> StreamingResponse:
    return StreamingResponse(
        _run_stream(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _run_stream(request: DemoRunRequest) -> Iterator[str]:
    events: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()

    def emit(event: str, data: dict[str, Any]) -> None:
        events.put((event, data))

    worker = threading.Thread(target=_run_worker, args=(request, emit), daemon=True)
    worker.start()

    while True:
        item = events.get()
        if item is None:
            break
        event, data = item
        if event == "__end__":
            break
        yield _sse(event, data)


def _run_worker(request: DemoRunRequest, emit) -> None:
    tmp_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        video_path = request.video_path
        if request.video_url:
            # Probe the URL's duration BEFORE downloading so an over-cap video is
            # rejected instantly instead of after a slow full download.
            url_duration = _probe_url_duration_seconds(request.video_url)
            if url_duration is not None and url_duration > DEMO_MAX_DURATION_SECONDS:
                emit("error", {"message": _too_long_message(url_duration)})
                return
            emit("progress", {"stage": "downloading", "message": "downloading video"})
            tmp_dir = tempfile.TemporaryDirectory(prefix="gist-demo-dl-")
            video_path = _download_video(request.video_url, Path(tmp_dir.name))

        duration = _probe_duration_seconds(video_path)
        if duration is not None and duration > DEMO_MAX_DURATION_SECONDS:
            emit("error", {"message": _too_long_message(duration)})
            return

        emit("progress", {"stage": "starting", "message": "preparing pipeline"})

        def progress(message: str) -> None:
            emit("progress", {"stage": "pipeline", "message": message})

        def on_candidates(candidate_set) -> None:
            emit(
                "scored",
                {
                    "visual": [_candidate_point(c) for c in candidate_set.visual],
                    "audio": [_candidate_point(c) for c in candidate_set.audio],
                },
            )

        ingestion, compression = LocalCompressionPipeline(
            output_root=request.output_root
        ).run(
            video_path=video_path,
            query=request.query,
            preset=request.preset,
            sample_count=request.sample_count,
            audio_window_seconds=request.audio_window_seconds,
            processing_mode=request.processing_mode,
            visual_scorer=request.visual_scorer,
            audio_scorer=request.audio_scorer,
            adaptive_budget=request.adaptive_budget,
            decompose_query=request.decompose_query,
            token_estimator=request.token_estimator,
            progress=progress,
            on_candidates=on_candidates,
        )

        answer = compression.answer
        provider = compression.answer_provider or "extractive"
        if request.answerer in ("openai", "claude"):
            emit(
                "progress",
                {"stage": "answering", "message": f"asking {request.answerer}"},
            )
            gateway_response = answer_with_hosted_llm(
                GatewayRequest(query=request.query, compression=compression),
                answerer=request.answerer,
                model=request.answerer_model,
                api_key=_load_api_key(request.answerer),
                max_frames=request.max_frames,
                # Intent-aware prompt: forbids outside knowledge (answer must be
                # grounded in Gist's selected evidence, not the model's priors)
                # and prioritizes transcript for speech-semantic queries.
                prompt_strategy="intent",
            )
            answer = gateway_response.answer
            provider = gateway_response.provider

        emit(
            "done",
            {
                "answer": answer,
                "provider": provider,
                "compression": compression.model_dump(mode="json"),
                "video": {
                    "duration_seconds": ingestion.metadata.duration_seconds,
                    "frame_count": len(ingestion.frames),
                    "audio_window_count": len(ingestion.audio_windows),
                },
            },
        )
    except Exception as exc:  # surface any failure to the client stream
        emit("error", {"message": f"{type(exc).__name__}: {exc}"})
    finally:
        if tmp_dir is not None:
            tmp_dir.cleanup()
        emit("__end__", {})


def _candidate_point(candidate: Any) -> dict[str, Any]:
    return {
        "timestamp_seconds": candidate.timestamp_seconds,
        "score": candidate.saliency_score if candidate.saliency_score is not None else 0.0,
        "text": (candidate.text or "")[:160],
    }


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _load_api_key(answerer: str) -> str | None:
    env_name = "OPENAI_API_KEY" if answerer == "openai" else "ANTHROPIC_API_KEY"
    key = os.getenv(env_name)
    if key:
        return key
    key_file = _KEY_FILES.get(env_name)
    if key_file and key_file.exists():
        return key_file.read_text().strip() or None
    return None


def _too_long_message(duration: float) -> str:
    return (
        f"This video is {duration / 60:.1f} min ({duration:.0f}s); the live demo "
        f"is capped at {DEMO_MAX_DURATION_SECONDS / 60:.0f} min so scoring stays "
        "fast on CPU. Try a shorter clip or a preset."
    )


def _probe_url_duration_seconds(url: str) -> float | None:
    """Fetch a remote video's duration via yt-dlp without downloading it."""
    try:
        completed = subprocess.run(
            ["yt-dlp", "--no-warnings", "--skip-download", "--print", "%(duration)s", url],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    try:
        return float(completed.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def _download_video(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "video.mp4"
    template = output_path.with_suffix(".%(ext)s")
    command = [
        "yt-dlp",
        "--no-playlist",
        "-f",
        "bv*[height<=360]+ba/b[height<=360]/best[height<=360]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        str(template),
        url,
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"video download failed: {completed.stderr.strip()}")
    for candidate in sorted(output_dir.glob("video.*")):
        if candidate.suffix.lower() == ".mp4":
            return candidate
    matches = sorted(output_dir.glob("video.*"))
    if matches:
        return matches[0]
    raise RuntimeError("video download produced no file")


def _probe_duration_seconds(video_path: Path) -> float | None:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None
