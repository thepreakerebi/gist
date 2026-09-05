"""HTTP surface for the video library.

The flow this serves is deliberately two-phase. ``POST /v1/library/videos``
starts an ingestion that runs the whole query-independent half of the pipeline
once; ``POST /v1/library/videos/{id}/query`` then answers questions against the
stored result. Ingestion of an hour-long video outlives any reasonable request
timeout, so it runs on a background thread with progress written to the
database — a client that reloads mid-ingestion still sees the real state.
"""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from gist.core.presets import CompressionPreset
from gist.db import connection as db
from gist.db import repository as repo
from gist.api.demo import _load_api_key
from gist.gateway.hosted_answerer import answer_with_hosted_llm
from gist.gateway.schemas import GatewayRequest
from gist.library import ingest as ingest_service
from gist.library import media
from gist.library import query as query_service

library_router = APIRouter(prefix="/v1/library", tags=["library"])

# Ingestions in flight in this process, so a duplicate submit does not start a
# second encode of the same video.
_active: set[str] = set()
_active_lock = threading.Lock()


class AddVideoRequest(BaseModel):
    url: str = Field(min_length=1)

    @field_validator("url")
    @classmethod
    def _normalize(cls, value: str) -> str:
        url = value.strip()
        if not url.startswith(("http://", "https://")):
            raise ValueError("provide a full http(s) video URL")
        return url


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    answerer: str = "openai"
    preset: CompressionPreset = CompressionPreset.BALANCED
    adaptive_budget: bool = True
    decompose_query: bool = True
    tail_merging: bool = False

    @field_validator("query")
    @classmethod
    def _normalize(cls, value: str) -> str:
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized


def _require_db() -> None:
    if not db.is_configured():
        raise HTTPException(
            status_code=503,
            detail="The video library needs DATABASE_URL set to a Neon connection string.",
        )


def _video_payload(record: repo.VideoRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "url": record.youtube_url,
        "youtube_id": record.youtube_id,
        "title": record.title,
        "duration_seconds": record.duration_seconds,
        "thumbnail_url": record.thumbnail_url,
        "status": record.status,
        "status_detail": record.status_detail,
        "progress": record.progress,
        "frame_count": record.frame_count,
        "audio_window_count": record.audio_window_count,
        "error": record.error,
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }


# ---------------------------------------------------------------- library ----


@library_router.get("/videos")
def list_videos() -> dict[str, Any]:
    _require_db()
    return {"videos": [_video_payload(record) for record in repo.list_videos()]}


@library_router.get("/videos/{video_id}")
def get_video(video_id: str) -> dict[str, Any]:
    _require_db()
    record = repo.get_video(video_id)
    if record is None:
        raise HTTPException(status_code=404, detail="video not found")

    conversation_id = repo.latest_conversation(video_id)
    messages = repo.list_messages(conversation_id) if conversation_id else []
    return {
        "video": _video_payload(record),
        "conversation_id": conversation_id,
        "messages": [_message_payload(message) for message in messages],
    }


def _message_payload(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(message["id"]),
        "role": message["role"],
        "query": message.get("query"),
        "answer": message.get("answer"),
        "answer_provider": message.get("answer_provider"),
        "selected_evidence": message.get("selected_evidence"),
        "metrics": message.get("metrics"),
        "clips": message.get("clips"),
        "created_at": message["created_at"].isoformat() if message.get("created_at") else None,
    }


@library_router.post("/videos", status_code=202)
def add_video(request: AddVideoRequest) -> dict[str, Any]:
    """Register a video and start ingesting it in the background."""

    _require_db()

    existing = repo.get_video_by_url(request.url)
    if existing is not None and existing.status in {"ready", "ingesting"}:
        return {"video": _video_payload(existing), "started": False}

    record = repo.create_video(
        request.url,
        title=media.youtube_id(request.url) or request.url,
        youtube_id=media.youtube_id(request.url),
    )
    _start_ingestion(record.id, request.url)
    return {"video": _video_payload(record), "started": True}


@library_router.delete("/videos/{video_id}", status_code=204)
def delete_video(video_id: str) -> None:
    _require_db()
    if repo.get_video(video_id) is None:
        raise HTTPException(status_code=404, detail="video not found")
    ingest_service.delete_video_assets(video_id)
    repo.delete_video(video_id)


def _start_ingestion(video_id: str, url: str) -> None:
    with _active_lock:
        if video_id in _active:
            return
        _active.add(video_id)

    def run() -> None:
        try:
            ingest_service.ingest_video(video_id, url)
        except Exception:  # noqa: BLE001 - already recorded on the row
            pass
        finally:
            with _active_lock:
                _active.discard(video_id)

    threading.Thread(target=run, daemon=True).start()


@library_router.get("/videos/{video_id}/events")
def ingestion_events(video_id: str) -> StreamingResponse:
    """Stream ingestion progress until the video is ready or failed.

    Progress is polled from the database rather than held in memory so that the
    stream reports correctly even when the ingestion was started by a different
    worker or before a reload.
    """

    _require_db()
    if repo.get_video(video_id) is None:
        raise HTTPException(status_code=404, detail="video not found")

    def stream() -> Iterator[str]:
        import time

        last: tuple[str, str | None, float] | None = None
        while True:
            record = repo.get_video(video_id)
            if record is None:
                yield _sse("error", {"message": "video was removed"})
                return

            state = (record.status, record.status_detail, record.progress)
            if state != last:
                last = state
                yield _sse("progress", _video_payload(record))

            if record.status in {"ready", "failed"}:
                yield _sse("done", _video_payload(record))
                return
            time.sleep(1.0)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------ query ----


@library_router.post("/videos/{video_id}/query")
def run_query(video_id: str, request: QueryRequest) -> StreamingResponse:
    """Answer a question about an ingested video, streaming the stages."""

    _require_db()
    record = repo.get_video(video_id)
    if record is None:
        raise HTTPException(status_code=404, detail="video not found")
    if record.status != "ready":
        raise HTTPException(
            status_code=409,
            detail=f"video is {record.status}; wait for ingestion to finish",
        )

    return StreamingResponse(
        _query_stream(record, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _query_stream(record: repo.VideoRecord, request: QueryRequest) -> Iterator[str]:
    events: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()

    def emit(event: str, data: dict[str, Any]) -> None:
        events.put((event, data))

    worker = threading.Thread(
        target=_query_worker, args=(record, request, emit), daemon=True
    )
    worker.start()

    while True:
        item = events.get()
        if item is None:
            break
        event, data = item
        if event == "__end__":
            break
        yield _sse(event, data)


def _query_worker(
    record: repo.VideoRecord,
    request: QueryRequest,
    emit: Any,
) -> None:
    try:
        conversation_id = repo.ensure_conversation(record.id)
        repo.append_message(conversation_id, role="user", query=request.query)
        emit("started", {"conversation_id": conversation_id, "query": request.query})

        emit("stage", {"stage": "scoring", "label": "Scoring stored evidence"})
        visual, audio = query_service.score_video(record.id, request.query)
        emit(
            "scored",
            {
                "candidates": [
                    _candidate_point(candidate, "visual") for candidate in visual
                ]
                + [_candidate_point(candidate, "audio") for candidate in audio],
            },
        )

        emit("stage", {"stage": "selecting", "label": "Compressing to key evidence"})
        compression = query_service.compress(
            record.id,
            request.query,
            duration_seconds=record.duration_seconds,
            preset=request.preset,
            adaptive_budget=request.adaptive_budget,
            decompose_query=request.decompose_query,
            tail_merging=request.tail_merging,
            visual=visual,
            audio=audio,
        )
        emit(
            "selected",
            {
                "selected": [item.model_dump(mode="json") for item in compression.selected],
                "metrics": compression.metrics.model_dump(mode="json"),
                "query_intent": compression.query_intent,
            },
        )

        clips: list[dict[str, Any]] = []
        if record.source_path and Path(record.source_path).exists():
            emit("stage", {"stage": "clips", "label": "Cutting evidence clips"})
            clips = query_service.cut_evidence_clips(
                record.id,
                compression,
                source_path=Path(record.source_path),
                duration_seconds=record.duration_seconds,
            )
            emit("clips", {"clips": [_clip_payload(record.id, clip) for clip in clips]})

        emit("stage", {"stage": "answering", "label": "Answering from the evidence"})
        answer, provider = _answer(request, compression)

        repo.append_message(
            conversation_id,
            role="assistant",
            answer=answer,
            answer_provider=provider,
            selected_evidence=[item.model_dump(mode="json") for item in compression.selected],
            metrics=compression.metrics.model_dump(mode="json"),
            clips=[_clip_payload(record.id, clip) for clip in clips],
        )

        emit(
            "done",
            {
                "answer": answer,
                "answer_provider": provider,
                "metrics": compression.metrics.model_dump(mode="json"),
                "clips": [_clip_payload(record.id, clip) for clip in clips],
            },
        )
    except Exception as exc:  # noqa: BLE001 - reported to the client
        emit("error", {"message": str(exc) or exc.__class__.__name__})
    finally:
        emit("__end__", {})


def _answer(request: QueryRequest, compression: Any) -> tuple[str | None, str | None]:
    if request.answerer not in {"openai", "claude"}:
        # Extractive: the compressor already produced an answer from the
        # selected evidence, with no hosted model involved.
        return compression.answer, compression.answer_provider or "extractive"

    try:
        response = answer_with_hosted_llm(
            GatewayRequest(query=compression.query, compression=compression),
            answerer=request.answerer,
            api_key=_load_api_key(request.answerer),
            # Intent-aware prompt: the answer must be grounded in Gist's
            # selected evidence rather than the model's own priors.
            prompt_strategy="intent",
        )
        return response.answer, response.provider
    except Exception as exc:  # noqa: BLE001
        # An answerer failure must not discard the compression result: the
        # selected evidence is the part that demonstrates the method.
        return f"(answerer unavailable: {exc})", None


def _candidate_point(candidate: Any, modality: str) -> dict[str, Any]:
    return {
        "id": candidate.id,
        "modality": modality,
        "timestamp_seconds": candidate.timestamp_seconds,
        "score": candidate.saliency_score,
        "text": candidate.text,
    }


def _clip_payload(video_id: str, clip: dict[str, Any]) -> dict[str, Any]:
    payload = dict(clip)
    payload["url"] = f"/v1/library/videos/{video_id}/clips/{Path(clip['path']).name}"
    payload.pop("path", None)
    return payload


@library_router.get("/videos/{video_id}/clips/{name}")
def get_clip(video_id: str, name: str) -> FileResponse:
    """Serve a cut evidence clip.

    FileResponse handles HTTP range requests, which is what lets the browser
    scrub the clip instead of downloading it whole before playing.
    """

    # Reject traversal: only a bare filename inside this video's clip directory.
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="invalid clip name")

    path = ingest_service.video_directory(video_id) / query_service.CLIP_CACHE_DIRNAME / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="clip not found")
    return FileResponse(path, media_type="video/mp4")


@library_router.get("/videos/{video_id}/frames/{name}")
def get_frame(video_id: str, name: str) -> FileResponse:
    """Serve a sampled frame image referenced by selected evidence."""

    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="invalid frame name")

    for directory in ingest_service.video_directory(video_id).rglob("frames"):
        candidate = directory / name
        if candidate.exists():
            return FileResponse(candidate, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="frame not found")


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
