"""Data access for the Gist video library and chat history.

Everything that knows Gist's table layout lives here. The rest of the codebase
sees domain objects — :class:`VideoRecord`, and plain
:class:`gist.core.schemas.Candidate` lists ready for the compressor.

The interesting function is :func:`score_candidates`. Because frame and audio
embeddings are persisted at ingestion, answering a query does not re-encode
anything: it embeds the query text once, then lets pgvector compute cosine
similarity against every stored candidate in a single statement. The rows come
back already shaped as scored candidates, which is exactly what the selector
consumes — so the per-query cost drops from "re-run Whisper and CLIP over an
hour of video" to one indexed scan.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from gist.core.schemas import Candidate
from gist.db.connection import cursor

# Cosine distance in pgvector is 1 - cosine_similarity, so similarity is
# 1 - distance. CLIP/CLAP similarities are small positive numbers in practice;
# the selector z-normalizes per modality anyway, so absolute scale is not
# load-bearing here.
_SIMILARITY = "1 - (embedding <=> %(vector)s::vector)"


@dataclass(frozen=True, slots=True)
class VideoRecord:
    id: str
    youtube_url: str
    youtube_id: str | None
    title: str
    duration_seconds: float
    thumbnail_url: str | None
    source_path: str | None
    status: str
    status_detail: str | None
    progress: float
    frame_count: int
    audio_window_count: int
    error: str | None
    created_at: datetime
    ingested_at: datetime | None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> VideoRecord:
        return cls(
            id=str(row["id"]),
            youtube_url=row["youtube_url"],
            youtube_id=row.get("youtube_id"),
            title=row["title"],
            duration_seconds=row["duration_seconds"],
            thumbnail_url=row.get("thumbnail_url"),
            source_path=row.get("source_path"),
            status=row["status"],
            status_detail=row.get("status_detail"),
            progress=row.get("progress") or 0.0,
            frame_count=row.get("frame_count") or 0,
            audio_window_count=row.get("audio_window_count") or 0,
            error=row.get("error"),
            created_at=row["created_at"],
            ingested_at=row.get("ingested_at"),
        )


# ---------------------------------------------------------------- videos ----


def create_video(youtube_url: str, *, title: str, youtube_id: str | None = None) -> VideoRecord:
    """Register a video as pending ingestion, or return the existing row.

    Re-submitting a URL already in the library is the common case (a user pastes
    a link they forgot they had), so this is an upsert rather than an error.
    """

    with cursor() as cur:
        cur.execute(
            """
            insert into videos (youtube_url, youtube_id, title, duration_seconds, status)
            values (%(url)s, %(yid)s, %(title)s, 0, 'pending')
            on conflict (youtube_url) do update set youtube_url = excluded.youtube_url
            returning *
            """,
            {"url": youtube_url, "yid": youtube_id, "title": title},
        )
        return VideoRecord.from_row(cur.fetchone())


def get_video(video_id: str) -> VideoRecord | None:
    with cursor() as cur:
        cur.execute("select * from videos where id = %s", (video_id,))
        row = cur.fetchone()
        return VideoRecord.from_row(row) if row else None


def get_video_by_url(youtube_url: str) -> VideoRecord | None:
    with cursor() as cur:
        cur.execute("select * from videos where youtube_url = %s", (youtube_url,))
        row = cur.fetchone()
        return VideoRecord.from_row(row) if row else None


def list_videos(limit: int = 100) -> list[VideoRecord]:
    with cursor() as cur:
        cur.execute("select * from videos order by created_at desc limit %s", (limit,))
        return [VideoRecord.from_row(row) for row in cur.fetchall()]


def update_video_status(
    video_id: str,
    *,
    status: str | None = None,
    status_detail: str | None = None,
    progress: float | None = None,
    error: str | None = None,
) -> None:
    """Patch ingestion progress. Only non-None fields are written."""

    fields: list[str] = []
    params: dict[str, Any] = {"id": video_id}
    for name, value in (
        ("status", status),
        ("status_detail", status_detail),
        ("progress", progress),
        ("error", error),
    ):
        if value is not None:
            fields.append(f"{name} = %({name})s")
            params[name] = value
    if not fields:
        return

    with cursor() as cur:
        cur.execute(f"update videos set {', '.join(fields)} where id = %(id)s", params)


def mark_video_ready(
    video_id: str,
    *,
    duration_seconds: float,
    frame_count: int,
    audio_window_count: int,
    transcript: str | None,
    source_path: str | None,
    thumbnail_url: str | None = None,
) -> None:
    with cursor() as cur:
        cur.execute(
            """
            update videos set
                status = 'ready',
                status_detail = null,
                progress = 1.0,
                error = null,
                duration_seconds = %(duration)s,
                frame_count = %(frames)s,
                audio_window_count = %(windows)s,
                transcript = %(transcript)s,
                source_path = coalesce(%(source)s, source_path),
                thumbnail_url = coalesce(%(thumb)s, thumbnail_url),
                ingested_at = now()
            where id = %(id)s
            """,
            {
                "id": video_id,
                "duration": duration_seconds,
                "frames": frame_count,
                "windows": audio_window_count,
                "transcript": transcript,
                "source": source_path,
                "thumb": thumbnail_url,
            },
        )


def update_video_metadata(
    video_id: str,
    *,
    title: str | None = None,
    duration_seconds: float | None = None,
    thumbnail_url: str | None = None,
    source_path: str | None = None,
) -> None:
    fields: list[str] = []
    params: dict[str, Any] = {"id": video_id}
    for name, value in (
        ("title", title),
        ("duration_seconds", duration_seconds),
        ("thumbnail_url", thumbnail_url),
        ("source_path", source_path),
    ):
        if value is not None:
            fields.append(f"{name} = %({name})s")
            params[name] = value
    if not fields:
        return

    with cursor() as cur:
        cur.execute(f"update videos set {', '.join(fields)} where id = %(id)s", params)


def delete_video(video_id: str) -> None:
    """Remove a video and everything cascading from it."""

    with cursor() as cur:
        cur.execute("delete from videos where id = %s", (video_id,))


def get_transcript(video_id: str) -> str | None:
    with cursor() as cur:
        cur.execute("select transcript from videos where id = %s", (video_id,))
        row = cur.fetchone()
        return row["transcript"] if row else None


# ------------------------------------------------------------ candidates ----


def replace_frames(video_id: str, frames: Sequence[dict[str, Any]]) -> None:
    """Write a video's frame candidates, replacing any previous ingestion."""

    with cursor() as cur:
        cur.execute("delete from frames where video_id = %s", (video_id,))
        if not frames:
            return
        cur.executemany(
            """
            insert into frames (
                video_id, frame_index, timestamp_seconds, asset_path, ocr_text,
                scene_start_seconds, scene_end_seconds, embedding
            ) values (
                %(video_id)s, %(frame_index)s, %(timestamp_seconds)s, %(asset_path)s,
                %(ocr_text)s, %(scene_start_seconds)s, %(scene_end_seconds)s,
                %(embedding)s::vector
            )
            """,
            [{"video_id": video_id, **frame} for frame in frames],
        )


def replace_audio_windows(video_id: str, windows: Sequence[dict[str, Any]]) -> None:
    """Write a video's audio candidates, replacing any previous ingestion."""

    with cursor() as cur:
        cur.execute("delete from audio_windows where video_id = %s", (video_id,))
        if not windows:
            return
        cur.executemany(
            """
            insert into audio_windows (
                video_id, window_index, start_seconds, end_seconds,
                transcript_text, asset_path, embedding
            ) values (
                %(video_id)s, %(window_index)s, %(start_seconds)s, %(end_seconds)s,
                %(transcript_text)s, %(asset_path)s, %(embedding)s::vector
            )
            """,
            [{"video_id": video_id, **window} for window in windows],
        )


def score_candidates(
    video_id: str,
    *,
    visual_vector: Sequence[float] | None,
    audio_vector: Sequence[float] | None,
) -> tuple[list[Candidate], list[Candidate]]:
    """Score every stored candidate for a video against the query embeddings.

    Returns ``(visual, audio)`` candidate lists carrying ``saliency_score``, which
    is precisely the shape :class:`gist.core.compressor.GistCompressor` expects —
    the selector treats a supplied saliency score as the relevance signal and
    never re-derives it.

    A ``None`` vector means that modality has no query embedding (its encoder is
    unavailable), in which case its candidates come back unscored and the
    selector falls back to lexical relevance over their text.
    """

    return (
        _score_frames(video_id, visual_vector),
        _score_audio_windows(video_id, audio_vector),
    )


def _score_frames(video_id: str, vector: Sequence[float] | None) -> list[Candidate]:
    score = _SIMILARITY if vector is not None else "null"
    with cursor() as cur:
        cur.execute(
            f"""
            select frame_index, timestamp_seconds, asset_path, ocr_text,
                   scene_start_seconds, scene_end_seconds,
                   case when embedding is null then null else {score} end as score
            from frames
            where video_id = %(video_id)s
            order by timestamp_seconds
            """,
            {"video_id": video_id, "vector": _as_vector(vector)},
        )
        return [
            Candidate(
                id=f"frame-{row['frame_index']}",
                timestamp_seconds=row["timestamp_seconds"],
                text=row["ocr_text"] or "",
                saliency_score=row["score"],
                asset_path=row["asset_path"],
                scene_start_seconds=row["scene_start_seconds"],
                scene_end_seconds=row["scene_end_seconds"],
            )
            for row in cur.fetchall()
        ]


def _score_audio_windows(video_id: str, vector: Sequence[float] | None) -> list[Candidate]:
    score = _SIMILARITY if vector is not None else "null"
    with cursor() as cur:
        cur.execute(
            f"""
            select window_index, start_seconds, end_seconds, transcript_text, asset_path,
                   case when embedding is null then null else {score} end as score
            from audio_windows
            where video_id = %(video_id)s
            order by start_seconds
            """,
            {"video_id": video_id, "vector": _as_vector(vector)},
        )
        return [
            Candidate(
                id=f"audio-{row['window_index']}",
                timestamp_seconds=row["start_seconds"],
                text=row["transcript_text"] or "",
                saliency_score=row["score"],
                asset_path=row["asset_path"],
            )
            for row in cur.fetchall()
        ]


def _as_vector(vector: Sequence[float] | None) -> str | None:
    """Render a Python sequence as a pgvector literal."""

    if vector is None:
        return None
    return "[" + ",".join(f"{value:.8f}" for value in vector) + "]"


# --------------------------------------------------------- conversations ----


def create_conversation(video_id: str, title: str | None = None) -> str:
    with cursor() as cur:
        cur.execute(
            "insert into conversations (video_id, title) values (%s, %s) returning id",
            (video_id, title),
        )
        return str(cur.fetchone()["id"])


def latest_conversation(video_id: str) -> str | None:
    with cursor() as cur:
        cur.execute(
            "select id from conversations where video_id = %s order by created_at desc limit 1",
            (video_id,),
        )
        row = cur.fetchone()
        return str(row["id"]) if row else None


def ensure_conversation(video_id: str) -> str:
    return latest_conversation(video_id) or create_conversation(video_id)


def append_message(
    conversation_id: str,
    *,
    role: str,
    query: str | None = None,
    answer: str | None = None,
    answer_provider: str | None = None,
    selected_evidence: Any = None,
    metrics: Any = None,
    clips: Any = None,
) -> str:
    with cursor() as cur:
        cur.execute(
            """
            insert into messages (
                conversation_id, role, query, answer, answer_provider,
                selected_evidence, metrics, clips
            ) values (
                %(conversation_id)s, %(role)s, %(query)s, %(answer)s, %(provider)s,
                %(evidence)s, %(metrics)s, %(clips)s
            ) returning id
            """,
            {
                "conversation_id": conversation_id,
                "role": role,
                "query": query,
                "answer": answer,
                "provider": answer_provider,
                "evidence": _as_json(selected_evidence),
                "metrics": _as_json(metrics),
                "clips": _as_json(clips),
            },
        )
        return str(cur.fetchone()["id"])


def list_messages(conversation_id: str) -> list[dict[str, Any]]:
    with cursor() as cur:
        cur.execute(
            "select * from messages where conversation_id = %s order by created_at",
            (conversation_id,),
        )
        return [dict(row) for row in cur.fetchall()]


def _as_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, default=str)
