"""Query-dependent selection over a already-ingested video.

The fast half of the split. Ingestion has already paid for the encoders, so
answering a question costs one text embedding per modality, a pgvector scan,
the selector, and the answerer — no frame decoding, no Whisper, no image tower.

Critically, this must produce the *same* selection the offline pipeline would.
The stored frame vectors are bit-identical to what ``score_frames`` computes
live (pinned by ``tests/test_stored_embedding_equivalence.py``), and the same
``GistCompressor`` runs on the result, so the library is a caching layer over
the measured pipeline rather than a second implementation of it.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gist.core.compressor import GistCompressor
from gist.core.presets import CompressionPreset
from gist.core.schemas import Candidate, CompressionRequest, CompressionResponse
from gist.db import repository as repo
from gist.library.ingest import video_directory
from gist.media.clips import adaptive_clip_span

ProgressCallback = Callable[[str, dict[str, Any]], None]

# Evidence clips are cut around the selected timestamp rather than stored ahead
# of time: which span matters is not known until the selector has run.
CLIP_CACHE_DIRNAME = "clips"


@dataclass(frozen=True, slots=True)
class QueryEmbeddings:
    visual: list[float] | None
    audio: list[float] | None


def embed_query(query: str) -> QueryEmbeddings:
    """Embed the query into CLIP's and CLAP's spaces.

    Either encoder may be unavailable (optional extras, no GPU, a load
    failure). A missing vector is not fatal: the repository returns that
    modality's candidates unscored and the selector falls back to lexical
    relevance over their transcript or OCR text.
    """

    return QueryEmbeddings(visual=_embed_visual(query), audio=_embed_audio(query))


def _embed_visual(query: str) -> list[float] | None:
    try:
        from gist.vision.clip import HuggingFaceClipFrameScorer

        return HuggingFaceClipFrameScorer().embed_text(query)
    except Exception:  # noqa: BLE001
        return None


def _embed_audio(query: str) -> list[float] | None:
    try:
        from gist.audio.clap import HuggingFaceClapAudioScorer

        return HuggingFaceClapAudioScorer().embed_text(query)
    except Exception:  # noqa: BLE001
        return None


def score_video(video_id: str, query: str) -> tuple[list[Candidate], list[Candidate]]:
    """Score every stored candidate for a video against the query."""

    embeddings = embed_query(query)
    return repo.score_candidates(
        video_id,
        visual_vector=embeddings.visual,
        audio_vector=embeddings.audio,
    )


def compress(
    video_id: str,
    query: str,
    *,
    duration_seconds: float,
    preset: CompressionPreset = CompressionPreset.BALANCED,
    adaptive_budget: bool = True,
    decompose_query: bool = True,
    tail_merging: bool = False,
    visual: list[Candidate] | None = None,
    audio: list[Candidate] | None = None,
) -> CompressionResponse:
    """Run the selector over stored candidates for one query."""

    if visual is None or audio is None:
        visual, audio = score_video(video_id, query)

    request = CompressionRequest(
        video_id=video_id,
        query=query,
        duration_seconds=max(duration_seconds, 1.0),
        preset=preset,
        adaptive_budget=adaptive_budget,
        decompose_query=decompose_query,
        task_aware_selection=True,
        tail_merging=tail_merging,
        visual_candidates=visual,
        audio_candidates=audio,
    )
    return GistCompressor().compress(request)


def cut_evidence_clips(
    video_id: str,
    compression: CompressionResponse,
    *,
    source_path: Path,
    duration_seconds: float,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Cut a short clip around each selected evidence timestamp.

    Stream-copy (``-c copy``) rather than re-encode: cutting is near-instant and
    costs no quality. The trade-off is that ffmpeg can only cut on keyframes, so
    a clip may start slightly before the requested point — which is harmless
    here, and preferable to making the user wait for a transcode.
    """

    clips: list[dict[str, Any]] = []
    output_dir = video_directory(video_id) / CLIP_CACHE_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    for item in compression.selected[:limit]:
        span = adaptive_clip_span(
            item,
            query=compression.query,
            query_intent=compression.query_intent,
            video_duration_seconds=duration_seconds,
        )
        name = f"{item.id}-{span.start_seconds:.2f}-{span.end_seconds:.2f}.mp4".replace("/", "_")
        output_path = output_dir / name
        if not output_path.exists() and not _cut(source_path, output_path, span):
            continue

        clips.append(
            {
                "candidate_id": item.id,
                "modality": item.modality.value,
                "start_seconds": span.start_seconds,
                "end_seconds": span.end_seconds,
                "timestamp_seconds": item.timestamp_seconds,
                "path": str(output_path),
                "reason": item.reason,
                "text": item.text,
            }
        )

    return clips


def _cut(source_path: Path, output_path: Path, span: Any) -> bool:
    duration = max(span.end_seconds - span.start_seconds, 0.5)
    completed = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            # -ss before -i seeks the container rather than decoding up to the
            # point, which is what keeps this fast on a long source.
            "-ss",
            f"{span.start_seconds:.3f}",
            "-i",
            str(source_path),
            "-t",
            f"{duration:.3f}",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0 and output_path.exists()
