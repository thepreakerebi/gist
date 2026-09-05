"""Query-independent ingestion: everything that can be done before a question exists.

This is the half of the pipeline that does not depend on what anyone will ask —
downloading, sampling frames, windowing audio, transcribing, OCR, and running
both contrastive encoders. It is also the slow half: Whisper alone takes minutes
on an hour-long video, and it produces exactly the same transcript regardless of
the query.

Running it once and persisting the result is what makes the library's flow
possible. Paste a link, wait once; every later question is a vector comparison
against stored embeddings rather than a re-encode. The query-dependent half
lives in :mod:`gist.library.query`.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from gist.audio.whisper import TranscriptQuality, resolve_whisper_settings
from gist.db import repository as repo
from gist.library import media
from gist.media.ingestion import MediaIngestor
from gist.media.longform import ProcessingMode
from gist.media.models import IngestedVideo

ProgressCallback = Callable[[str, float], None]

# Where retained sources and their derived assets live. Only the path is stored
# in Postgres; see gist/db/schema.sql for why the media itself is not.
LIBRARY_ROOT = Path(".gist/library")

# Frames sampled per video. 128 over an hour is one frame every ~28s, which is
# what the long-video suite uses; the selector's job is to find the few that
# matter, so a denser sample mostly buys encode time.
SAMPLE_COUNT = 128
AUDIO_WINDOW_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class IngestionResult:
    video_id: str
    frame_count: int
    audio_window_count: int
    duration_seconds: float


def video_directory(video_id: str) -> Path:
    return LIBRARY_ROOT / video_id


def ingest_video(
    video_id: str,
    youtube_url: str,
    *,
    progress: ProgressCallback | None = None,
) -> IngestionResult:
    """Download, encode and persist a video so it can be queried later.

    Progress is reported to both the callback (for live SSE) and the database
    (so a reload mid-ingestion still shows the right state). Any failure marks
    the row failed with a readable message rather than leaving it stuck in
    ``ingesting`` forever.
    """

    def report(message: str, fraction: float) -> None:
        repo.update_video_status(video_id, status_detail=message, progress=fraction)
        if progress is not None:
            progress(message, fraction)

    try:
        repo.update_video_status(video_id, status="ingesting", progress=0.0)

        target = video_directory(video_id)
        report("fetching video metadata", 0.02)
        info = media.probe_remote(youtube_url)
        repo.update_video_metadata(
            video_id,
            title=info.title,
            duration_seconds=info.duration_seconds,
            thumbnail_url=info.thumbnail_url,
        )

        report("downloading video", 0.05)
        source_path = media.download(youtube_url, target)
        repo.update_video_metadata(video_id, source_path=str(source_path))

        report("sampling frames and audio", 0.25)
        # Derived assets (frames, audio windows) land beside the source inside
        # the video's own library directory, so deleting the video removes them.
        ingested = MediaIngestor(output_root=target / "assets").ingest(
            video_path=source_path,
            sample_count=SAMPLE_COUNT,
            audio_window_seconds=AUDIO_WINDOW_SECONDS,
            processing_mode=ProcessingMode.AUTO,
        )

        report("transcribing speech", 0.40)
        transcripts = _transcribe(ingested)

        report("reading on-screen text", 0.60)
        ocr_text = _extract_ocr(ingested)

        report("encoding frames", 0.70)
        frame_vectors = _embed_frames(ingested)

        report("encoding audio", 0.85)
        audio_vectors = _embed_audio(ingested)

        report("saving to library", 0.95)
        _persist(
            video_id,
            ingested,
            transcripts=transcripts,
            ocr_text=ocr_text,
            frame_vectors=frame_vectors,
            audio_vectors=audio_vectors,
        )

        full_transcript = " ".join(
            text.strip() for _, text in sorted(transcripts.items()) if text.strip()
        )
        repo.mark_video_ready(
            video_id,
            duration_seconds=ingested.metadata.duration_seconds,
            frame_count=len(ingested.frames),
            audio_window_count=len(ingested.audio_windows),
            transcript=full_transcript or None,
            source_path=str(source_path),
        )
        if progress is not None:
            progress("ready", 1.0)

        return IngestionResult(
            video_id=video_id,
            frame_count=len(ingested.frames),
            audio_window_count=len(ingested.audio_windows),
            duration_seconds=ingested.metadata.duration_seconds,
        )

    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        repo.update_video_status(
            video_id,
            status="failed",
            status_detail=None,
            error=str(exc) or exc.__class__.__name__,
        )
        if progress is not None:
            progress(f"failed: {exc}", 1.0)
        raise


def delete_video_assets(video_id: str) -> None:
    """Remove a video's on-disk assets. The DB rows cascade separately."""

    shutil.rmtree(video_directory(video_id), ignore_errors=True)


# ------------------------------------------------------------- internals ----


def _transcribe(ingested: IngestedVideo) -> dict[Path, str]:
    if not ingested.audio_windows:
        return {}
    try:
        from gist.audio.whisper import FasterWhisperTranscriber

        settings = resolve_whisper_settings(quality=TranscriptQuality.BALANCED)
        transcriber = FasterWhisperTranscriber(
            model_size=settings.model_size,
            device=settings.device,
            compute_type=settings.compute_type,
            beam_size=settings.beam_size,
        )
        return transcriber.transcribe_windows(ingested.audio_windows)
    except Exception:  # noqa: BLE001
        # Speech is one signal of several. A video with no usable audio should
        # still land in the library and be answerable from frames and OCR.
        return {}


# Tesseract run over a frame that contains no text does not return nothing — it
# returns confident-looking noise ('| _"! el ae Ey " Se (a* 268 Coal bss'). Shown
# as evidence in the UI that reads as a broken product, and as candidate text it
# pollutes the lexical fallback. These gates keep real captions and slide titles
# while discarding hallucinated glyphs.
_OCR_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_OCR_ALLOWED_PUNCTUATION = frozenset(".,:;'\"-!?()&/%$#@")
_OCR_MIN_WORDS = 2
_OCR_MIN_CLEAN_RATIO = 0.8
# Measured separation on real frames: Tesseract noise fragments a photographic
# frame into many one- and two-letter tokens (long-token share 0.43-0.53, mean
# length 2.1-2.8), while genuine slide text is made of whole words (share 1.0,
# mean length 5.2-11.0). Alphabetic-ness alone does not discriminate — the noise
# is letters too — but token *shape* does, and it needs no dictionary, which
# keeps this working on a bare deployment host.
_OCR_MIN_LONG_TOKEN_SHARE = 0.7
_OCR_MIN_MEAN_TOKEN_LENGTH = 3.5

# Real titles legitimately contain short function words ("Legged Locomotion *in*
# Nature"), so these are not counted against the short-token share. Without the
# exemption the gate rejected genuine slide titles.
_OCR_FUNCTION_WORDS = frozenset(
    "a an and as at be by do for from he i in is it of on or so the to we with".split()
)


def _plausible_ocr_text(raw: str) -> str | None:
    """Return the text if it looks like real writing, else None."""

    text = " ".join(raw.split())
    if len(text) < 4:
        return None

    words = _OCR_WORD.findall(text)
    if len(words) < _OCR_MIN_WORDS:
        return None

    # Noise is dominated by stray glyphs; real text is mostly letters and digits.
    clean = sum(
        1
        for character in text
        if character.isalnum() or character.isspace() or character in _OCR_ALLOWED_PUNCTUATION
    )
    if clean / len(text) < _OCR_MIN_CLEAN_RATIO:
        return None

    long_share = sum(
        1 for word in words if len(word) >= 3 or word.lower() in _OCR_FUNCTION_WORDS
    ) / len(words)
    mean_length = sum(len(word) for word in words) / len(words)
    if long_share < _OCR_MIN_LONG_TOKEN_SHARE:
        return None
    if mean_length < _OCR_MIN_MEAN_TOKEN_LENGTH:
        return None

    return text


def _extract_ocr(ingested: IngestedVideo) -> dict[Path, str]:
    if not ingested.frames:
        return {}
    try:
        from gist.vision.ocr import TesseractFrameOcr

        raw = TesseractFrameOcr().extract_text(ingested.frames)
    except Exception:  # noqa: BLE001 - tesseract is optional
        return {}

    filtered: dict[Path, str] = {}
    for path, text in raw.items():
        plausible = _plausible_ocr_text(text or "")
        if plausible:
            filtered[path] = plausible
    return filtered


def _embed_frames(ingested: IngestedVideo) -> dict[Path, list[float]]:
    if not ingested.frames:
        return {}
    from gist.vision.clip import HuggingFaceClipFrameScorer

    scorer = HuggingFaceClipFrameScorer()
    return {
        frame.path: list(embedding.vector)
        for frame, embedding in zip(
            ingested.frames, scorer.embed_frames(ingested.frames), strict=True
        )
    }


def _embed_audio(ingested: IngestedVideo) -> dict[Path, list[float]]:
    if not ingested.audio_windows:
        return {}
    try:
        from gist.audio.clap import HuggingFaceClapAudioScorer

        return HuggingFaceClapAudioScorer().embed_windows(ingested.audio_windows)
    except Exception:  # noqa: BLE001
        # Without CLAP the audio candidates still carry their transcript text,
        # and the selector falls back to lexical relevance over it.
        return {}


def _persist(
    video_id: str,
    ingested: IngestedVideo,
    *,
    transcripts: dict[Path, str],
    ocr_text: dict[Path, str],
    frame_vectors: dict[Path, list[float]],
    audio_vectors: dict[Path, list[float]],
) -> None:
    repo.replace_frames(
        video_id,
        [
            {
                "frame_index": frame.index,
                "timestamp_seconds": frame.timestamp_seconds,
                "asset_path": str(frame.path),
                "ocr_text": (ocr_text.get(frame.path) or "").strip() or None,
                "scene_start_seconds": None,
                "scene_end_seconds": None,
                "embedding": repo._as_vector(frame_vectors.get(frame.path)),
            }
            for frame in ingested.frames
        ],
    )
    repo.replace_audio_windows(
        video_id,
        [
            {
                "window_index": window.index,
                "start_seconds": window.start_seconds,
                "end_seconds": window.start_seconds + window.duration_seconds,
                "transcript_text": (transcripts.get(window.path) or "").strip() or None,
                "asset_path": str(window.path),
                "embedding": repo._as_vector(audio_vectors.get(window.path)),
            }
            for window in ingested.audio_windows
        ],
    )
