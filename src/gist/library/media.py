"""YouTube acquisition for the video library.

Split out of ``gist.api.demo`` so ingestion and the legacy one-shot demo share
one implementation of "fetch this URL and tell me about it".
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_YOUTUBE_ID = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/|shorts/|live/))([A-Za-z0-9_-]{11})"
)


@dataclass(frozen=True, slots=True)
class RemoteVideoInfo:
    title: str
    duration_seconds: float | None
    thumbnail_url: str | None
    youtube_id: str | None


def youtube_id(url: str) -> str | None:
    match = _YOUTUBE_ID.search(url)
    return match.group(1) if match else None


def probe_remote(url: str, *, timeout: int = 60) -> RemoteVideoInfo:
    """Read a remote video's metadata without downloading it.

    Used before ingestion starts so the library can show a real title and
    thumbnail while the download runs, instead of a bare URL.
    """

    try:
        completed = subprocess.run(
            ["yt-dlp", "--no-warnings", "--skip-download", "--dump-single-json", "--no-playlist", url],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("yt-dlp is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("timed out reading video metadata") from exc

    if completed.returncode != 0:
        raise RuntimeError(_clean_ytdlp_error(completed.stderr))

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("could not parse video metadata") from exc

    duration = payload.get("duration")
    return RemoteVideoInfo(
        title=payload.get("title") or url,
        duration_seconds=float(duration) if duration else None,
        thumbnail_url=payload.get("thumbnail"),
        youtube_id=payload.get("id") or youtube_id(url),
    )


def download(url: str, output_dir: Path, *, max_height: int = 360) -> Path:
    """Download a video at demo resolution.

    Capped at 360p deliberately: frames are resized to 224x224 for CLIP anyway,
    so a higher-resolution source costs download time and disk for no gain in
    scoring quality. It is still enough to look right in the clip player.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "video.mp4"
    template = output_path.with_suffix(".%(ext)s")
    completed = subprocess.run(
        [
            "yt-dlp",
            "--no-playlist",
            "-f",
            f"bv*[height<={max_height}]+ba/b[height<={max_height}]/best[height<={max_height}]/best",
            "--merge-output-format",
            "mp4",
            "-o",
            str(template),
            url,
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(_clean_ytdlp_error(completed.stderr))

    for candidate in sorted(output_dir.glob("video.*")):
        if candidate.suffix.lower() == ".mp4":
            return candidate
    matches = sorted(output_dir.glob("video.*"))
    if matches:
        return matches[0]
    raise RuntimeError("video download produced no file")


def _clean_ytdlp_error(stderr: str) -> str:
    """Turn yt-dlp's stderr into something worth showing a user."""

    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    for line in lines:
        if line.startswith("ERROR:"):
            return line.removeprefix("ERROR:").strip()
    return lines[-1] if lines else "video download failed"
