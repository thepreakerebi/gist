"""Fetch Video-MME source videos onto a benchmark pod.

Shared by every runner under ``results/``: the efficiency benchmark, the OmniZip
head to head and the OmniScope head to head all expect the same layout,
``<outdir>/videomme-<videoID>.mp4``.

Capped at 360p H.264 because frames are resized for the encoder anyway, and
because a 24 GB card is the constraint, not source resolution. At 360p a long
Video-MME video is roughly 50 to 100 MB, so the full 18 video pool is a couple of
gigabytes rather than the 756 MB per video the originals cost.

**Report what is missing before the run, not after.** A video that fails to
download silently removes its questions from the denominator, and an evaluation
whose n quietly changed between conditions is not an evaluation. This exits
non-zero and names the failures rather than letting a runner discover them one
``SKIP`` line at a time.

Usage:
    python fetch_videos.py /root/videomme_av6.json /content/vids       # 6 videos
    python fetch_videos.py /root/videomme_av_all.json /content/vids    # 18 videos
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

QFILE = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/videomme_av6.json")
OUTDIR = Path(sys.argv[2] if len(sys.argv) > 2 else "/content/vids")

FORMAT = (
    "bv*[height<=360][vcodec^=avc1]+ba[acodec^=mp4a]/"
    "bv*[height<=360]+ba/b[height<=360]/best"
)


def main() -> int:
    ytdlp = shutil.which("yt-dlp")
    if ytdlp is None:
        print("yt-dlp is not on PATH. Install it with `pip install -U yt-dlp`.")
        return 2
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = json.loads(QFILE.read_text())

    sources: dict[str, str] = {}
    questions: dict[str, int] = {}
    for row in rows:
        sources.setdefault(row["videoID"], row["url"])
        questions[row["videoID"]] = questions.get(row["videoID"], 0) + 1

    print(f"{len(rows)} questions across {len(sources)} videos from {QFILE.name}")

    failed: list[str] = []
    for i, (vid, url) in enumerate(sources.items(), start=1):
        dest = OUTDIR / f"videomme-{vid}.mp4"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[{i}/{len(sources)}] {vid}: cached")
            continue
        print(f"[{i}/{len(sources)}] {vid}: fetching {url}", flush=True)
        try:
            result = subprocess.run(
                [ytdlp, "-f", FORMAT, "--merge-output-format", "mp4",
                 "-o", str(dest), "--no-playlist", "--quiet", "--no-warnings", url],
                capture_output=True, text=True,
            )
        except OSError as exc:
            # which() found it but it will not execute - a stale shim, or a
            # shebang pointing at an interpreter that no longer exists.
            print(f"yt-dlp at {ytdlp} will not run: {exc}")
            print("Reinstall it with `pip install -U yt-dlp` and try again.")
            return 2
        if result.returncode != 0 or not dest.exists():
            failed.append(vid)
            print(f"    FAILED: {result.stderr.strip()[:200]}", flush=True)

    got = len(sources) - len(failed)
    lost = sum(questions[v] for v in failed)
    print(f"\n{got}/{len(sources)} videos available")
    if failed:
        print(f"MISSING {len(failed)} videos, which removes {lost} of {len(rows)} questions:")
        for vid in failed:
            print(f"  {vid}  ({questions[vid]} questions)  {sources[vid]}")
        print(
            "\nEffective n for this run is "
            f"{len(rows) - lost}, not {len(rows)}. Record that number with the results."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
