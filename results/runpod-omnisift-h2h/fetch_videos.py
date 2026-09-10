"""Fetch the six Video-MME source videos onto a benchmark pod.

Named to match what the runners expect: /content/vids/videomme-<videoID>.mp4.
Capped at 360p H.264 because frames are resized for the encoder anyway, and
because a 24 GB card is the constraint, not source resolution.
"""

import json
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
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = json.loads(QFILE.read_text())

    sources: dict[str, str] = {}
    for row in rows:
        sources.setdefault(row["videoID"], row["url"])

    missing = []
    for video_id, url in sources.items():
        out = OUTDIR / f"videomme-{video_id}.mp4"
        if out.exists():
            print(f"have  {video_id}", flush=True)
            continue

        subprocess.run(
            ["yt-dlp", "--no-playlist", "-f", FORMAT,
             "--merge-output-format", "mp4", "-o", str(out), url],
            check=False,
        )
        if out.exists():
            print(f"ok    {video_id}  {out.stat().st_size / 1e6:.0f} MB", flush=True)
        else:
            missing.append(video_id)
            print(f"MISS  {video_id}  {url}", flush=True)

    print(f"\n{len(sources) - len(missing)}/{len(sources)} videos present in {OUTDIR}")
    if missing:
        # Not fatal: the runner skips what it cannot find. But the comparison is
        # only valid on questions every condition actually answered, so this has
        # to be visible rather than silent.
        print("MISSING (questions for these will be skipped):", ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
