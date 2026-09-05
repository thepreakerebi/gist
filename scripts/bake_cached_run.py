"""Bake a known-good offline snapshot of the library demo.

Why this exists: a live demo depends on a reachable API, a working network, and
model weights loading on time. On a stage, in front of a panel, any one of those
failing means there is nothing to show. This captures real runs — the actual
payloads the API streams — into ``web/public/cached-runs/`` so the same UI can
replay them from static files when the API cannot be reached.

It is a *snapshot of real output*, never a fabrication: every payload here was
produced by the real pipeline against the real video. The UI labels a replayed
run as cached rather than passing it off as live, because a demo that quietly
lies about what it just computed is worse than one that visibly falls back.

Usage (API must be running, videos already ingested):

    uv run python scripts/bake_cached_run.py \\
        --api http://127.0.0.1:8000 \\
        --query "What does he say is cool about the elephants?" \\
        --query "Where is he standing?"

With no --video, every ready video in the library is captured; --query may be
repeated and is run against each video.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

OUTPUT_ROOT = Path("web/public/cached-runs")
CLIP_DIRNAME = "clips"


def _get(api: str, path: str, timeout: float = 30.0) -> Any:
    with urllib.request.urlopen(f"{api}{path}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _stream_query(api: str, video_id: str, query: str, timeout: float) -> dict[str, Any]:
    """Run one query and collect the events the UI would have received."""

    request = urllib.request.Request(
        f"{api}/v1/library/videos/{video_id}/query",
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    captured: dict[str, Any] = {
        "video_id": video_id,
        "query": query,
        "scored": [],
        "selected": [],
        "metrics": None,
        "clips": [],
        "answer": None,
        "answer_provider": None,
    }

    event: str | None = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("event:"):
                event = line[6:].strip()
                continue
            if not line.startswith("data:"):
                continue
            try:
                payload = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue

            if event == "scored":
                captured["scored"] = payload.get("candidates", [])
            elif event == "selected":
                captured["selected"] = payload.get("selected", [])
                captured["metrics"] = payload.get("metrics")
            elif event == "clips":
                captured["clips"] = payload.get("clips", [])
            elif event == "done":
                captured["answer"] = payload.get("answer")
                captured["answer_provider"] = payload.get("answer_provider")
                captured["metrics"] = payload.get("metrics") or captured["metrics"]
                captured["clips"] = payload.get("clips") or captured["clips"]
            elif event == "error":
                raise RuntimeError(payload.get("message", "query failed"))

    return captured


def _copy_clips(api: str, run: dict[str, Any], clip_root: Path) -> None:
    """Pull each evidence clip into public/ and repoint the run at the copy.

    Without this the cached run would still reference API-served clip URLs, and
    the offline fallback would show an answer with broken video beside it —
    which is precisely the failure it exists to prevent.
    """

    clip_root.mkdir(parents=True, exist_ok=True)
    for clip in run["clips"]:
        url = clip.get("url")
        if not url:
            continue
        name = url.rsplit("/", 1)[-1]
        destination = clip_root / name
        try:
            with urllib.request.urlopen(f"{api}{url}", timeout=60) as response:
                destination.write_bytes(response.read())
        except (urllib.error.URLError, OSError) as exc:
            print(f"    ! could not fetch clip {name}: {exc}", file=sys.stderr)
            continue
        clip["url"] = f"/cached-runs/{CLIP_DIRNAME}/{name}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--video",
        action="append",
        dest="videos",
        help="Video id to capture. Repeatable. Defaults to every ready video.",
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        required=True,
        help="Question to run against each video. Repeatable.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)

    api = args.api.rstrip("/")

    try:
        library = _get(api, "/v1/library/videos")["videos"]
    except (urllib.error.URLError, OSError) as exc:
        print(f"could not reach the API at {api}: {exc}", file=sys.stderr)
        return 1

    ready = [video for video in library if video["status"] == "ready"]
    if args.videos:
        wanted = set(args.videos)
        ready = [video for video in ready if video["id"] in wanted]
    if not ready:
        print("no ready videos to capture", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    clip_root = args.output / CLIP_DIRNAME
    shutil.rmtree(clip_root, ignore_errors=True)

    details: dict[str, Any] = {}
    runs: list[dict[str, Any]] = []

    for video in ready:
        print(f"- {video['title']} ({video['id']})")
        details[video["id"]] = _get(api, f"/v1/library/videos/{video['id']}")

        for query in args.queries:
            print(f"    query: {query}")
            try:
                run = _stream_query(api, video["id"], query, args.timeout)
            except Exception as exc:  # noqa: BLE001 - reported, then skipped
                print(f"    ! failed: {exc}", file=sys.stderr)
                continue
            _copy_clips(api, run, clip_root)
            runs.append(run)
            kept = len(run["selected"])
            print(f"      kept {kept} of {len(run['scored'])}, {len(run['clips'])} clip(s)")

    if not runs:
        print("no runs captured", file=sys.stderr)
        return 1

    manifest = {
        "videos": ready,
        "details": details,
        "runs": runs,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Old single-run artifacts would otherwise linger and confuse the fallback.
    for stale in args.output.glob("*.json"):
        if stale.name != "manifest.json":
            stale.unlink()

    print(f"\nwrote {args.output / 'manifest.json'}: {len(ready)} video(s), {len(runs)} run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
