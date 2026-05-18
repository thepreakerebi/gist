#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download


DEFAULT_REPO = "lmms-lab/Video-MME"
DEFAULT_PARQUET = "videomme/test-00000-of-00001.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a small real Video-MME subset for local Gist benchmark runs."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/videomme-subset"))
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--duration", default="short")
    parser.add_argument("--video-count", type=int, default=2)
    parser.add_argument("--questions-per-video", type=int, default=3)
    parser.add_argument("--yt-dlp-bin", default="yt-dlp")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_root = args.output_dir / "videos"
    video_root.mkdir(parents=True, exist_ok=True)

    parquet_path = hf_hub_download(
        repo_id=args.repo,
        repo_type="dataset",
        filename=args.parquet,
        local_dir=args.output_dir / "hf",
    )
    frame = pd.read_parquet(parquet_path)
    if args.duration and "duration" in frame.columns:
        frame = frame[frame["duration"].astype(str).str.lower() == args.duration.lower()]

    prepared: list[dict[str, Any]] = []
    grouped = frame.groupby("video_id", sort=True)
    for video_id, rows in grouped:
        if len(prepared) >= args.video_count * args.questions_per_video:
            break
        if len(prepared) // args.questions_per_video >= args.video_count:
            break

        row = rows.iloc[0]
        url = str(row["url"])
        video_path = video_root / f"{video_id}.mp4"
        if args.force and video_path.exists():
            video_path.unlink()
        if not video_path.exists() and not _download_video(
            url=url,
            output_path=video_path,
            yt_dlp_bin=args.yt_dlp_bin,
        ):
            continue

        for _, question_row in rows.head(args.questions_per_video).iterrows():
            prepared.append(_benchmark_record(question_row, video_path, url))

    dataset_path = args.output_dir / "videomme-subset.jsonl"
    dataset_path.write_text("\n".join(json.dumps(item) for item in prepared) + "\n")
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "repo": args.repo,
                "parquet": args.parquet,
                "duration": args.duration,
                "video_count": args.video_count,
                "questions_per_video": args.questions_per_video,
                "examples": len(prepared),
                "dataset": str(dataset_path),
                "video_root": str(video_root),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"dataset={dataset_path}")
    print(f"video_root={video_root}")
    print(f"manifest={manifest_path}")
    print(f"examples={len(prepared)}")
    return 0 if prepared else 1


def _download_video(url: str, output_path: Path, yt_dlp_bin: str) -> bool:
    temp_template = output_path.with_suffix(".%(ext)s")
    command = [
        yt_dlp_bin,
        "--no-playlist",
        "-f",
        "bv*[height<=360]+ba/b[height<=360]/best[height<=360]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        str(temp_template),
        url,
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        print(f"skip {url}: {completed.stderr.strip()}")
        return False

    candidates = sorted(output_path.parent.glob(output_path.stem + ".*"))
    for candidate in candidates:
        if candidate.suffix.lower() == ".mp4":
            candidate.rename(output_path)
            return True
    return output_path.exists()


def _benchmark_record(row: pd.Series, video_path: Path, url: str) -> dict[str, Any]:
    options = row.get("options", [])
    if isinstance(options, str):
        options = [line.strip() for line in options.splitlines() if line.strip()]
    return {
        "id": str(row.get("question_id")),
        "video_id": str(row.get("video_id")),
        "query": str(row.get("question")),
        "duration_seconds": 1.0,
        "video_path": str(video_path),
        "source_url": url,
        "choices": [str(option) for option in options],
        "answer": str(row.get("answer")),
    }


if __name__ == "__main__":
    raise SystemExit(main())
