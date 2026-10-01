#!/usr/bin/env python3
"""Build a modality-balanced pool from Video-MME's long split.

Why
---
The existing pool (`videomme_av_all.json`) is a contiguous block of the long
split truncated by download success — see `scripts/analyse_videomme_subset.py`.
It was never filtered for modality and it came out lopsided: 2% of its questions
name speech or sound against 15% of the split, and all 18 of its videos sit in a
single domain, Life Record.

For a method whose argument is arbitrating a budget *between* two modalities,
that is the wrong instrument twice over. Questions where sound never matters give
the audio pathway nothing to win, and one domain cannot show whether the result
generalises.

Selecting only audio questions would just mirror the fault: a question answerable
from the transcript alone exercises Whisper, not the arbitration. So this builds
a pool balanced **across** modality phrasing rather than tilted toward either.

Selection, stated so it can be argued with
------------------------------------------
1. Video-MME's **long** band only (30-60 min, `video_id` 601-900).
2. Each question is bucketed by what its text names:
   - `both`   names speech or sound **and** something seen — the questions that
              actually need arbitration, and the scarcest at 21 in the split
   - `audio`  names speech or sound only
   - `visual` names something seen only
   - unmarked questions are not eligible
3. Videos are taken in order of how many eligible questions they carry, because
   a video costs one download whether it yields one question or three.
4. A video is skipped if it would push one bucket more than `--tolerance` ahead
   of the other, or push one domain past `--max-domain-share` of the pool.
5. Every `both` question from a selected video is always kept.

The keyword test is a heuristic and is wrong in both directions: it misses
questions needing audio that do not say so, and it catches idioms like "tell
apart". **Every selected question is printed for review** and the pool is not
meant to be used unread. A model deciding what needs audio would put a model's
judgement inside the sampling frame of the evaluation it later takes part in.

This writes a *new* pool and does not touch the existing one. The arbitrary block
stays as the unbiased sample; this is the targeted one. Reporting both is
stronger than replacing one with the other.

Usage
-----
    uv run python scripts/build_videomme_pool.py --review
    uv run python scripts/build_videomme_pool.py --write .gist/benchmark/videomme_balanced.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

PARQUET = Path("data/videomme-real-subset/hf/videomme/test-00000-of-00001.parquet")

AUDIO_MARKERS = re.compile(
    r"\b(?:say|says|said|saying|mention|mentions|mentioned|speaker|narrator|"
    r"tell|tells|told|explain|explains|hear|heard|sound|sounds|music|voice|"
    r"speak|speaks|spoken|talk|talks|announce|announces|according to|"
    r"interview|conversation|dialogue|lyric|lyrics|song|sing|sings)\b",
    re.IGNORECASE,
)

VISUAL_MARKERS = re.compile(
    r"\b(?:show|shows|shown|appear|appears|see|seen|visible|screen|wear|wearing|"
    r"colou?r|scene|display|displays|image|picture|frame|gesture|background|logo|"
    r"written|sign|chart|graph|slide|hold|holding|stand|standing)\b",
    re.IGNORECASE,
)


def bucket_of(row: pd.Series) -> str:
    if row["audio"] and row["visual"]:
        return "both"
    if row["audio"]:
        return "audio"
    if row["visual"]:
        return "visual"
    return "unmarked"


DEAD_IDS = Path("data/eval/videomme-unavailable.txt")


def load_dead(path: Path) -> set[str]:
    """YouTube ids that no longer resolve.

    Video-MME is sourced from YouTube and its videos rot: uploaders delete them
    or make them private. A dead id cannot be fetched on a pod either, so it must
    be excluded at selection time rather than discovered as a smaller n after the
    download.
    """
    if not path.exists():
        return set()
    return {
        line.split("#")[0].strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


def load_long(parquet: Path) -> pd.DataFrame:
    if not parquet.exists():
        raise SystemExit(f"parquet not found at {parquet}")
    frame = pd.read_parquet(parquet)
    long = frame[frame["duration"] == "long"].copy()
    long["audio"] = long["question"].str.contains(AUDIO_MARKERS)
    long["visual"] = long["question"].str.contains(VISUAL_MARKERS)
    long["bucket"] = long.apply(bucket_of, axis=1)
    dead = load_dead(DEAD_IDS)
    if dead:
        before = long["videoID"].nunique()
        long = long[~long["videoID"].isin(dead)]
        print(f"excluded {before - long['videoID'].nunique()} unavailable videos")
    return long


def select(
    long: pd.DataFrame,
    target: int,
    tolerance: int,
    max_domain_share: float,
) -> pd.DataFrame:
    eligible = long[long["bucket"] != "unmarked"]

    # One download yields every eligible question in that video, so prefer videos
    # carrying several, and prefer those carrying a `both` question at all.
    stats = (
        eligible.groupby("video_id")
        .agg(
            n=("bucket", "size"),
            both=("bucket", lambda s: int((s == "both").sum())),
            domain=("domain", "first"),
        )
        .sort_values(["both", "n"], ascending=False)
    )

    chosen: list[str] = []
    counts: Counter[str] = Counter()
    domains: Counter[str] = Counter()

    for video_id, row in stats.iterrows():
        if sum(counts.values()) >= target:
            break
        rows = eligible[eligible["video_id"] == video_id]
        adds = Counter(rows["bucket"])

        # Measured against the target, not against the pool so far: judging a
        # share when the pool holds three questions rejects everything.
        if (domains[row["domain"]] + len(rows)) / target > max_domain_share:
            continue

        audio_after = counts["audio"] + adds["audio"]
        visual_after = counts["visual"] + adds["visual"]
        if abs(audio_after - visual_after) > tolerance and sum(counts.values()) > 0:
            continue

        chosen.append(str(video_id))
        counts += adds
        domains[row["domain"]] += len(rows)

    return eligible[eligible["video_id"].isin(chosen)]


def benchmark_record(row: pd.Series) -> dict[str, Any]:
    """Match the shape the existing pools use, so every runner reads it as-is."""
    return {
        "video_id": str(row["video_id"]),
        "duration": str(row["duration"]),
        "domain": str(row["domain"]),
        "sub_category": str(row["sub_category"]),
        "url": str(row["url"]),
        "videoID": str(row["videoID"]),
        "question_id": str(row["question_id"]),
        "task_type": str(row["task_type"]),
        "question": str(row["question"]),
        "options": [str(option) for option in row["options"]],
        "answer": str(row["answer"]),
        "modality_bucket": str(row["bucket"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=PARQUET)
    parser.add_argument("--target", type=int, default=60, help="questions to aim for")
    parser.add_argument(
        "--tolerance",
        type=int,
        default=6,
        help="how far audio and visual counts may diverge",
    )
    parser.add_argument(
        "--max-domain-share",
        type=float,
        default=0.34,
        help="no domain may exceed this share of the pool",
    )
    parser.add_argument("--write", type=Path, help="write the pool JSON here")
    parser.add_argument("--review", action="store_true", help="print every selected question")
    args = parser.parse_args()

    long = load_long(args.parquet)
    selected = select(long, args.target, args.tolerance, args.max_domain_share)

    print("Video-MME long split")
    print(f"  {long['video_id'].nunique()} videos, {len(long)} questions")
    for name, n in long["bucket"].value_counts().items():
        print(f"    {name:9} {n:3}")
    print()
    print("Selected pool")
    print(f"  {selected['video_id'].nunique()} videos, {len(selected)} questions")
    for name, n in selected["bucket"].value_counts().items():
        print(f"    {name:9} {n:3}  ({n / len(selected):.0%})")
    print()
    print("  by domain:")
    for domain, n in selected["domain"].value_counts().items():
        print(f"    {domain:22} {n:3}  ({n / len(selected):.0%})")
    print()
    print("  by task type:")
    for task, n in selected["task_type"].value_counts().items():
        print(f"    {task:22} {n:3}")

    disk = {
        p.stem.removeprefix("videomme-")
        for p in Path(".gist/videos/archive").glob("videomme-*.mp4")
    }
    have = {v for v in selected["videoID"] if v in disk}
    print()
    print(f"  already downloaded: {len(have)} of {selected['videoID'].nunique()}")
    print(f"  still to fetch    : {selected['videoID'].nunique() - len(have)}")

    if args.review:
        print("\n--- every selected question, for review ---")
        for video_id, rows in selected.groupby("video_id"):
            head = rows.iloc[0]
            print(f"\nvideo {video_id}  ({head['domain']} / {head['sub_category']})")
            for _, row in rows.iterrows():
                print(f"  [{row['bucket']:6}] {row['task_type']:22} {row['question']}")

    if args.write:
        records = [benchmark_record(row) for _, row in selected.iterrows()]
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(records, indent=2) + "\n")
        print(f"\nwrote {len(records)} questions to {args.write}")
        print("Read them before running anything on them.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
