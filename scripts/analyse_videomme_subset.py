#!/usr/bin/env python3
"""Describe the Video-MME long split, and state exactly how our subset was drawn.

Why this exists
---------------
The project reports results on 51 questions over 18 Video-MME videos. Until this
script, nothing in the repository said how those 18 were chosen, and the prose in
`docs/datasets.md` claimed they were "filtered to questions requiring audio and
video". **That claim was wrong**, and this script is what established it.

What the selection actually was, reconstructed from the data:

  * the long split is `video_id` 601-900, 300 videos, 900 questions
  * our subset is the contiguous block 867-887
  * three ids inside that block (874, 876, 878) are missing, and they exist in
    the benchmark, so they were lost to failed downloads rather than excluded
  * every question of every surviving video was kept, except video 867 (1 of 3)
    and 887 (2 of 3), which is why n is 51 and not 54

So the sample is **an arbitrary contiguous block truncated by download success**,
not a curated audio-visual subset. That is worth stating plainly rather than
dressing up: an arbitrary block is a defensible sample precisely because nothing
about the method influenced it, whereas a hand-picked one would invite the
question of how it was picked.

This script does not change the corpus. It describes it, checks that the pool
files on disk still match that description, and prints the distribution analysis
that a reader would otherwise have to take on trust.

Usage
-----
    uv run python scripts/analyse_videomme_subset.py
    uv run python scripts/analyse_videomme_subset.py --markdown reports/videomme-subset.md
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
BENCHMARK_DIR = Path(".gist/benchmark")
POOLS = (
    "videomme_av6.json",
    "videomme_av_all.json",
    "videomme_long.json",
    "videomme_balanced.json",
)
DEAD_IDS = Path("data/eval/videomme-unavailable.txt")

# A question is counted as carrying an audio marker if its text names speech or
# sound. This is a crude keyword test and is used only to characterise the
# sample, never to select it. It will miss questions that need audio without
# saying so, which is exactly why it was not used as a filter.
AUDIO_MARKERS = re.compile(
    r"\b(?:say|says|said|saying|mention|mentions|mentioned|speaker|narrator|"
    r"tell|tells|told|explain|explains|hear|heard|sound|sounds|music|voice|"
    r"speak|speaks|spoken|talk|talks|announce|announces|according to)\b",
    re.IGNORECASE,
)


def load_long_split(parquet: Path) -> pd.DataFrame:
    if not parquet.exists():
        raise SystemExit(
            f"Video-MME parquet not found at {parquet}.\n"
            "Fetch it with scripts/prepare_videomme_subset.py first."
        )
    frame = pd.read_parquet(parquet)
    return frame[frame["duration"] == "long"].copy()


def load_pool(name: str) -> list[dict[str, Any]] | None:
    path = BENCHMARK_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def describe_population(long: pd.DataFrame) -> list[str]:
    ids = long["video_id"].astype(int)
    lines = [
        "## The population we drew from",
        "",
        f"Video-MME's **long** split: **{ids.nunique()} videos**, "
        f"**{len(long)} questions**, `video_id` {ids.min()}-{ids.max()}.",
        "",
        "Video-MME defines long as 30 to 60 minutes, so every video in this split "
        "clears the project's 30 minute floor by construction. It also means the "
        "split cannot supply an hour-plus recording at all, which is why the "
        "curated corpus exists alongside it.",
        "",
        "### Questions by task type",
        "",
        "| Task type | Questions | Share |",
        "| :--- | ---: | ---: |",
    ]
    counts = long["task_type"].value_counts()
    for task, n in counts.items():
        lines.append(f"| {task} | {n} | {n / len(long):.1%} |")
    lines += [
        "",
        f"Across **{long['domain'].nunique()} domains** and "
        f"**{long['sub_category'].nunique()} sub-categories**.",
        "",
    ]
    return lines


def describe_selection(long: pd.DataFrame, pool: list[dict[str, Any]]) -> list[str]:
    chosen = sorted({int(row["video_id"]) for row in pool})
    block = list(range(chosen[0], chosen[-1] + 1))
    gaps = [v for v in block if v not in chosen]
    present = set(long["video_id"].astype(str))
    gaps_exist = [g for g in gaps if str(g) in present]

    kept = Counter(row["video_id"] for row in pool)
    available = long[long["video_id"].isin({str(c) for c in chosen})]
    avail = Counter(available["video_id"])
    truncated = {v: (avail[v], kept[v]) for v in avail if avail[v] != kept[v]}

    lines = [
        "## How our subset was actually drawn",
        "",
        f"**{len(chosen)} videos, {len(pool)} questions.**",
        "",
        f"- Contiguous block `video_id` **{chosen[0]}-{chosen[-1]}** "
        f"({len(block)} ids).",
        f"- Missing from inside the block: **{gaps or 'none'}**.",
    ]
    if gaps_exist:
        lines.append(
            "- Those ids **do exist** in the benchmark, so they were lost to failed "
            "downloads, not excluded by any criterion."
        )
    if truncated:
        detail = ", ".join(f"{v} ({k} of {a})" for v, (a, k) in sorted(truncated.items()))
        lines.append(f"- Videos where fewer questions were kept than exist: {detail}.")
    lines += [
        "",
        "**There was no audio-visual filtering.** The selection is a block of the "
        "long split truncated by what downloaded. State it that way.",
        "",
    ]
    return lines


def describe_audio_balance(long: pd.DataFrame, pool: list[dict[str, Any]]) -> list[str]:
    ids = {row["question_id"] for row in pool}
    chosen = long[long["question_id"].isin(ids)]
    pop_rate = long["question"].str.contains(AUDIO_MARKERS).mean()
    sel_rate = chosen["question"].str.contains(AUDIO_MARKERS).mean()

    lines = [
        "## Where the sample sits on audio",
        "",
        "Counting questions whose text names speech or sound. This is a crude "
        "keyword test, used to characterise the sample and never to select it.",
        "",
        "| | Audio-marker rate |",
        "| :--- | ---: |",
        f"| All {len(long)} long questions | {pop_rate:.0%} |",
        f"| Our {len(chosen)} questions | {sel_rate:.0%} |",
        "",
    ]
    if sel_rate < pop_rate:
        lines += [
            "**The sample is audio-poorer than the population it came from.** That "
            "cuts against the method rather than for it: fewer questions where "
            "sound carries the answer means fewer chances for cross-modal "
            "arbitration to show its value. A conservative bias is a safe one to "
            "report, and it is the honest reading of these two numbers.",
            "",
        ]
    return lines


def load_dead() -> set[str]:
    if not DEAD_IDS.exists():
        return set()
    return {
        line.split("#")[0].strip()
        for line in DEAD_IDS.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


def describe_pools(long: pd.DataFrame) -> list[str]:
    dead = load_dead()
    lines = [
        "## The pool files, and whether they still match",
        "",
        "A missing video is either **dead** — removed or privatised on YouTube, so it "
        "can never be fetched — or merely **unfetched**. The two need different "
        "responses: a dead video caps the pool's n permanently, an unfetched one is a "
        "download away.",
        "",
        "| Pool | Questions | Videos | Downloaded | Dead | Unfetched |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    video_dir = Path(".gist/videos/archive")
    on_disk = {p.stem.removeprefix("videomme-") for p in video_dir.glob("videomme-*.mp4")}

    for name in POOLS:
        pool = load_pool(name)
        if pool is None:
            lines.append(f"| `{name}` | — | — | not present | — | — |")
            continue
        vids = {row["videoID"] for row in pool}
        have = vids & on_disk
        missing = vids - on_disk
        gone = missing & dead
        pending = missing - dead
        lost_q = sum(1 for row in pool if row["videoID"] in gone)
        dead_cell = f"**{len(gone)}** (−{lost_q}q)" if gone else "0"
        lines.append(
            f"| `{name}` | {len(pool)} | {len(vids)} | {len(have)}/{len(vids)} | "
            f"{dead_cell} | {len(pending)} |"
        )

    lines += [
        "",
        "`fetch_videos.py` exits non-zero and names what is missing, so a pod run "
        "stops at the fetch step rather than quietly changing n between conditions. "
        "Dead ids live in `data/eval/videomme-unavailable.txt` and are excluded at "
        "selection time, so a rebuilt pool picks replacements instead.",
        "",
    ]
    return lines


def describe_scaling(long: pd.DataFrame, pool: list[dict[str, Any]]) -> list[str]:
    per_video = len(long) / long["video_id"].nunique()
    return [
        "## What scaling up would cost",
        "",
        f"The long split averages **{per_video:.1f} questions per video**, so n "
        "grows by downloading more of the 300, not by mining the 18 already held.",
        "",
        "| Videos | Approximate questions |",
        "| ---: | ---: |",
        f"| {len({r['video_id'] for r in pool})} (now) | {len(pool)} |",
        f"| 50 | ~{round(50 * per_video)} |",
        f"| 100 | ~{round(100 * per_video)} |",
        f"| 300 (all) | {len(long)} |",
        "",
        "Worth knowing the number before deciding it is worth paying. More n "
        "tightens the interval around a result that is already parity; it does not "
        "change a claim.",
        "",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=PARQUET)
    parser.add_argument("--pool", default="videomme_av_all.json")
    parser.add_argument("--markdown", type=Path, help="write the report here as well")
    args = parser.parse_args()

    long = load_long_split(args.parquet)
    pool = load_pool(args.pool)
    if pool is None:
        raise SystemExit(f"pool not found: {BENCHMARK_DIR / args.pool}")

    report = ["# Video-MME: the population, and how our subset was drawn", ""]
    report += describe_population(long)
    report += describe_selection(long, pool)
    report += describe_audio_balance(long, pool)
    report += describe_pools(long)
    report += describe_scaling(long, pool)

    text = "\n".join(report)
    print(text)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text + "\n")
        print(f"\nwritten to {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
