#!/usr/bin/env python3
"""Generate `notebooks/01-dataset-eda.ipynb`.

The notebook is a *rendering* of the selection scripts, not a reimplementation of
them. Every cell imports from `scripts/analyse_videomme_subset.py` and
`scripts/build_videomme_pool.py` and then draws what those functions return.

That rule is the whole point. A notebook holding its own copy of the bucketing
regexes or the selection rule would drift from the scripts the moment either
changed, and the drifted copy is the one nobody runs in CI. Here there is one
source of truth and two renderings of it: a markdown report for the repository,
and tables and charts for a reader.

Regenerate with:
    uv run python notebooks/build_dataset_eda.py
    uv run jupyter nbconvert --execute --inplace notebooks/01-dataset-eda.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "01-dataset-eda.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(True),
    }


CELLS = [
    md("""
# Gist — dataset EDA

Where the evaluation data comes from, what is in it, and how the sample was drawn.

**Nothing here is reimplemented.** Every cell imports from the two selection
scripts and renders what they return:

- `scripts/analyse_videomme_subset.py` — describes the population and reconstructs
  how the original subset was drawn
- `scripts/build_videomme_pool.py` — buckets questions by modality and builds the
  balanced pool

One source of truth, two renderings: a markdown report for the repository, and
this notebook for a reader. Regenerate with
`uv run jupyter nbconvert --execute --inplace notebooks/01-dataset-eda.ipynb`.
"""),
    code("""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
# The scripts resolve .gist/ and data/ relative to the working directory, so run
# from the repository root rather than from notebooks/.
os.chdir(REPO)
sys.path.insert(0, str(REPO / "scripts"))

from analyse_videomme_subset import load_long_split, load_pool          # noqa: E402
from build_videomme_pool import load_long, select                       # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
ACCENT, MUTED = "#c2410c", "#d4d4d8"
pd.set_option("display.max_colwidth", 90)
"""),
    md("""
## 1. The source

Video-MME ships as one table on HuggingFace, `lmms-lab/Video-MME`. Only the **long**
band is ever used: a two-minute clip cannot test compression on long video, so 600
of its 900 videos were never candidates.
"""),
    code("""
raw = pd.read_parquet(REPO / "data/videomme-real-subset/hf/videomme/test-00000-of-00001.parquet")
bands = (raw.groupby("duration")
            .agg(videos=("video_id", "nunique"), questions=("video_id", "size"))
            .reindex(["short", "medium", "long"]))
bands["duration range"] = ["11 s – 2 min", "4 – 15 min", "30 – 60 min"]
bands
"""),
    code("""
long = load_long_split(REPO / "data/videomme-real-subset/hf/videomme/test-00000-of-00001.parquet")
print(f"long split: {long.video_id.nunique()} videos, {len(long)} questions, "
      f"video_id {long.video_id.astype(int).min()}–{long.video_id.astype(int).max()}")
print(f"{long.domain.nunique()} domains, {long.sub_category.nunique()} sub-categories")
"""),
    md("## 2. What the long split is made of"),
    code("""
tasks = long.task_type.value_counts()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
tasks.sort_values().plot.barh(ax=ax1, color=MUTED)
ax1.set_title("Questions by task type", loc="left")
ax1.set_xlabel("questions")
long.domain.value_counts().sort_values().plot.barh(ax=ax2, color=MUTED)
ax2.set_title("Videos by domain", loc="left")
ax2.set_xlabel("questions")
plt.tight_layout(); plt.show()
tasks.to_frame("questions").assign(share=lambda d: (d.questions / len(long)).map("{:.1%}".format))
"""),
    md("""
## 3. How long are the videos, really?

Video-MME *defines* the long band as 30–60 minutes. Measured from the files on
disk rather than taken on trust.
"""),
    code("""
import subprocess

def duration_minutes(path: Path) -> float | None:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True).stdout.strip()
    return float(out) / 60 if out else None

files = sorted((REPO / ".gist/videos/archive").glob("videomme-*.mp4"))
durations = pd.Series(
    {f.stem.removeprefix("videomme-"): duration_minutes(f) for f in files}
).dropna()

fig, ax = plt.subplots(figsize=(6, 2.8))
ax.hist(durations, bins=14, color=MUTED, edgecolor="white")
ax.axvline(30, color=ACCENT, lw=1.2, ls="--")
ax.annotate("30 min floor", (30, ax.get_ylim()[1] * 0.85), color=ACCENT, fontsize=8,
            xytext=(4, 0), textcoords="offset points")
ax.set_xlabel("minutes"); ax.set_ylabel("videos")
ax.set_title(f"Measured duration of {len(durations)} downloaded Video-MME videos", loc="left")
plt.tight_layout(); plt.show()
durations.describe().round(1).to_frame("minutes")
"""),
    md("""
## 4. How the original subset was actually drawn

This is the part that was documented wrongly for a long time. The repository
claimed the questions were "filtered to those requiring audio and video". They
were not.
"""),
    code("""
pool_old = load_pool("videomme_av_all.json")
chosen = sorted(int(r["video_id"]) for r in {r["video_id"]: r for r in pool_old}.values())
block = range(chosen[0], chosen[-1] + 1)
gaps = [v for v in block if v not in chosen]

print(f"{len(chosen)} videos, {len(pool_old)} questions")
print(f"contiguous block: {chosen[0]}–{chosen[-1]}  ({len(list(block))} ids)")
print(f"gaps inside the block: {gaps}")
print(f"those ids exist in the benchmark: "
      f"{all(str(g) in set(long.video_id) for g in gaps)}  <- lost to failed downloads")
"""),
    code("""
fig, ax = plt.subplots(figsize=(10, 1.5))
all_ids = sorted(long.video_id.astype(int).unique())
ax.scatter(all_ids, [0] * len(all_ids), s=6, color=MUTED, label="long split (300)")
ax.scatter(chosen, [0] * len(chosen), s=22, color=ACCENT, label="our subset (18)")
ax.set_yticks([]); ax.set_xlabel("video_id")
ax.set_title("The subset is a contiguous block, not a selection", loc="left")
ax.legend(frameon=False, loc="upper left", ncols=2)
plt.tight_layout(); plt.show()
"""),
    md("""
## 5. The bias nobody had noticed

Counting questions whose text names speech or sound, using the same keyword test
the scripts use.
"""),
    code("""
from analyse_videomme_subset import AUDIO_MARKERS

ids = {r["question_id"] for r in pool_old}
old_sel = long[long.question_id.isin(ids)]
rates = pd.Series({
    "all 900 long questions": long.question.str.contains(AUDIO_MARKERS).mean(),
    "our original 51": old_sel.question.str.contains(AUDIO_MARKERS).mean(),
})

fig, ax = plt.subplots(figsize=(5, 1.9))
rates.plot.barh(ax=ax, color=[MUTED, ACCENT])
ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
ax.set_title("Share of questions naming speech or sound", loc="left")
plt.tight_layout(); plt.show()
rates.map("{:.0%}".format).to_frame("audio-marker rate")
"""),
    md("""
**The sample was audio-poorer than the benchmark it came from.** That bias runs
*against* the method: fewer questions where sound carries the answer means fewer
chances for cross-modal arbitration to matter. A conservative bias is a safe one
to report — but it is the wrong instrument for demonstrating the contribution,
which is what section 6 fixes.
"""),
    md("""
## 6. The balanced pool

`build_videomme_pool.load_long` buckets every question by what its text names, and
`select` picks videos while holding the buckets and the domains in balance.
"""),
    code("""
bucketed = load_long(REPO / "data/videomme-real-subset/hf/videomme/test-00000-of-00001.parquet")
population = bucketed.bucket.value_counts()
population.to_frame("questions in the long split")
"""),
    md("""
Only a handful of questions in the whole split name **both** something heard and
something seen. Getting to that number took two corrections to the keyword test:

- **`BOILERPLATE`** — Video-MME phrases many questions as *"according to what is
  shown in the video"*. That is framing, not a claim that the answer is visible;
  it appears on questions answered entirely by narration. Left in, it inflated the
  `both` bucket threefold.
- **`SHOW_AS_NOUN`** — *reality shows*, *competition shows*. The noun, not the verb.

Both are stripped before matching. The corrected count is the honest one.
"""),
    code("""
pool = select(bucketed, target=60, tolerance=6, max_domain_share=0.34)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
counts = pool.bucket.value_counts()
counts.plot.bar(ax=ax1, color=[ACCENT if b == "both" else MUTED for b in counts.index], rot=0)
ax1.set_title("Balanced pool by modality bucket", loc="left"); ax1.set_ylabel("questions")

comparison = pd.DataFrame({
    "old block": old_sel.domain.value_counts(),
    "balanced pool": pool.domain.value_counts(),
}).fillna(0)
comparison.plot.barh(ax=ax2, color=[MUTED, ACCENT])
ax2.set_title("Domain spread", loc="left"); ax2.set_xlabel("questions")
ax2.legend(frameon=False)
plt.tight_layout(); plt.show()

print(f"{pool.video_id.nunique()} videos, {len(pool)} questions")
counts.to_frame("questions").assign(share=lambda d: (d.questions / len(pool)).map("{:.0%}".format))
"""),
    md("""
The old block was **100% one domain**, Life Record. The balanced pool spans six
with none above a third — so a result can no longer be an artefact of one kind of
video.
"""),
    md("## 7. The questions that need both modalities"),
    code("""
both = pool[pool.bucket == "both"][["video_id", "domain", "task_type", "question"]]
both.reset_index(drop=True)
"""),
    md("""
These are the scarce resource and the ones that actually test the contribution:
identify someone by what they are *wearing*, then answer from what they *say*.

Together with the curated corpus — where screening 24 drafted candidates produced
one usable case — this says something about the field rather than about this
project: **questions genuinely requiring both modalities are rare even in
benchmarks marketed as multimodal.**
"""),
    md("""
## 8. Link rot

Video-MME is sourced from YouTube, and uploaders delete or privatise their videos
after a benchmark ships. Dead ids are recorded and excluded at *selection* time, so
a pool's n is known before a GPU is rented rather than discovered after a download.
"""),
    code("""
dead = [ln.split("#")[0].strip() for ln in
        (REPO / "data/eval/videomme-unavailable.txt").read_text().splitlines()
        if ln.strip() and not ln.startswith("#")]
archive = REPO / ".gist/videos/archive"
on_disk = {p.stem.removeprefix("videomme-") for p in archive.glob("videomme-*.mp4")}

rows = []
for name in ("videomme_av6.json", "videomme_av_all.json", "videomme_long.json",
             "videomme_balanced.json"):
    p = load_pool(name)
    if p is None:
        continue
    vids = {r["videoID"] for r in p}
    gone = vids & set(dead)
    rows.append({"pool": name, "questions": len(p), "videos": len(vids),
                 "downloaded": len(vids & on_disk), "dead": len(gone),
                 "questions lost": sum(1 for r in p if r["videoID"] in gone)})
pd.DataFrame(rows).set_index("pool")
"""),
    md("""
A **dead** video caps a pool's n permanently; an **unfetched** one is a download
away. Worth knowing which before renting a GPU.

---

## What this notebook is for

It renders the selection logic so a reader can see the distributions rather than
take them on trust. The committed markdown report at `reports/videomme-subset.md`
is the same analysis in text form, produced by the same functions, and is what CI
and the pod runners read.
"""),
]


def main() -> int:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1) + "\n")
    print(f"wrote {OUT} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
