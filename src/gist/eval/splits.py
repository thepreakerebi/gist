"""Freeze a held-out split of the evaluation corpus.

Why this exists: the coverage heuristics, the presets and the selector's
thresholds were all tuned while looking at the long-video suite. Any number
produced on that suite is therefore a *development* number, however honestly it
was measured. A held-out split is the only thing that can answer "does this
generalize, or did you fit it?" — and its value decays with every further
tuning run against the corpus, so it is frozen once and then left alone.

**The split is grouped by source video, not by case.** Cases drawn from the same
recording share a speaker, a visual style, a transcript quality and an OCR
failure mode; putting some in dev and some in held-out would let the method be
tuned on a video it is then tested on. Grouping costs balance — the corpus is
skewed, so no grouped split is evenly stratified — but a leaky split measures
nothing at all.

The manifest records a hash of the case ids it selected. If the corpus changes,
the hash stops matching and the split is stale rather than silently wrong.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DATASET = Path("data/eval/long-video-quality.jsonl")
MANIFEST = Path("data/eval/splits/held-out.json")

# Chosen deliberately rather than sampled. The held-out side must (a) contain
# videos never tuned against, (b) touch every query category, and above all
# (c) carry real weight in speech_semantic, because cross-modal arbitration is
# the project's central claim and that is the stratum where it is load-bearing.
# The two bio-motor-control lectures stay in dev: they are 20 of 39 cases, and
# holding either out would leave dev unable to exercise visual_object_action.
HELD_OUT_VIDEOS = (
    "microsoft-kinect-keynote-art-code-2011",
    "night-of-the-living-dead-1968",
    "nasa-sts115-postflight-briefing-2006",
    "paul-graham-y-combinator",
    "yt-arthistory-high-renaissance",
    "yt-history-augustus-rome",
)


def source_video(case: dict[str, Any]) -> str:
    """Recover the recording a case was authored against."""

    path = case.get("compression_path") or ""
    parts = Path(path).parts
    # .gist/{runs,curation}/<video>/<query-slug>/compression.json
    return parts[2] if len(parts) > 2 else "unknown"


def load_cases(dataset: Path = DATASET) -> list[dict[str, Any]]:
    return [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]


def build_manifest(dataset: Path = DATASET) -> dict[str, Any]:
    cases = load_cases(dataset)
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_video[source_video(case)].append(case)

    unknown = set(HELD_OUT_VIDEOS) - set(by_video)
    if unknown:
        raise SystemExit(f"held-out videos not present in the corpus: {sorted(unknown)}")

    held_out = [case for video in HELD_OUT_VIDEOS for case in by_video[video]]
    held_ids = {case["id"] for case in held_out}
    dev = [case for case in cases if case["id"] not in held_ids]

    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "cases": len(subset),
            "videos": sorted({source_video(case) for case in subset}),
            "categories": dict(
                sorted(Counter(case.get("query_category") for case in subset).items())
            ),
        }

    ids = sorted(held_ids)
    return {
        "created": datetime.now(UTC).strftime("%Y-%m-%d"),
        "dataset": str(dataset),
        "policy": (
            "Grouped by source video: no recording appears in both splits. "
            "Frozen once; the held-out split is to be executed a single time, "
            "after development against the dev split is complete."
        ),
        "corpus_fingerprint": hashlib.sha256(
            "\n".join(sorted(case["id"] for case in cases)).encode()
        ).hexdigest()[:16],
        "held_out_fingerprint": hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16],
        "held_out": {**summarize(held_out), "case_ids": ids},
        "dev": {**summarize(dev), "case_ids": sorted(case["id"] for case in dev)},
    }


def load_manifest(manifest: Path = MANIFEST) -> dict[str, Any]:
    if not manifest.exists():
        raise SystemExit(f"no frozen split at {manifest}; run: python -m gist.eval.splits --write")
    return json.loads(manifest.read_text())


def verify(manifest: Path = MANIFEST, dataset: Path = DATASET) -> bool:
    """Check the frozen split still matches the corpus it was cut from."""

    frozen = load_manifest(manifest)
    current = hashlib.sha256(
        "\n".join(sorted(case["id"] for case in load_cases(dataset))).encode()
    ).hexdigest()[:16]
    return frozen["corpus_fingerprint"] == current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--write", action="store_true", help="Freeze the split to disk.")
    parser.add_argument("--verify", action="store_true", help="Check the frozen split is current.")
    args = parser.parse_args(argv)

    if args.verify:
        ok = verify(args.manifest, args.dataset)
        print("split is current" if ok else "SPLIT IS STALE — the corpus changed since freezing")
        return 0 if ok else 1

    manifest = build_manifest(args.dataset)
    held, dev = manifest["held_out"], manifest["dev"]

    print(f"dev      {dev['cases']:>3} cases  {len(dev['videos'])} videos  {dev['categories']}")
    print(f"held-out {held['cases']:>3} cases  {len(held['videos'])} videos  {held['categories']}")
    print(f"fingerprint {manifest['held_out_fingerprint']}")

    if args.write:
        if args.manifest.exists():
            existing = json.loads(args.manifest.read_text())
            if existing["held_out_fingerprint"] != manifest["held_out_fingerprint"]:
                raise SystemExit(
                    "a different split is already frozen at this path. Re-cutting a "
                    "held-out split after development invalidates it; delete the file "
                    "deliberately if that is genuinely what you intend."
                )
            print("unchanged")
            return 0
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"froze {args.manifest}")
    else:
        print("(dry run — pass --write to freeze)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
