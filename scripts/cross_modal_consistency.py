"""Cross-modal consistency diagnostic (plan §8).

Is Gist genuinely fusing both modalities, or leaning on one? For each speech
case we score the answer-bearing segment (the ground-truth ``relevant_ranges``)
independently in each modality against the query:

  audio_rel  = semantic cosine(query, transcript window overlapping the range)
  visual_rel = CLIP cosine(query, frames whose timestamp falls in the range)

Both are turned into a *percentile* against that case's own full distribution of
window / frame scores, so the two modalities are on a comparable 0..1 scale. We
then classify the answer segment:

  consensus       — both modalities rank the answer segment high (>= HI pct)
  vision-dominant — visual high, audio low
  audio-dominant  — audio high, visual low
  weak            — neither modality localizes it

A healthy cross-modal system shows a mix (some consensus, some single-modality
dominant) — proving each modality contributes independent signal on different
questions, not that one modality is decorative.

Reuses the transcripts cached by scripts/ab_semantic_vs_lexical.py (run that
first) and runs CLIP on CPU over only the answer-range frames, so it is cheap.
"""

from __future__ import annotations

import json
from pathlib import Path

from gist.core.semantic import SemanticTextScorer

DATA = "data/eval/long-video-quality.jsonl"
AB_CACHE = Path(".gist/ab-transcripts.json")
HI, LO = 0.80, 0.50  # percentile thresholds for "high" / "low" relevance


def pct_rank(value: float, pool: list[float]) -> float:
    """Fraction of pool values <= value (0..1)."""
    if not pool:
        return 0.0
    return sum(1 for v in pool if v <= value) / len(pool)


def in_range(t: float, ranges: list[dict], tol: float) -> bool:
    return any(r["start_seconds"] - tol <= t <= r["end_seconds"] + tol for r in ranges)


def classify(a_pct: float, v_pct: float) -> str:
    a_hi, v_hi = a_pct >= HI, v_pct >= HI
    a_lo, v_lo = a_pct <= LO, v_pct <= LO
    if a_hi and v_hi:
        return "consensus"
    if v_hi and a_lo:
        return "vision-dominant"
    if a_hi and v_lo:
        return "audio-dominant"
    if a_hi or v_hi:
        return "consensus" if not (a_lo or v_lo) else ("audio-dominant" if a_hi else "vision-dominant")
    return "weak"


def main() -> None:
    if not AB_CACHE.exists():
        raise SystemExit("Run scripts/ab_semantic_vs_lexical.py first to build .gist/ab-transcripts.json")
    transcripts = json.load(open(AB_CACHE))
    sem = SemanticTextScorer()

    from gist.media.models import ExtractedFrame
    from gist.vision.clip import HuggingFaceClipFrameScorer

    clip = HuggingFaceClipFrameScorer()

    cases = [json.loads(l) for l in open(DATA)]
    cases = [c for c in cases if c.get("query_category") == "speech_semantic"]

    counts: dict[str, int] = {"consensus": 0, "vision-dominant": 0, "audio-dominant": 0, "weak": 0}
    rows = []
    for c in cases:
        art = json.load(open(c["compression_path"]))
        query = art["compression"]["query"]
        aw = art["ingestion"]["audio_windows"]
        frames_meta = art["ingestion"]["frames"]
        ranges = c["relevant_ranges"]
        tol = c.get("timestamp_tolerance_seconds", 30)

        # --- audio: percentile of the answer-window semantic score ---
        win_texts = [transcripts.get(w["path"], "") for w in aw]
        win_scores = sem.score_texts(query, win_texts)
        ans_audio = [
            win_scores[i]
            for i, w in enumerate(aw)
            if in_range(w["start_seconds"] + w["duration_seconds"] / 2, ranges, tol)
        ]
        a_val = max(ans_audio) if ans_audio else 0.0
        a_pct = pct_rank(a_val, win_scores)

        # --- visual: CLIP over answer-range frames vs a sampled background ---
        ans_frames = [f for f in frames_meta if in_range(f["timestamp_seconds"], ranges, tol)]
        # background sample (every ~20th frame) for the percentile pool
        bg = frames_meta[:: max(1, len(frames_meta) // 40)]
        pool_meta = {f["index"]: f for f in bg}
        for f in ans_frames:
            pool_meta[f["index"]] = f
        ef = [
            ExtractedFrame(index=f["index"], timestamp_seconds=f["timestamp_seconds"], path=Path(f["path"]))
            for f in pool_meta.values()
            if Path(f["path"]).exists()
        ]
        v_val, v_pct = 0.0, 0.0
        if ef:
            vs = clip.score_frames(ef, query=query)  # {path: score}
            all_v = list(vs.values())
            ans_v = [vs.get(Path(f["path"]), 0.0) for f in ans_frames if Path(f["path"]).exists()]
            v_val = max(ans_v) if ans_v else 0.0
            v_pct = pct_rank(v_val, all_v)

        label = classify(a_pct, v_pct)
        counts[label] += 1
        rows.append((c["id"], round(a_pct, 2), round(v_pct, 2), label))

    print(f"{'case':44} {'audio_pct':9} {'vis_pct':7} label")
    for cid, ap, vp, lab in rows:
        print(f"{cid[:44]:44} {ap:<9} {vp:<7} {lab}")
    n = sum(counts.values())
    print("\nsegment classification over", n, "answer segments:")
    for k in ["consensus", "vision-dominant", "audio-dominant", "weak"]:
        print(f"  {k:16} {counts[k]:2}  ({counts[k]/n:.0%})")
    localized = n - counts["weak"]
    print(f"\nlocalized by >=1 modality: {localized}/{n} ({localized/n:.0%})")


if __name__ == "__main__":
    main()
