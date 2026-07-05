"""A/B: semantic vs lexical transcript-span selection on real speech cases.

LLM-free selection-quality test. For each speech_semantic quality case we have
the full transcript per 30s audio window (from the Whisper cache) plus a
ground-truth ``relevant_ranges`` (where the answer is actually spoken). We score
every window two ways and take the top-K:

  lexical  = scoring.lexical_relevance (token overlap)
  semantic = SemanticTextScorer (all-MiniLM-L6-v2 cosine sim)

Metric = did a selected window land inside the answer range (± tolerance)?
Report per-case hit + gold-window rank for each method, then aggregate hit-rate.
This measures exactly what #1 (semantic span selection) is supposed to fix.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

from gist.core.schemas import Candidate
from gist.core.scoring import lexical_relevance
from gist.core.semantic import SemanticTextScorer

K = 4  # audio budget used across the eval harnesses
DATA = "data/eval/long-video-quality.jsonl"
AB_CACHE = Path(".gist/ab-transcripts.json")


def load_ab_cache() -> dict[str, str]:
    if AB_CACHE.exists():
        return json.load(open(AB_CACHE))
    return {}


def transcribe_missing(paths: list[str], cache: dict[str, str]) -> None:
    """Transcribe any wav not yet in the A/B cache (the shared Whisper cache
    holds stale empty entries for these metadata-refresh paths, so we run fresh
    and store under our own key)."""
    todo = [p for p in paths if p not in cache]
    if not todo:
        return
    from faster_whisper import WhisperModel

    print(f"transcribing {len(todo)} windows (base/int8/cpu)...", flush=True)
    model = WhisperModel("base", device="cpu", compute_type="int8")
    for n, p in enumerate(todo, 1):
        try:
            segments, _ = model.transcribe(p, beam_size=1)
            cache[p] = " ".join(s.text for s in segments).strip()
        except Exception as exc:  # keep going; empty text just scores 0
            cache[p] = ""
            print(f"  fail {p}: {exc}", flush=True)
        if n % 50 == 0:
            AB_CACHE.write_text(json.dumps(cache))
            print(f"  {n}/{len(todo)}", flush=True)
    AB_CACHE.write_text(json.dumps(cache))


def in_range(mid: float, ranges: list[dict], tol: float) -> bool:
    for r in ranges:
        if r["start_seconds"] - tol <= mid <= r["end_seconds"] + tol:
            return True
    return False


def gold_rank(order: list[int], mids: list[float], ranges: list[dict], tol: float) -> int | None:
    """1-indexed rank of the first selected window that lands in the answer range."""
    for pos, i in enumerate(order, start=1):
        if in_range(mids[i], ranges, tol):
            return pos
    return None


def main() -> None:
    scorer = SemanticTextScorer()

    cases = [json.loads(l) for l in open(DATA)]
    cases = [c for c in cases if c.get("query_category") == "speech_semantic"]

    # Collect every window path across the cases, transcribe misses once.
    cache = load_ab_cache()
    all_paths: list[str] = []
    for c in cases:
        art = json.load(open(c["compression_path"]))
        for w in art["ingestion"]["audio_windows"]:
            all_paths.append(w["path"])
    transcribe_missing(sorted(set(all_paths)), cache)

    agg = {"lexical_hit": 0, "semantic_hit": 0, "n": 0}
    rows = []
    for c in cases:
        art = json.load(open(c["compression_path"]))
        aw = art["ingestion"]["audio_windows"]
        query = art["compression"]["query"]
        ranges = c["relevant_ranges"]
        tol = c.get("timestamp_tolerance_seconds", 30)

        mids, texts = [], []
        for w in aw:
            mid = w["start_seconds"] + w["duration_seconds"] / 2
            mids.append(mid)
            texts.append(cache.get(w["path"], ""))

        # lexical scores via candidates (mirrors the real selection path)
        cands = [Candidate(id=str(i), timestamp_seconds=mids[i], text=texts[i]) for i in range(len(aw))]
        lex = [lexical_relevance(query, cd) for cd in cands]
        sem = scorer.score_texts(query, texts)

        lex_order = sorted(range(len(aw)), key=lambda i: lex[i], reverse=True)
        sem_order = sorted(range(len(aw)), key=lambda i: sem[i], reverse=True)

        lex_hit = any(in_range(mids[i], ranges, tol) for i in lex_order[:K])
        sem_hit = any(in_range(mids[i], ranges, tol) for i in sem_order[:K])
        lex_rank = gold_rank(lex_order, mids, ranges, tol)
        sem_rank = gold_rank(sem_order, mids, ranges, tol)

        agg["lexical_hit"] += int(lex_hit)
        agg["semantic_hit"] += int(sem_hit)
        agg["n"] += 1
        rows.append((c["id"], lex_hit, sem_hit, lex_rank, sem_rank))

    print(f"{'case':44} {'lex@%d'%K:6} {'sem@%d'%K:6} {'lex_rank':8} {'sem_rank':8}")
    for cid, lh, sh, lr, sr in rows:
        mark = ""
        if sh and not lh:
            mark = "  <- semantic wins"
        elif lh and not sh:
            mark = "  <- lexical wins"
        print(f"{cid[:44]:44} {str(lh):6} {str(sh):6} {str(lr):8} {str(sr):8}{mark}")

    n = agg["n"]
    print(f"\nlexical  hit@{K}: {agg['lexical_hit']}/{n} ({agg['lexical_hit']/n:.0%})")
    print(f"semantic hit@{K}: {agg['semantic_hit']}/{n} ({agg['semantic_hit']/n:.0%})")


if __name__ == "__main__":
    main()
