# results/

Committed experimental record. Everything here survives a fresh clone, which is the
point: `reports/` is gitignored, and the 2026-09-16 citation audit found a published
table row whose raw data existed only on one laptop.

## What is here

| Directory | What it is | Run? |
| :-------- | :--------- | :--- |
| `runpod-fp16/` | Gist vs uniform, Qwen2.5-Omni-7B fp16 unquantised, n=18. Plus `bootstrap.md`, the pre-registered non-inferiority test | yes |
| `n51-4bit/` | The n=51 4-bit run behind Table II. Raw per-question output | yes |
| `runpod-omnizip-h2h/` | Head to head against OmniZip, the training-free method Gist was actually benchmarked against | yes |
| `heuristics-ablation/` | Per-intent heuristics on vs off, 39 cases, plus `bootstrap-dev.md` on the frozen dev split | yes |
| `split-budget-ablation/` | Pooled vs split-then-rank allocation *inside* Gist | yes |
| `efficiency-measured/` | Wall clock and peak GPU memory, RTX 4090, n=18. Answer stage 1.37x faster and 29.4% less activation memory than uniform. Selection costs a median 88 s per video to ingest, then 1.26 s per question from cached embeddings (`cached_selection.py`). Tail latency, error rate and budgets in `latency.md`. Outcome in `POD_SESSION.md` | yes |
| `runpod-omniscope-h2h/` | Head to head against OmniScope, the closest published method to Gist | **no** |
| `omnipack-replication/` | A reimplementation of OmniPack, written because the authors released no code. Ratio sweep at n=48: 22/22/24/20 correct against 22 uncompressed | yes |
| `load-test/` | k6 load test of the library API, laptop CPU: browsing holds 200 users, 40 different questions hold 25 users and break at 50 (P95 5.5 s). Found and fixed the encoders reloading on every question | yes |
| `web-performance/` | Lighthouse on both pages and the main flow under emulated Fast 4G, Slow 4G and Slow 3G: every flow completes on Slow 3G; the video page missed the 2.5 s LCP target on a throttled phone (3.2 s); server-rendering it brought that to 1.58 s | yes |
| `edge-cases/` | Silent, no-audio, music-only, French, 3-second and static videos, and unanswerable, Chinese, emoji, very long and blank questions, through the real library path. Nothing crashed; one silent defect found and fixed (long questions dropped every audio score) | yes |
| `held-out/` | The 12 cases held out by recording, reported separately (2026-11-06): full Gist 58.3% [33.3, 83.3] against 82.6% on dev; ahead of uniform, visual-only and top-k; tied with transcript-only. Not a clean unseen test: these cases were also run inside both 35-case ablations | yes |
| `fetch_videos.py` | Shared video fetcher for every runner above | — |

## Suggested order on a fresh pod

Two runs are outstanding. Do them in this order, and read the first before paying
for the second.

**1. `efficiency-measured/` — about $2, one hour.**

The highest-value work left. Every efficiency figure the project quotes is currently
analytic, derived from encoder configs, and peak memory cannot be derived from a FLOP
count at all. This measures both. Until it exists, the paper's central claim is
supported by arithmetic rather than observation.

Read the `select` stage before deciding anything else. It is Gist's own scoring cost,
and if it exceeds the encoding it saves then the contribution needs rephrasing and
the OmniScope run can wait.

**2. `runpod-omniscope-h2h/` — about $0.50, one hour.**

OmniScope converged independently on Gist's query-conditioned per-modality
arbitration, uses the same CLIP signal family, is training-free, and is the
split-budget shape that `split-budget-ablation/` could only simulate internally.

It also settles the placement question from source rather than from prose: its own
`get_cached_video_embeds` runs the full visual encoder over every frame, and
`get_cached_audio_embeds` the full audio tower, *before* anything is pruned.

**Not runnable: OmniPack.** The current state of the art (arXiv 2608.03812) published
an empty repository - one branch, a two line README, zero kilobytes of code, last
pushed 2026-07-31. Of the twelve audiovisual methods surveyed, only OmniZip, OmniSIFT
and OmniScope shipped working implementations. This belongs in the paper as a stated
fact, because "the SOTA has not released its code" is checkable, and it is a much
better sentence than any hedge about compute budgets.

## Run index

One row per measured run: what produced it, on what, and the commit that added its raw
output as it stands now. Every row's scorer models (CLIP, CLAP, MiniLM, faster-whisper) are the commits
pinned in `src/gist/model_revisions.py`, and every row that answers questions uses
Qwen2.5-Omni-7B at commit `ae9e169`. Run dates are taken from each directory's own
README; where none is stated, the column gives the date the raw output was committed.

| Run | n | Answering model | Hardware | Run date | Raw output | Raw output commit |
| :-- | --: | :-- | :-- | :-- | :-- | :-- |
| `runpod-fp16/` | 18 | Qwen2.5-Omni-7B fp16 | RTX 4090 24 GB | committed 2026-07-06 | `av7bfp16_results.jsonl` | `c70f7c0` |
| `runpod-omnizip-h2h/` | 18 | Qwen2.5-Omni-7B fp16 | RTX 4090 24 GB | committed 2026-07-06 | `omnizip_h2h_results.jsonl` | `da6cea1` |
| `n51-4bit/` | 51 | Qwen2.5-Omni-7B 4-bit | not recorded | promoted 2026-09-22 | `av7b_51q_results.jsonl` | `2d1bf7f` |
| `heuristics-ablation/` | 35 | none, selection scored against reference timestamps | not recorded | re-run 2026-09-22 | `ablation-heuristics-ab.json` | `054e118` |
| `split-budget-ablation/` | 35 | none, selection scored against reference timestamps | not recorded | re-run 2026-09-22 | `split-budget-ablation.json` | `054e118` |
| `efficiency-measured/` | 18 | Qwen2.5-Omni-7B fp16 | RTX 4090 24 GB; cached selection on a laptop CPU | 2026-10-28 | `efficiency_measurements.jsonl`, `cached_selection.jsonl` | `f074be4`, `984b7ae` |
| `omnipack-replication/` | 48 | Qwen2.5-Omni-7B fp16 | RTX 4090 24 GB | 2026-10-28 | `omnipack_h2h_results.jsonl` | `f074be4` |
| `gist-vs-omnipack/` | 48 | Qwen2.5-Omni-7B fp16 | RTX 4090 24 GB | committed 2026-11-04 | `answers.jsonl`, `measurements.jsonl` | `e909b65` |
| `gist-vs-omnipack-long/` | 48 | Qwen2.5-Omni-7B fp16 | RTX 4090 24 GB | committed 2026-11-04 | `answers.jsonl`, `measurements.jsonl` | `9eda2cd` |

Gist trains nothing, so there are no loss or accuracy curves to track. The equivalent
record is the per-question output above, the frozen split in `data/eval/splits/`, and
the pinned model commits. A run is reproduced by checking out its commit and running
the script named in its directory.

## Rules this directory follows

- **Raw per-question output, not just summaries.** Two of Table II's four rows have
  no per-question output anywhere and are recoverable only from a commit hash. Do
  not add a third.
- **Name the baseline beside every ratio.** 97.1% and 36.7% are against *different*
  baselines, and either one quoted alone misleads.
- **Analytic and measured numbers never share a table.** `gist.eval.efficiency`
  enforces this by reporting them separately.
- **Caveats live with the result, not in the paper only.** Every README here states
  what its run does not show.
