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
| `efficiency-measured/` | Wall clock and peak GPU memory. Replaces the analytic FLOPs | **no** |
| `runpod-omniscope-h2h/` | Head to head against OmniScope, the closest published method to Gist | **no** |
| `omnipack-replication/` | Our reimplementation of OmniPack, written because they released no code. Unit tested; calibration and head to head outstanding | **no** |
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
