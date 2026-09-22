# n=51 AV benchmark, Qwen2.5-Omni-7B 4-bit

Raw per-question output for the **n=51, 7B, 4-bit** row of Table II in the paper.

Promoted out of the gitignored `reports/` on 2026-09-22. The citation audit
(2026-09-16) traced all four Table II rows and found that two of them had no
per-question output kept anywhere, recoverable only from a commit hash and a
script. This one *did* still have its raw data, and it was sitting in a directory
that never reaches GitHub. One laptop failure and a published table row becomes
unreproducible, so it now lives under `results/` with the rest of the record.

## Numbers, recomputed from the JSONL on promotion

| Condition | Accuracy | avg frames | avg audio windows |
| :-------- | -------: | ---------: | ----------------: |
| Full (uniform) | 25/51 (49%) | 8 | — |
| Gist-selected  | 26/51 (51%) | 3.000 | 2.961 |

Answer agreement between the two conditions: 30/51.

These reconcile exactly with the figures quoted in the paper and in the audit,
which is the point of keeping the raw file rather than the summary.

## What this run is and is not

- **Is:** the largest-n accuracy comparison the project has. Gist edges the
  uniform baseline by one question at ~3/8 of the visual budget.
- **Is not:** a claim of significance. One question at n=51 is well inside
  noise, and the bootstrap machinery in `gist.eval.bootstrap` exists precisely
  so that this is stated rather than glossed. Read it alongside
  `results/runpod-fp16/bootstrap.md`.
- **Is not:** comparable to published Video-MME scores. See
  `results/runpod-omnizip-h2h/README.md` for why — OmniZip scores 33% on our
  subset against 65.9 on the full benchmark.

## Provenance

- Model: Qwen2.5-Omni-7B, 4-bit quantised
- Questions: 51 Video-MME AV questions across 18 videos
  (`.gist/benchmark/videomme_av_all.json`)
- Runner: `scripts/av_bench_7b_colab.py`, recorded by commit `3bc27d0`
- The fp16 unquantised repeat of the smaller 18-question set is in
  `results/runpod-fp16/`, and confirms quantisation was not masking anything.
