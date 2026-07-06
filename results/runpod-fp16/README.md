# fp16 unquantized Qwen2.5-Omni-7B — Gist vs full (RunPod RTX 4090)

Closes the fp16 OOM caveat from the earlier free-tier T4 attempt: the 7B Omni-LLM
was run **unquantized (fp16, ~16 GB weights)** on a single 24 GB RTX 4090, alongside
the CLIP/CLAP/Whisper scorers, with no OOM.

**Runner:** `scripts/av_bench_7b_fp16.py` (arg = questions JSON)
**Env:** RTX 4090 24 GB, torch 2.6.0+cu124, transformers 5.13, `enable_audio_output=False`
**Data:** 18 Video-MME AV questions across 6 videos (`.gist/benchmark/videomme_av6.json`),
64 candidate frames + 30 s audio windows per video.

**Results** (`av7bfp16_results.jsonl`, one row per question):

| Condition | Accuracy | avg frames | avg audio |
| :-------- | -------: | ---------: | --------: |
| Full (uniform) | 5/18 (28%) | 8 | 4 |
| Gist-selected  | 6/18 (33%) | 2.7 | 3.2 |

Gist edges full by one question at ~1/3 the visual budget, unquantized — consistent
with the 4-bit n=51 result: **parity/slight-edge at far less compute; quantization
was not masking anything.**

Caveat: absolute accuracy is near-chance (~25% for 4-option MC) on this hard
long-video subset with sparse evidence; the durable claim is the *structural* one
(matched accuracy at reduced compute), and at n=18 the 5-vs-6 gap is within noise.
