# SOTA head-to-head: OmniZip vs full-context — Qwen2.5-Omni-7B (RunPod RTX 4090)

Runs OmniZip's **actual method** (its `omnizip.modeling_qwen2_5_omni` fork +
`model.thinker.omnizip_config`, transformers==4.52.3 as OmniZip pins) directly on
our 18 Video-MME AV questions — **not** lmms-eval, so no full-dataset download.
Each question is answered twice through the identical OmniZip model:

- **full**    = `omnizip_config = None` (no token compression)
- **omnizip** = paper defaults `rho_video=0.6, rho_audio=0.3, g=3, contextual_ratio=0.05`

**Result (18 questions, 8 frames each):**

| Condition | Accuracy | Config |
| :-------- | -------: | :----- |
| Full (uncompressed) | 6/18 (33%) | rho_video=1.0 |
| OmniZip (compressed) | 6/18 (33%) | rho_video=0.6, rho_audio=0.3 |

**OmniZip's token pruning matched full-context on 18/18 questions (100% answer
agreement)** — it compressed the token stream without changing a single answer.

## How this compares to Gist (the structural point)

On these *same 18 questions*, Gist scored 6/18 (33%) at ~2.7 frames
(`results/runpod-fp16/`). All three land at the same accuracy — the difference is
**where** compute is spent:

- **OmniZip** encodes the dense video+audio, then prunes tokens *inside* the LLM
  forward (post-encoder compression). Its peak memory is the *uncompressed*
  footprint — which is why it needs ≥40–80 GB for full-length videos and OOM'd on
  a 24 GB card until inputs were trimmed to 90 s (see notes below).
- **Gist** scores + selects a handful of frames/windows *before* the encoders run
  (pre-encoder compression), so it never materializes the dense footprint and ran
  unquantized on 24 GB with room to spare.

At matched accuracy, Gist's pre-encoder selection saves the dual-encoder FLOPs
that OmniZip still pays — the capstone's structural thesis.

## Honest constraints on this run (24 GB budget)

- **Videos trimmed to 90 s.** With `use_audio_in_video=True`, OmniZip's audio
  encoder allocates ~16.8 GB on full-length (34–58 min) clips *before* pruning —
  OOM on 24 GB (OmniZip's paper uses 80 GB GPUs). Trimming to 90 s (and 8 frames)
  made every question fit. Both conditions see identical trimmed input, so the
  full-vs-OmniZip comparison is fair; it just runs on 90 s clips.
- **One-line robustness patch to OmniZip.** `omnizip_core.py` computed the audio
  top-k count `k` from the token count `N` but applied it to a shorter
  `attn_logits` tensor, raising `selected index k out of range` on short clips.
  Clamped `k = min(k, attn_logits.shape[-1])` — a safety clamp that doesn't change
  behavior on normal-length inputs.
- **n=18, near-chance absolute accuracy** (~25% floor for 4-option MC on a hard
  long-video subset with 8 frames). The durable finding is the *structural*
  comparison and OmniZip's 100% answer-preservation, not the absolute number.

Cost: single RTX 4090 @ $0.69/hr; pod terminated after the run.
