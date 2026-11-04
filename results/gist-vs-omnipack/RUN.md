# Gist against the OmniPack rebuild, on common ground

Earlier results compared each method with its own baseline:

- `omnipack-replication/` compared the OmniPack rebuild with uncompressed Qwen.
- `efficiency-measured/` compared Gist with uniform sampling.

The two used different inputs, prompts and answer paths, so their numbers cannot
be compared with each other. This directory compares the methods directly.

## What is held fixed

- **Same input.** Every arm starts from the 32 frames and the audio track that
  `qwen_omni_utils` gives Qwen2.5-Omni-7B for the video.
- **Same model, prompt and answer path.** All arms use fp16 and the same
  multiple-choice prompt. They answer through the hand-written prefill and greedy
  decode in `omnipack_h2h.py`. That decode is checked against stock `generate` on
  the first question.
- **Same token positions.** Kept tokens keep their original rope positions in
  every arm (assumption A7).
- **Same GPU and process.** All arms run on one RTX 4090, in one process, on the
  same 48 Video-MME questions.

## The shared unit

Qwen encodes frames in pairs (one temporal patch). The harness encodes audio in
segments aligned to the audio tower's attention chunks. Attention never crosses
either unit, so encoding a subset of units gives the same tokens as encoding
everything and slicing. The run checks this and records it as an `exactness`
note. On CPU the maximum difference was 0.

| Arm | Encoder work | Tokens to the language model |
| :-- | :-- | :-- |
| `full` | every unit | all |
| `omnipack_r*` | every unit, plus the final-layer attention readout | stage 1 at the ratio, then stage 2 inside the LLM |
| `gist` | only the units holding the frames and windows Gist chose | those units' tokens |
| `uniform` | as many units as Gist chose, evenly spaced | those units' tokens |

Gist's choice is made before any Qwen encoder runs. CLIP scores the 32 frames.
CLAP scores the audio segments. Whisper transcribes the segments routed to
speech, and MiniLM matches the transcripts to the question. The compressor
settings are the ones behind every earlier Gist result.

## What is measured

Each stage is measured on the same GPU and tagged `per_video` or
`per_question`. Every stage records:

- **Wall time.**
- **Peak memory.**
- **GPU energy**, from NVML's board energy counter. This covers Whisper too,
  which runs in CTranslate2 rather than PyTorch.

**FLOPs** are counted with `FlopCounterMode` in a separate two-video pass,
because the counter slows the code it counts. They are carried to the full run
by per-unit rates. Whisper's FLOPs are derived from its architecture.

**Product Whisper sensitivity.** The Whisper setting the product uses (base,
beam 3, every segment) is timed beside the evaluation setting (tiny, beam 1,
speech segments). It is not used for selection.

## How the analysis assigns cost

A post-encoder method pays its encoders once per video and reuses the tokens for
every question. Gist pays its scorers once per video, and pays the Qwen encoders
on a few units for each question. `analyze.py` therefore reports per-video and
per-question cost separately. It also reports the number of questions per video
beyond which one method's total is lower than the other's.

## Running

On CPU first. This needs a transformers 4.52.3 environment:

    python results/gist-vs-omnipack/smoke.py
    H2H_COUNT_FLOPS=1 python results/gist-vs-omnipack/smoke.py

On the pod, after `setup`:

    bash results/pod_session.sh h2h_plumb
    bash results/pod_session.sh h2h
    bash results/pod_session.sh h2h_flops

Then, at home:

    python results/gist-vs-omnipack/analyze.py results/gist-vs-omnipack

## Limits

- **The OmniPack arm is a rebuild**, under the seven assumptions in
  `omnipack-replication/RUN.md`. It is not the authors' code.
- **CPU work is reported as time only**, because CPU energy cannot be read
  inside a container.
- **Gist's candidate pool is restricted to the 32 frames Qwen is given.** This
  keeps the input identical across arms. In normal use Gist scores a larger
  pool.
