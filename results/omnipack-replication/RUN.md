# Head to head against a reimplementation of OmniPack

**Status: implementation written and unit tested; the calibration and head-to-head
runs are not yet done.** Both need a GPU.

**This is not OmniPack.** It is our implementation of the method described in
Su et al., arXiv:2608.03812, written because the authors released no code. Every
number this produces must be labelled as a reimplementation. Never write "Gist
outperforms OmniPack."

## Why this exists

`github.com/RowanSu/OmniPack` is a single branch containing a two-line README:
0 KB of code, last pushed 2026-07-31, no fork or mirror. Verified against the
GitHub API on 2026-09-22. The paper announces code availability and there is
none, so no one outside the authors can compare against the current state of the
art. That sentence belongs in the paper whether or not this run ever happens, and
it should not be hedged with a compute-budget explanation: a reviewer can check
the repository in ten seconds.

## Two findings that do not need this run

Both were established by reading the paper before any code was written, and
both stand on the paper alone.

**1. OmniPack is post-encoder, in its own words.** "The modality-specific
encoder-projector pipelines map them into the LLM embedding space, producing Nv
visual tokens"; "for the encoder tokens Xm of modality m, we retain Km tokens
before the LLM". Its importance signal is the final encoder layer's attention,
which cannot be read without running the encoder. Their term **"pre-LLM" means
post-encoder**; Gist's "pre-encoder" means before both. Say this explicitly, and
quote them rather than paraphrasing.

**2. Their FLOPs exclude the encoders.** "Vision and audio encoders, modality
projectors, textual tokens, token-selection operations, and the language-model
head are excluded." So 98.0% of performance at 16.7% of FLOPs is a language-model
figure. Encoder compute -- the quantity Gist reduces -- is zeroed out by
definition in their accounting, so a comparison reported on their basis would
show Gist's saving as nothing.

This is not a criticism of their paper. Excluding a fixed cost is reasonable when
every compared method pays it identically. It stops being reasonable when a method
that does not pay it enters the table, which is exactly the argument Gist makes.

**Consequence for this run: report total cost, encoders included, and say so in
the caption.** Wall clock and peak memory do this naturally -- you cannot run a
vision tower without spending the time and the memory. Use the same measurement
harness as `results/efficiency-measured/`.

## The four assumptions

The paper specifies more than most: every hyperparameter has a value, and DPC-KNN
is given in full in Appendix B.1. Four details are missing. Each is marked
**ASSUMPTION** in `omnipack.py` and must be listed in any table caption or
appendix that reports these numbers.

| # | Gap | What the paper says | Our default |
| - | :-- | :------------------ | :---------- |
| A1 | `N(.)` has no formula | prose calls it "modality-wise min-max normalization" | min-max to [0,1] per modality |
| A2 | How attention centrality and variation signals combine | cues "are added to" the centrality vector; no coefficients; video's two variation terms have no stated relative weight | unweighted sum of normalized signals |
| A3 | `d_pos` has no formula | "normalized temporal distance for audio or spatiotemporal distance for video", each "independently normalized to [0,1]" | audio: normalized time index; video: Euclidean over normalized (frame, row, column) |
| A4 | Pooling for `q` and `p` | "mean-pool"; simple vs attention-weighted unstated | simple mean |

Paper-specified values used as given: eta_v=0.25, eta_a=0.35, lambda=0.20,
tau_v=0.10, tau_a=0.05, zeta=0.10, DPC-KNN k=7, inner retention 50%, insertion
layer 18 for Qwen2.5-Omni-7B.

## Calibrate before you compare

**Do not run the head to head first.** A reimplementation that quietly
underperforms makes Gist look good for the wrong reason, and the critique writes
itself: you implemented their method badly. Calibration is what separates "their
method scores X" from "our code has a bug".

The paper reports four pre-LLM retention ratios: 25%, 20%, 15%, 10%, with inner
compression at 50% of the pre-LLM tokens. Run the reimplementation at a setting
they report, on the benchmark they report it on, and compare against their
published figure.

Two outcomes, both publishable:

- **Lands near their number.** The comparison is credible. Say what the gap is
  and move on to the head to head.
- **Does not land near it.** Report that the published method does not reproduce
  from its description under four documented assumptions. This is a *stronger*
  result than an accuracy row, and it is the honest one.

Either way the finding does not depend on Gist winning anything.

## Run

Environment matches the other head to heads: RTX 4090 24 GB, torch 2.6.0+cu124,
`enable_audio_output=False`, videos at 360p via `results/fetch_videos.py`.

```bash
python results/fetch_videos.py /root/videomme_av6.json /content/vids
python results/omnipack-replication/omnipack_h2h.py /root/videomme_av6.json
```

Record the effective n that `fetch_videos.py` prints. A silently missing video
changes n between conditions, which is worse than a failed download.

**Sweep all four ratios.** The paper reports no single headline value, so picking
one invites the accusation that you picked their weakest setting. Sweeping is
cheap here because the model loads once.

## Cost

The method is training-free, so this is inference only. At Community Cloud RTX
4090 pricing (~$0.34/hr), budget about **$1-2** for calibration plus the sweep at
n=18, and allow $5 for a re-run. The Qwen2.5-Omni-7B pull is ~16 GB and dominates
setup.

## Order this against the other outstanding runs

Do `results/efficiency-measured/` first. It is $2 and it replaces every analytic
FLOP figure in the paper with a measured one. If its `select` stage comes back
larger than the encoding it saves, the framing of the whole contribution changes,
and that would reshape what this comparison needs to show. There is no sense
paying for a head to head whose caption you might have to rewrite.

## First run: it is a debugging session, not a measurement session

Every other runner under `results/` was written against a live pod. This one was
not, because OmniPack shipped no code to borrow a working harness from. Three
functions in `omnipack_h2h.py` are deliberately left raising `NotImplementedError`
rather than guessed:

| Function | What it must do |
| :------- | :-------------- |
| `build_inputs` | processor inputs for one video plus prompt, at the configured frames and max_pixels |
| `encode` | run both towers, return visual and audio token tensors plus `frames`, `patches`, `media_span`, `text_span` |
| `generate` | splice compressed embeddings into `inputs_embeds` and generate |

They are unimplemented on purpose. Their correct form depends on the transformers
version installed on the pod -- the processor's video and audio keyword names,
whether the visual tower takes `grid_thw` positionally, and where media tokens
land in the prompt all moved between releases. A guess would produce a file that
looks finished and fails three layers from the cause, burning pod time.

Fill them in against the installed version, using OmniScope's
`tools/qwenomni_prune_inference.py` as the worked example: it does exactly this,
for the same checkpoint, and is already cloned for the OmniScope head to head.

Order for the first session:

1. Implement the three functions. Confirm `decoder_layers` finds the layer list.
2. **Run the `full` arm alone and check it reproduces stock Qwen answers.** If the
   control arm is wrong, nothing downstream means anything. This is the single
   most important check in the session.
3. Run stage 1 only, `OMNIPACK_STAGE2=0`. Confirm the token counts in the `select`
   records match the retention ratio.
4. Enable stage 2. The hook prunes mid-forward, which changes sequence length and
   invalidates the attention mask, position ids and KV cache for later layers; it
   fires on prefill only. **This is the part of the implementation most likely to
   be wrong.** If answers degrade sharply or generation breaks, fall back to
   stage 1 only and say so in the write-up rather than shipping a broken stage 2.
5. Only then calibrate, then sweep.

A stage-1-only result is still publishable. It is the half of the method that
carries the placement argument, and the honest caption says stage 2 was not
validated.

## What is not done here

- The three model-shaped functions above, and therefore any execution at all.
- Validation of the stage 2 hook against a live forward pass.
- Calibration against the paper's reported numbers.
- The head to head and the ratio sweep.
