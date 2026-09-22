# Head to head against OmniScope

**Status: not yet run.** The runner is written against OmniScope's released API and
needs an NVIDIA GPU, which the development MacBook does not have.

## Why OmniScope, and not OmniPack

OmniPack (arXiv 2608.03812) is the current state of the art and the obvious
comparison. **Its code was never released.** As of 2026-09-22 the repository at
`github.com/RowanSu/OmniPack` contains a two line README and nothing else: one
branch, zero kilobytes of code, last pushed 2026-07-31. No fork or mirror carries
an implementation. The paper's "code is available" is not, at present, true.

That is worth stating plainly in the paper, because it converts a weakness into a
verifiable fact. "We did not benchmark against the state of the art" invites the
assumption that we could not afford to. "The state of the art has not released its
implementation, so no one outside the authors can reproduce or compare against it"
is a different statement, and it is checkable by anyone in ten seconds.

Of the twelve audiovisual methods surveyed, exactly three have working public
implementations: OmniZip (already benchmarked, `results/runpod-omnizip-h2h/`),
OmniSIFT (`dingyue772/OmniSIFT`, trained) and OmniScope (`MAC-AutoML/OmniScope`,
training-free). OmniFocus, OmniDelta, OmniDrop, Macer, ReMo and OmniRefine have no
public code under their paper names.

OmniScope is the most informative of the three:

- It **converged on Gist's central idea** independently: query-conditioned,
  per-modality relevance with separate budgets. It even scores frames with
  CLIP ViT-L/14@336 against the question, which is Gist's first pillar.
- It is the **split-budget shape** that `results/split-budget-ablation/` could only
  simulate internally. That ablation was explicit that it did not compare against
  OmniScope. This closes that gap.
- It is **training-free**, so the comparison holds training constant and isolates
  placement and allocation, which are the variables under study.

## The placement claim, settled by their code rather than their prose

This is the most useful thing in this directory. OmniScope's paper describes a
"pre-LLM" compression stage, which is easy to misread as pre-encoder. Their own
implementation settles it. In `tools/qwenomni_prune_inference.py`:

```python
video_embeds, _ = model.thinker.visual(pixel_values_videos..., grid_thw=video_grid_thw)
...
audio_outputs = model.thinker.audio_tower(input_features_processed, ...)
audio_embeds = audio_outputs.last_hidden_state
```

Both encoders run over **every** frame and the **whole** audio track, and only then
does `qwen_prune_inference_with_cache` prune. OmniPack's paper says the same thing
in words: the "modality-specific encoder-projector pipelines map them into the LLM
embedding space, producing N_v visual tokens and N_a audio tokens", and compression
follows.

So "pre-LLM" and "pre-encoder" are different positions, and the paper should say so
in one sentence before a reviewer conflates them. The `encode` stage in this runner
is timed separately from `prune_inference` precisely to put a number on the gap.

## Cost

One 7B checkpoint (already needed), plus CLIP ViT-L/14@336 (~1.7 GB). Two arms per
question over 18 questions. Budget about an hour on an RTX 4090, so roughly **$0.50
of community cloud time**, plus the flash-attn build.

## 1. Pod

- GPU: **RTX 4090 24 GB**, same card as the OmniZip run
- Template: PyTorch CUDA 12.4 image
- Volume: **120 GB**

## 2. Setup

OmniScope pins `transformers==4.52.3`, which is the **same pin OmniZip uses**, so
both head to heads can share one pod if you want to run them back to back.

```bash
cd /root
git clone https://github.com/MAC-AutoML/OmniScope.git
cd OmniScope
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install --no-build-isolation flash-attn==2.7.4.post1   # slow; start it early

cd /root && git clone https://github.com/thepreakerebi/gist.git
cd gist && pip install -e . --no-deps   # we only need gist.eval.profiling here

python -c "import torch; print('cuda', torch.cuda.is_available())"
python -c "from gist.eval.profiling import DeviceInfo; print(DeviceInfo.capture().describe())"
```

`init_model` hardcodes `attn_implementation="flash_attention_2"`, so flash-attn is
not optional. It is the longest step in the setup; kick it off before anything else.

## 3. Data

The same 18 questions and 6 videos as the other head to heads:

```bash
python /root/gist/results/fetch_videos.py /root/videomme_av6.json /content/vids
```

If you already ran the efficiency benchmark on this pod, the videos are cached and
this is a no-op.

## 4. Run

```bash
cd /root/OmniScope
OMNISCOPE_ROOT=/root/OmniScope \
QWEN_MODEL_PATH=Qwen/Qwen2.5-Omni-7B \
CLIP_MODEL_NAME=openai/clip-vit-large-patch14-336 \
python /root/gist/results/runpod-omniscope-h2h/omniscope_h2h.py /root/videomme_av6.json \
  2>&1 | tee /tmp/omniscope.log
```

If the encode stage OOMs, drop the budget rather than the videos:

```bash
OMNISCOPE_FRAMES=16 OMNISCOPE_TRIM=90 python ... # 90 s trim, as the OmniZip run used
```

Both arms always see identical inputs, so trimming keeps the comparison fair. Record
whatever you trimmed to.

## 5. Report

```bash
cd /root/gist
python -m gist.eval.efficiency \
  --measurements /tmp/omniscope_measurements.jsonl \
  --measured-baseline full --measured-stage prune_inference \
  --measured-markdown results/runpod-omniscope-h2h/measured.md \
  --measured-json results/runpod-omniscope-h2h/measured.json

# and the encoder cost that pruning never touches
python -m gist.eval.efficiency \
  --measurements /tmp/omniscope_measurements.jsonl \
  --measured-baseline omniscope --measured-stage encode \
  --measured-markdown results/runpod-omniscope-h2h/encode-cost.md
```

Copy back both JSONLs and commit them.

## What the result means, in each direction

Decide this before seeing the numbers.

- **If OmniScope scores higher than Gist**, the paper says so and the contribution
  is unchanged, because the claim was never "higher accuracy". Report the `encode`
  stage beside it: OmniScope pays the full dual-encoder forward on every frame and
  the whole audio track to get that score, and Gist does not.
- **If Gist matches or beats it**, resist the temptation to lead with that. n=18,
  a reduced frame budget and an unreproduced configuration cannot support an
  accuracy claim over a published system. It is a supporting observation, not a
  headline.
- **Either way**, the `encode` timing is the number that belongs in the abstract,
  because it is the one the architecture predicts and no other method can reduce.

## Caveats to carry into the paper

- **Not a reproduction.** Their evaluation uses frames_num=128 at
  max_pixels=768*28*28. This runs 32 at 128*28*28 to fit 24 GB and to match the
  OmniZip run. A lower absolute score is our budget, not their method.
- **`full` is OmniScope's fork with pruning off**, not stock Qwen2.5-Omni. That is
  the correct internal control and it is not interchangeable with the full-context
  numbers in `results/runpod-fp16/`.
- **Their defaults, not a tuned configuration.** visual 0.6 / audio 0.25 come from
  their own `eval_worldSense.py`. Both are environment-overridable; if you change
  them, say so.
- **n=18, near-chance absolute accuracy.** The same caveat every run in this repo
  carries. This run is most valuable for its cost measurements, not its accuracy.
