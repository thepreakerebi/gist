"""Head to head: uniform baseline against our reimplementation of OmniPack.

**Not yet executed. Written on CPU, never run.** Every other runner under
`results/` was written against a live pod; this one could not be, because
OmniPack released no code to borrow a working harness from. Treat the first pod
session as a debugging session, not a measurement session, and read the
"First run" section of RUN.md before starting it.

**This is not OmniPack.** It is our implementation of the published description,
under four documented assumptions. Label every number accordingly.

What is measured, and why the stages are split:

  encode      the full dual-encoder forward. OmniPack pays this unconditionally
              because it compresses *after* the encoders; Gist does not. Timing
              it separately is the entire point of the comparison, and it is the
              cost their own FLOPs accounting excludes.
  select      our stage 1 (dual selection plus merging). Their analogue of Gist's
              `select`, so the two are directly comparable.
  answer      generation, with stage 2 applied inside the LLM if enabled.

Arms:

  full        the same model with compression switched off. The internal control:
              it isolates compression from every other difference in this harness.
  omnipack    stage 1 at the swept retention ratio, plus stage 2 at 50% if
              enabled.

Usage on the pod:

    python results/omnipack-replication/omnipack_h2h.py /root/videomme_av6.json

    OMNIPACK_RATIOS=0.25,0.20,0.15,0.10   sweep, per the paper's four settings
    OMNIPACK_STAGE2=0                     stage 1 only, the fallback arm
    OMNIPACK_FRAMES=32                    matched to the OmniZip/OmniScope runs
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from omnipack import (  # noqa: E402
    INNER_LAYER_QWEN_7B,
    INNER_RETENTION,
    stage1_compress,
    stage2_select,
)

from gist.eval.profiling import DeviceInfo, MeasurementLog  # noqa: E402

QFILE = sys.argv[1] if len(sys.argv) > 1 else "/root/videomme_av6.json"
VIDDIR = os.environ.get("GIST_VIDEOS", "/content/vids")
MEASF = os.environ.get("GIST_MEASUREMENTS", "/tmp/omnipack_measurements.jsonl")
RESF = os.environ.get("GIST_RESULTS", "/tmp/omnipack_h2h_results.jsonl")

QWEN_MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen2.5-Omni-7B")
FRAMES_NUM = int(os.environ.get("OMNIPACK_FRAMES", "32"))
MAX_PIXELS = int(os.environ.get("OMNIPACK_MAX_PIXELS", str(128 * 28 * 28)))
RATIOS = [float(r) for r in os.environ.get("OMNIPACK_RATIOS", "0.25").split(",")]
USE_STAGE2 = os.environ.get("OMNIPACK_STAGE2", "1") == "1"
INNER_LAYER = int(os.environ.get("OMNIPACK_INNER_LAYER", str(INNER_LAYER_QWEN_7B)))


def sh(cmd: str):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def die(message: str) -> None:
    """Fail loudly. A silent fallback here would corrupt the comparison."""
    print(f"FATAL: {message}", flush=True)
    raise SystemExit(2)


# ---------------------------------------------------------------------------
# Model access
#
# These entry points are not guessed. OmniScope's released
# `tools/qwenomni_prune_inference.py` calls `model.thinker.visual(...)` and
# `model.thinker.audio_tower(...)` on the same checkpoint, which is also the
# evidence that their method runs the encoders over everything before pruning.
# Confirm the exact signatures against the transformers version on the pod before
# trusting a measurement; they move between releases.
# ---------------------------------------------------------------------------


def load_model():
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    processor = Qwen2_5OmniProcessor.from_pretrained(QWEN_MODEL_PATH)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        QWEN_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cuda",
        enable_audio_output=False,
    )
    model.eval()
    return model, processor


def decoder_layers(model):
    thinker = getattr(model, "thinker", None)
    if thinker is None:
        die("model has no .thinker; this harness targets Qwen2.5-Omni")
    for path in ("model.layers", "language_model.model.layers", "model.model.layers"):
        node = thinker
        try:
            for part in path.split("."):
                node = getattr(node, part)
            return node
        except AttributeError:
            continue
    die("could not locate the decoder layer list on model.thinker")


class Stage2Hook:
    """Applies stage 2 once, during prefill, at the insertion layer.

    Pruning mid-forward changes the sequence length, which invalidates the
    attention mask, the position ids and the KV cache for every later layer. This
    hook therefore fires only on the prefill pass (sequence length > 1) and lets
    generation proceed on the compressed sequence. **That behaviour is the part of
    this file most likely to be wrong**; validate it on the pod by checking that
    the `full` arm reproduces stock Qwen answers before trusting the compressed
    arm, and fall back to OMNIPACK_STAGE2=0 if it does not.
    """

    def __init__(self, media_span: tuple[int, int], text_span: tuple[int, int]):
        self.media_span = media_span
        self.text_span = text_span
        self.fired = False
        self.kept_tokens: int | None = None

    def __call__(self, module, args, output):
        if self.fired:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.dim() != 3 or hidden.shape[1] <= 1:
            return output
        self.fired = True

        batch = hidden[0]
        m0, m1 = self.media_span
        t0, t1 = self.text_span
        media = batch[m0:m1]
        text = batch[t0:t1]
        if media.shape[0] == 0 or text.shape[0] == 0:
            return output

        kept, index = stage2_select(
            media.float(),
            query_states=text.float(),
            other_modality_states=None,
            retention=INNER_RETENTION,
        )
        self.kept_tokens = int(kept.shape[0])
        rebuilt = torch.cat([batch[:m0], kept.to(batch.dtype), batch[m1:]], dim=0)
        rebuilt = rebuilt.unsqueeze(0)
        if isinstance(output, tuple):
            return (rebuilt,) + tuple(output[1:])
        return rebuilt


def main() -> int:
    if not torch.cuda.is_available():
        die("no CUDA device. This runner is for the pod; nothing here runs on CPU.")

    rows = json.loads(Path(QFILE).read_text())
    by_vid: dict[str, list] = {}
    for row in rows:
        by_vid.setdefault(row["videoID"], []).append(row)
    print(f"{len(rows)} questions across {len(by_vid)} videos", flush=True)
    print(f"ratios={RATIOS} stage2={USE_STAGE2} layer={INNER_LAYER}", flush=True)

    device = DeviceInfo.capture()
    print(f"device: {device.describe()}", flush=True)
    log = MeasurementLog(MEASF, device=device)

    model, processor = load_model()
    layers = decoder_layers(model)
    if len(layers) <= INNER_LAYER:
        die(f"insertion layer {INNER_LAYER} beyond {len(layers)} decoder layers")
    torch.cuda.synchronize()
    print(f"resident after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

    conditions = ["full"] + [f"omnipack_r{r:g}" for r in RATIOS]
    correct = {c: 0 for c in conditions}
    attempted = {c: 0 for c in conditions}

    for vid, questions in by_vid.items():
        source = f"{VIDDIR}/videomme-{vid}.mp4"
        if not Path(source).exists():
            print(f"SKIP {vid} (no video)", flush=True)
            continue

        for item in questions:
            qid = item["question_id"]
            gold = str(item["answer"]).strip()[:1].upper()
            prompt = (
                "Select the best answer to the following multiple-choice question "
                "based on the video. Respond with only the letter (A, B, C, or D) of "
                "the correct option. Question: " + item["question"] + "\n"
                + " ".join(item["options"]) + "\nThe best answer is:"
            )

            answers: dict[str, str] = {}
            for condition in conditions:
                ratio = None
                if condition != "full":
                    ratio = float(condition.split("_r")[1])
                torch.cuda.empty_cache()

                try:
                    # Stage `encode`: paid in full by both arms, because a
                    # post-encoder method cannot avoid it. This is the number the
                    # comparison exists to expose.
                    with log.measure("encode", condition=condition, video=vid) as rec:
                        inputs = build_inputs(processor, source, prompt)
                        visual, audio, spans = encode(model, inputs)
                        rec.extra["frames"] = FRAMES_NUM
                        rec.extra["visual_tokens"] = 0 if visual is None else int(visual.shape[0])
                        rec.extra["audio_tokens"] = 0 if audio is None else int(audio.shape[0])
                    if rec.oom:
                        print(f"{qid} {condition}: OOM in encode", flush=True)
                        continue

                    if ratio is not None:
                        with log.measure("select", condition=condition, qid=qid) as rec:
                            if visual is not None:
                                out = stage1_compress(
                                    visual.float(), None, modality="visual",
                                    retention=ratio,
                                    frames=spans["frames"], patches=spans["patches"],
                                )
                                visual = out.embeds.to(visual.dtype)
                                rec.extra["visual_before"] = out.n_before
                                rec.extra["visual_after"] = out.n_after
                            if audio is not None:
                                out = stage1_compress(
                                    audio.float(), None, modality="audio", retention=ratio,
                                )
                                audio = out.embeds.to(audio.dtype)
                                rec.extra["audio_before"] = out.n_before
                                rec.extra["audio_after"] = out.n_after

                    hook = None
                    handle = None
                    if ratio is not None and USE_STAGE2:
                        hook = Stage2Hook(spans["media_span"], spans["text_span"])
                        handle = layers[INNER_LAYER].register_forward_hook(hook)

                    with log.measure("answer", condition=condition, qid=qid) as rec:
                        text = generate(model, processor, inputs, visual, audio)
                        match = re.search(r"[ABCD]", text.upper())
                        answer = match.group(0) if match else "?"
                        rec.extra["answer"] = answer
                        rec.extra["gold"] = gold
                        rec.extra["correct"] = answer == gold
                        if hook is not None:
                            rec.extra["stage2_kept"] = hook.kept_tokens
                    if handle is not None:
                        handle.remove()
                    if rec.oom:
                        print(f"{qid} {condition}: OOM in answer", flush=True)
                        continue

                except NotImplementedError as exc:
                    die(str(exc))
                except Exception as exc:  # noqa: BLE001 - logged, run continues
                    print(f"{qid} {condition}: ERR {type(exc).__name__}: {exc}"[:200], flush=True)
                    continue

                answers[condition] = answer
                attempted[condition] += 1
                correct[condition] += int(answer == gold)

            with open(RESF, "a") as fh:
                fh.write(json.dumps({"qid": qid, "gold": gold, **answers}) + "\n")
            print(
                f"{qid} gold={gold} "
                + " ".join(f"{c}={a}{'Y' if a == gold else 'n'}" for c, a in answers.items()),
                flush=True,
            )

    for condition in conditions:
        n = attempted[condition]
        if n:
            print(
                f"RESULT_{condition.upper()}: {correct[condition]}/{n} "
                f"({correct[condition] / n:.0%})",
                flush=True,
            )
    print(f"MEASUREMENTS: {MEASF}", flush=True)
    print("OMNIPACK_H2H_DONE", flush=True)
    return 0


# ---------------------------------------------------------------------------
# The three model-shaped functions below are deliberately unimplemented.
#
# Their correct form depends on the transformers version installed on the pod:
# the processor's video/audio keyword names, whether the visual tower takes
# `grid_thw` positionally, and where the media tokens land in the prompt all
# moved between releases. Guessing them here would produce a file that looks
# finished and fails on the pod with an error three layers from the cause.
#
# Fill these in as the first task of the pod session, against the installed
# version, using OmniScope's `tools/qwenomni_prune_inference.py` as the worked
# example -- it does exactly this for the same checkpoint. RUN.md has the
# checklist.
# ---------------------------------------------------------------------------


def build_inputs(processor, video_path: str, prompt: str):
    raise NotImplementedError(
        "build_inputs: construct processor inputs for one video plus prompt at "
        f"{FRAMES_NUM} frames, max_pixels={MAX_PIXELS}. See RUN.md, First run."
    )


def encode(model, inputs):
    """Run both towers and return (visual_tokens, audio_tokens, spans).

    spans must carry: frames, patches, media_span and text_span as (start, end)
    token offsets into the prefill sequence, which stage 2 needs.
    """
    raise NotImplementedError(
        "encode: call model.thinker.visual(...) and model.thinker.audio_tower(...), "
        "then locate the media and text spans. See RUN.md, First run."
    )


def generate(model, processor, inputs, visual, audio) -> str:
    raise NotImplementedError(
        "generate: splice the compressed embeddings into inputs_embeds and run "
        "generate(). See RUN.md, First run."
    )


if __name__ == "__main__":
    raise SystemExit(main())
