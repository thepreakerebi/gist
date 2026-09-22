"""Head to head against OmniScope, the closest published method to Gist.

OmniScope (arXiv 2607.23193) is the comparison worth making, for three reasons
that no other released method satisfies at once:

1. **It converged on Gist's idea.** Query-conditioned, per-modality relevance
   with separate budgets. It even uses CLIP ViT-L/14@336 scored against the
   question for the visual signal, which is Gist's first pillar. Independent
   arrival is evidence the principle is sound, and the honest framing is
   convergence rather than novelty.
2. **It is the split-budget shape Gist's own ablation only simulated.**
   `results/split-budget-ablation/` compared pooled against split *inside* Gist
   and was careful to say it did not compare against OmniScope. This does.
3. **Its code exists and runs.** Of the twelve audiovisual methods surveyed,
   only OmniZip, OmniSIFT and OmniScope have working public implementations.
   OmniPack, the current SOTA, published an empty repository - see RUN.md.

The variable under study is placement, and OmniScope's own code settles where it
sits better than its prose does. `get_cached_video_embeds` runs
`model.thinker.visual(...)` over every one of `frames_num` frames, and
`get_cached_audio_embeds` runs `model.thinker.audio_tower(...)` over the whole
track, *before* a single token is pruned. So the encoders do their full work
unconditionally. That is what Gist avoids, and it is why the `encode` stage below
is timed separately from `prune_inference`: the split isolates exactly the cost a
post-encoder method cannot escape.

Not a reproduction of OmniScope's published numbers. Their evaluation runs
frames_num=128 at max_pixels=768*28*28 on much larger cards. This runs 32 frames
at 128*28*28 to match the OmniZip head to head and to fit 24 GB. A lower score
here is our budget, not their method - the same caveat the OmniZip run carries.

Usage on the pod (from inside the cloned OmniScope repo):
    python omniscope_h2h.py /root/videomme_av6.json
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.environ.get("OMNISCOPE_ROOT", "/root/OmniScope"))

from tools.qwenomni_prune_inference import (  # noqa: E402
    ClipScorer,
    compute_audio_semantic_scores_batch,
    get_cached_audio_embeds,
    get_cached_video_embeds,
    init_model,
    qwen_prune_inference_with_cache,
)

from gist.eval.profiling import DeviceInfo, MeasurementLog  # noqa: E402

QFILE = sys.argv[1] if len(sys.argv) > 1 else "/root/videomme_av6.json"
VIDDIR = os.environ.get("GIST_VIDEOS", "/content/vids")
MEASF = os.environ.get("GIST_MEASUREMENTS", "/tmp/omniscope_measurements.jsonl")
RESF = os.environ.get("GIST_RESULTS", "/tmp/omniscope_h2h_results.jsonl")

QWEN_MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen2.5-Omni-7B")
CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL_NAME", "openai/clip-vit-large-patch14-336")

# Budget matched to the OmniZip head to head and to a 24 GB card, not to
# OmniScope's paper. Both arms see identical inputs, so the comparison is fair;
# it just runs smaller than they published.
FRAMES_NUM = int(os.environ.get("OMNISCOPE_FRAMES", "32"))
MAX_PIXELS = int(os.environ.get("OMNISCOPE_MAX_PIXELS", str(128 * 28 * 28)))
TRIM_SECONDS = float(os.environ.get("OMNISCOPE_TRIM", "0"))  # 0 = full length

# OmniScope's own defaults, from their eval_worldSense.py.
VISUAL_PRUNE_RATIO = float(os.environ.get("OMNISCOPE_VISUAL_RATIO", "0.6"))
AUDIO_PRUNE_RATIO = float(os.environ.get("OMNISCOPE_AUDIO_RATIO", "0.25"))
AUDIO_BOOST_SELF, AUDIO_BOOST_NEIGHBOR = 1.2, 1.5
AUDIO_BOOST_RADIUS, AUDIO_BOOST_DECAY, BOOST_PERCENTILE = 3, 0.5, 80


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def apply_neighbor_boost(scores, percentile=80, self_boost=1.2,
                         neighbor_boost=1.5, radius=3, decay=0.5):
    """Verbatim from OmniScope's eval_worldSense.py, so their method is theirs."""
    import numpy as np

    if len(scores) <= 1:
        return scores
    arr = np.array(scores, dtype=float)
    threshold = np.percentile(arr, percentile)
    high = np.where(arr >= threshold)[0]
    mult = np.ones_like(arr)
    for idx in high:
        mult[idx] = max(mult[idx], self_boost)
    for idx in high:
        for d in range(1, radius + 1):
            factor = 1.0 + (neighbor_boost - 1.0) * (decay**d)
            for neighbor in (idx - d, idx + d):
                if 0 <= neighbor < len(arr):
                    mult[neighbor] = max(mult[neighbor], factor)
    return (arr * mult).tolist()


def temporal_merge_scores(scores_per_question, merge_size=2):
    """Verbatim from OmniScope's eval_worldSense.py."""
    merged_all = []
    for scores in scores_per_question:
        merged = []
        for i in range(0, len(scores) - merge_size + 1, merge_size):
            chunk = scores[i : i + merge_size]
            merged.append(sum(chunk) / len(chunk))
        remainder = len(scores) % merge_size
        if remainder:
            merged.append(sum(scores[-remainder:]) / remainder)
        merged_all.append(merged)
    return merged_all


rows = json.loads(Path(QFILE).read_text())
by_vid = {}
for row in rows:
    by_vid.setdefault(row["videoID"], []).append(row)
print(f"{len(rows)} questions across {len(by_vid)} videos", flush=True)

device = DeviceInfo.capture()
print(f"device: {device.describe()}", flush=True)
log = MeasurementLog(MEASF, device=device)

init_model(QWEN_MODEL_PATH)
scorer = ClipScorer(model_name=CLIP_MODEL_NAME, device="cuda", cache_image_features=True)

if torch.cuda.is_available():
    torch.cuda.synchronize()
    print(f"resident after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

correct = {"full": 0, "omniscope": 0}
attempted = {"full": 0, "omniscope": 0}
agree = 0


def trimmed(video: str, vid: str) -> str:
    if not TRIM_SECONDS:
        return video
    out = f"/tmp/trim_{vid}.mp4"
    if not Path(out).exists():
        sh(f"ffmpeg -v error -y -i {video} -t {TRIM_SECONDS} -c copy {out}")
    return out


for vid, questions in by_vid.items():
    source = f"{VIDDIR}/videomme-{vid}.mp4"
    if not Path(source).exists():
        print(f"SKIP {vid} (no video)", flush=True)
        continue
    video_path = trimmed(source, vid)

    probe = sh(
        f"ffprobe -v error -show_entries format=duration -of csv=p=0 {video_path}"
    ).stdout.strip()
    video_time = float(probe) if probe else 0.0

    frames_in = FRAMES_NUM + (FRAMES_NUM % 2)
    query_list = [q["question"] + "\n" + " ".join(q["options"]) for q in questions]

    # Stage `encode`: the full dual-encoder forward that OmniScope pays before it
    # prunes anything. This is the cost Gist's placement avoids, so it is timed on
    # its own rather than folded into inference.
    try:
        with log.measure("encode", condition="omniscope", video=vid) as rec:
            embeds, grid_thw, audios, videos, video_kwargs = get_cached_video_embeds(
                video_path, frames_in, max_pixels=MAX_PIXELS
            )
            audio_embeds, audio_lengths = get_cached_audio_embeds(audios)
            rec.extra["frames"] = frames_in
            rec.extra["visual_tokens"] = int(embeds.shape[0])
            if audio_embeds is not None:
                rec.extra["audio_tokens"] = int(audio_embeds.shape[1])
    except Exception as exc:  # noqa: BLE001 - logged, then the video is skipped
        print(f"{vid} encode: ERR {type(exc).__name__}: {exc}"[:200], flush=True)
        continue
    if rec.oom:
        print(f"{vid} encode: OOM at {frames_in} frames", flush=True)
        continue

    # Stage `score`: OmniScope's CLIP and audio-semantic scoring. Timed because it
    # is their analogue of Gist's `select`, so the two are directly comparable.
    try:
        with log.measure("score", condition="omniscope", video=vid):
            raw_visual = scorer.compute_frame_scores_batch(
                query_list, video_path=video_path, cached_videos=videos,
                frames_num=frames_in, video_id=vid,
            )
            visual_scores = temporal_merge_scores(raw_visual, merge_size=2)
            t_qwen = grid_thw[0][0].item()
            for i, row_scores in enumerate(visual_scores):
                if len(row_scores) > t_qwen:
                    visual_scores[i] = row_scores[:t_qwen]
                elif len(row_scores) < t_qwen:
                    visual_scores[i] = row_scores + [1.0] * (t_qwen - len(row_scores))
            audio_scores = compute_audio_semantic_scores_batch(audio_embeds, query_list)
    except Exception as exc:  # noqa: BLE001 - logged, then the video is skipped
        print(f"{vid} score: ERR {type(exc).__name__}: {exc}"[:200], flush=True)
        del embeds, grid_thw, audio_embeds, audio_lengths
        torch.cuda.empty_cache()
        continue

    for q_num, item in enumerate(questions):
        qid = item["question_id"]
        gold = str(item["answer"]).strip()[:1].upper()
        prompt = (
            "Select the best answer to the following multiple-choice question based on "
            "the video. Respond with only the letter (A, B, C, or D) of the correct "
            "option. Question: " + item["question"] + "\n"
            + " ".join(item["options"]) + "\nThe best answer is:"
        )

        vis = [max(0, w * 100) ** 3 for w in visual_scores[q_num]]
        aud = apply_neighbor_boost(
            [max(0, w * 10) ** 2 for w in audio_scores[q_num]],
            percentile=BOOST_PERCENTILE, self_boost=AUDIO_BOOST_SELF,
            neighbor_boost=AUDIO_BOOST_NEIGHBOR, radius=AUDIO_BOOST_RADIUS,
            decay=AUDIO_BOOST_DECAY,
        )

        # `full` is OmniScope's own model with pruning switched off, not stock
        # Qwen. That is the correct internal control: it isolates the compression
        # from every other difference in their fork.
        arms = {
            "full": (1.0, 1.0),
            "omniscope": (VISUAL_PRUNE_RATIO, AUDIO_PRUNE_RATIO),
        }
        answers = {}
        for cond, (v_ratio, a_ratio) in arms.items():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                with log.measure("prune_inference", condition=cond, qid=qid) as rec:
                    rec.extra["visual_prune_ratio"] = v_ratio
                    rec.extra["audio_prune_ratio"] = a_ratio
                    out = qwen_prune_inference_with_cache(
                        video_path, prompt, vis, frames_in,
                        embeds, grid_thw, audios, videos, video_kwargs,
                        video_duration=video_time,
                        audio_semantic_scores=aud,
                        visual_prune_ratio=v_ratio,
                        audio_prune_ratio=a_ratio,
                        max_pixels=MAX_PIXELS,
                        cached_audio_embeds=audio_embeds,
                        cached_audio_output_lengths=audio_lengths,
                    )
                    match = re.search(r"[ABCD]", str(out).upper())
                    answer = match.group(0) if match else "?"
                    rec.extra["answer"] = answer
                    rec.extra["gold"] = gold
                    rec.extra["correct"] = answer == gold
            except Exception as exc:  # noqa: BLE001 - logged, then the run continues
                print(f"{qid} {cond}: ERR {type(exc).__name__}: {exc}"[:200], flush=True)
                continue
            if rec.oom:
                print(f"{qid} {cond}: OOM", flush=True)
                continue
            answers[cond] = answer
            attempted[cond] += 1
            correct[cond] += int(answer == gold)

        if len(answers) == 2:
            agree += int(answers["full"] == answers["omniscope"])
        with open(RESF, "a") as fh:
            fh.write(json.dumps({"qid": qid, "gold": gold, **answers}) + "\n")
        print(
            f"{qid} gold={gold} "
            + " ".join(f"{c}={a}{'Y' if a == gold else 'n'}" for c, a in answers.items()),
            flush=True,
        )

    del embeds, grid_thw, audio_embeds, audio_lengths
    torch.cuda.empty_cache()

for cond in ("full", "omniscope"):
    n = attempted[cond]
    if n:
        print(f"RESULT_{cond.upper()}: {correct[cond]}/{n} ({correct[cond] / n:.0%})", flush=True)
print(f"ANSWER_AGREEMENT: {agree}", flush=True)
print(f"MEASUREMENTS: {MEASF}", flush=True)
print("OMNISCOPE_H2H_DONE", flush=True)
