"""SOTA head-to-head: OmniSIFT vs uncompressed, on the same 18 Video-MME AV questions.

Mirrors results/runpod-omnizip-h2h/omnizip_h2h.py so the three-way comparison is
apples to apples. Drives OmniSIFT's *method* directly through its released
checkpoint, not lmms-eval, so no full Video-MME download is needed.

Each question is answered twice through the identical OmniSIFT model:

  full      = compression_config = None          (no token compression)
  omnisift  = compression_config = rho defaults  (its two-stage compression)

Three things to be honest about when reading the output:

1. **The uncompressed arm is not base Qwen.** OmniSIFT-7B is its own fine-tuned
   checkpoint, so "full" here means *that* model with compression switched off,
   not the stock Qwen2.5-Omni-7B the OmniZip run used. It is the right internal
   control for OmniSIFT, and it is not interchangeable with the earlier full-context
   number. Both are reported.

2. **This is not a reproduction of the paper's benchmark numbers.** OmniSIFT's own
   evaluation runs at max_frames=256 and max_pixels=256*28*28. We run at the frame
   budget the OmniZip head-to-head used, because the point is a matched comparison
   on our questions and our hardware. A 256-frame run will not fit on a 24 GB card;
   OmniZip already OOM'd well below that.

3. **The rho values are the repository's quick-start example**, not a value the
   paper states for its benchmark table. Override with RHO_VIDEO / RHO_AUDIO once
   the paper's evaluation config is confirmed.

Env: whatever the official Qwen2.5-Omni codebase pins, plus the omnisift package
from github.com/dingyue772/OmniSIFT.
"""

import json
import os
import re
import sys

os.environ.setdefault("PYTHONUNBUFFERED", "1")

import torch
from omnisift import Qwen2_5OmniForConditionalGeneration
from qwen_omni_utils import process_mm_info
from transformers import AutoProcessor

QFILE = sys.argv[1] if len(sys.argv) > 1 else "/root/videomme_av6.json"
VIDDIR = sys.argv[2] if len(sys.argv) > 2 else "/content/vids"
MID = os.environ.get("OMNISIFT_MODEL", "dingyue1011/OmniSIFT-7B")
RESF = os.environ.get("RESULTS", "/tmp/omnisift_h2h_results.jsonl")

# Matched to the OmniZip head-to-head so the two runs are comparable.
MAXFRAMES = int(os.environ.get("MAXFRAMES", "32"))
MAXPIX = int(os.environ.get("MAXPIX", str(128 * 28 * 28)))

# Fraction of tokens REMOVED, per the repository quick start.
SIFT_CFG = {
    "rho_audio": float(os.environ.get("RHO_AUDIO", "0.3")),
    "rho_video": float(os.environ.get("RHO_VIDEO", "0.7")),
}

SYS = {
    "role": "system",
    "content": [
        {
            "type": "text",
            "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
            "Group, capable of perceiving auditory and visual inputs, as well as "
            "generating text and speech.",
        }
    ],
}

with open(QFILE) as fh:
    rows = json.load(fh)
print(f"{len(rows)} questions | model {MID} | frames {MAXFRAMES} | cfg {SIFT_CFG}", flush=True)

proc = AutoProcessor.from_pretrained(MID)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MID, torch_dtype=torch.float16, device_map="cuda:0", attn_implementation="sdpa"
).eval()
model.thinker.compression_config = None
print("model ready", flush=True)


def answer(video, q, opts, compress):
    model.thinker.compression_config = SIFT_CFG if compress else None
    prompt = (
        "You are given a video with audio. Using ONLY what you see and hear, "
        "answer the multiple-choice question with ONLY the letter.\n"
        f"Question: {q}\nOptions:\n" + "\n".join(opts)
    )
    conv = [
        SYS,
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video, "max_pixels": MAXPIX, "nframes": MAXFRAMES},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conv, use_audio_in_video=True)
    nframes = int(videos[0].shape[0]) if videos else 0
    if hasattr(model, "thinker"):
        model.thinker.nframes = nframes
    inputs = proc(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=True,
    )
    inputs = inputs.to(model.device).to(model.dtype)
    with torch.no_grad():
        out = model.generate(
            **inputs, use_audio_in_video=True, return_audio=False,
            do_sample=False, max_new_tokens=8,
        )
    ans = proc.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]
    m = re.search(r"[ABCD]", ans.upper())
    return (m.group(0) if m else "?"), nframes


acc = {"full": 0, "omnisift": 0}
n = 0
peak_gb = 0.0

for r in rows:
    vid = r["videoID"]
    video = f"{VIDDIR}/videomme-{vid}.mp4"
    if not os.path.exists(video):
        print(f"SKIP {vid} (no video)", flush=True)
        continue

    q, opts = r["question"], r["options"]
    gold = str(r["answer"]).strip()[:1].upper()
    try:
        torch.cuda.reset_peak_memory_stats()
        af, nf = answer(video, q, opts, False)
        ao, _ = answer(video, q, opts, True)
        peak_gb = max(peak_gb, torch.cuda.max_memory_allocated() / 1e9)
    except Exception as exc:  # noqa: BLE001 - recorded, then skipped
        print(f"ERR {r['question_id']}: {repr(exc)[:200]}", flush=True)
        continue

    n += 1
    acc["full"] += int(af == gold)
    acc["omnisift"] += int(ao == gold)
    with open(RESF, "a") as fh:
        fh.write(
            json.dumps(
                {
                    "qid": r["question_id"], "gold": gold,
                    "full": af, "omnisift": ao, "nframes": nf,
                }
            )
            + "\n"
        )
    print(
        f"{r['question_id']} gold={gold} full={af}{'Y' if af == gold else 'n'} "
        f"omnisift={ao}{'Y' if ao == gold else 'n'} nframes={nf}",
        flush=True,
    )

if n:
    print(
        f"RESULT_H2H_FULL:     {acc['full']}/{n} ({acc['full'] / n:.0%})  "
        f"frames~{MAXFRAMES}  compression off",
        flush=True,
    )
    print(
        f"RESULT_H2H_OMNISIFT: {acc['omnisift']}/{n} ({acc['omnisift'] / n:.0%})  "
        f"frames~{MAXFRAMES}  rho_video={SIFT_CFG['rho_video']} "
        f"rho_audio={SIFT_CFG['rho_audio']}",
        flush=True,
    )
    print(f"RESULT_PEAK_GB:      {peak_gb:.1f}", flush=True)
print("OMNISIFT_H2H_DONE", flush=True)
