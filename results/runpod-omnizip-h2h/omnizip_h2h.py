"""SOTA head-to-head: OmniZip vs full-context through Qwen2.5-Omni-7B.

Drives OmniZip's *method* directly (its omnizip.modeling_qwen2_5_omni +
model.thinker.omnizip_config) — NOT lmms-eval, so no Video-MME dataset download.
Each of the same 18 Video-MME AV questions (videomme_av6.json) is answered twice
through the identical OmniZip model:

  full     = video+audio, omnizip_config = None            (no token compression)
  omnizip  = video+audio, omnizip_config = paper defaults   (rho_video 0.6 / rho_audio 0.3)

Reports MC accuracy + frames encoded per condition. Gist's numbers on these exact
18 questions were measured separately (av_bench_7b_fp16.py); the three-way write-up
compares OmniZip (dense frames + token pruning) against Gist (few frames selected
before encoding) at matched accuracy — the dual-encoder-FLOP-savings argument.

Env: torch 2.6, transformers 4.52.3 (OmniZip's pin), attn_implementation='sdpa'
(no flash-attn needed; compression is token pruning, not an attention kernel).
"""
import os, re, json, sys
os.environ.setdefault("PYTHONUNBUFFERED", "1")
import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor
from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration

QFILE = sys.argv[1] if len(sys.argv) > 1 else "/root/videomme_av6.json"
VIDDIR = sys.argv[2] if len(sys.argv) > 2 else "/content/vids"
MID = "Qwen/Qwen2.5-Omni-7B"
RESF = "/tmp/omnizip_h2h_results.jsonl"
MAXFRAMES = int(os.environ.get("MAXFRAMES", "32"))
MAXPIX = 128 * 28 * 28  # OmniZip paper VIDEO_MAX_PIXELS
OZ_CFG = {"rho_audio": 0.3, "rho_video": 0.6, "g": 3, "contextual_ratio": 0.05}

SYS = {"role": "system", "content": [{"type": "text", "text":
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."}]}

rows = json.load(open(QFILE))
print(f"{len(rows)} questions", flush=True)

proc = Qwen2_5OmniProcessor.from_pretrained(MID)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MID, torch_dtype=torch.float16, device_map="cuda:0", attn_implementation="sdpa").eval()
model.omnizip_config = None
print("model ready", flush=True)


def answer(video, q, opts, use_omnizip):
    model.thinker.omnizip_config = OZ_CFG if use_omnizip else None
    prompt = ("You are given a video with audio. Using ONLY what you see and hear, "
              "answer the multiple-choice question with ONLY the letter.\n"
              f"Question: {q}\nOptions:\n" + "\n".join(opts))
    conv = [SYS, {"role": "user", "content": [
        {"type": "video", "video": video, "max_pixels": MAXPIX, "nframes": MAXFRAMES},
        {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conv, use_audio_in_video=True)
    nframes = int(videos[0].shape[0]) if videos else 0
    if hasattr(model, "thinker"):
        model.thinker.nframes = nframes
    inputs = proc(text=text, audio=audios, images=images, videos=videos,
                  return_tensors="pt", padding=True, use_audio_in_video=True)
    inputs = inputs.to(model.device).to(model.dtype)
    with torch.no_grad():
        out = model.generate(**inputs, use_audio_in_video=True, return_audio=False,
                             do_sample=False, max_new_tokens=8)
    ans = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    m = re.search(r"[ABCD]", ans.upper())
    return (m.group(0) if m else "?"), nframes


acc = {"full": 0, "omnizip": 0}
n = 0
for r in rows:
    vid = r["videoID"]
    video = f"{VIDDIR}/videomme-{vid}.mp4"
    if not os.path.exists(video):
        print(f"SKIP {vid} (no video)", flush=True)
        continue
    q, opts = r["question"], r["options"]
    gold = str(r["answer"]).strip()[:1].upper()
    try:
        af, nf = answer(video, q, opts, False)
        ao, _ = answer(video, q, opts, True)
    except Exception as exc:
        print(f"ERR {r['question_id']}: {repr(exc)[:200]}", flush=True)
        continue
    n += 1
    acc["full"] += int(af == gold)
    acc["omnizip"] += int(ao == gold)
    with open(RESF, "a") as fh:
        fh.write(json.dumps({"qid": r["question_id"], "gold": gold,
                             "full": af, "omnizip": ao, "nframes": nf}) + "\n")
    print(f"{r['question_id']} gold={gold} full={af}{'Y' if af==gold else 'n'} "
          f"omnizip={ao}{'Y' if ao==gold else 'n'} nframes={nf}", flush=True)

if n:
    print(f"RESULT_H2H_FULL:    {acc['full']}/{n} ({acc['full']/n:.0%})  frames~{MAXFRAMES} rho_video=1.0", flush=True)
    print(f"RESULT_H2H_OMNIZIP: {acc['omnizip']}/{n} ({acc['omnizip']/n:.0%})  frames~{MAXFRAMES} rho_video=0.6 rho_audio=0.3", flush=True)
print("OMNIZIP_H2H_DONE", flush=True)
