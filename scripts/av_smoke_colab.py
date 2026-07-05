"""AV smoke: Gist-selected vs full frames+audio through a real Omni-LLM.

One Video-MME video, its gold MC questions, answered two ways by Qwen2.5-Omni-3B:
  full  = uniform 8 frames + 4 audio windows across the whole video
  gist  = Gist-selected 8 frames (CLIP) + 4 audio windows (speech/sound dispatcher)
Scores MC letter vs gold. Small n — a first real Omni datapoint, labelled a smoke.
"""
import os, re, json, subprocess, time
os.environ.setdefault("LD_LIBRARY_PATH", "/usr/lib64-nvidia")
from pathlib import Path
import torch

VID = "eZ5lFBPpRkM"
VIDEO = f"/content/{VID}.mp4"
QUESTIONS = [
    {"q": "What is the relationship of the woman standing next to the main character?",
     "opts": ["A. His wife.", "B. His daughter.", "C. His sister.", "D. His friend."], "gold": "A"},
    {"q": "When the people in the video saw the turtles for the first time, how many days into the trip was it?",
     "opts": ["A. Day 2.", "B. Day 3.", "C. Day 4.", "D. Day 5."], "gold": "B"},
    {"q": "What does the protagonist of the video record?",
     "opts": ["A. Travel logs with friends.", "B. Bali attractions review.",
              "C. A Bali travel vlog.", "D. A documentary."], "gold": "C"},
]
FRAME_BUDGET, AUDIO_BUDGET = 8, 4

def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True)

# 1. download
if not Path(VIDEO).exists():
    print("downloading video...", flush=True)
    sh(f"yt-dlp -q -f 'bv*[height<=480]+ba/b[height<=480]/b' --merge-output-format mp4 -o {VIDEO} https://www.youtube.com/watch?v={VID}")
dur = float(sh(f"ffprobe -v error -show_entries format=duration -of csv=p=0 {VIDEO}").stdout.strip())
print(f"duration={dur:.0f}s", flush=True)

# 2. candidate frames (64 uniform) + audio windows (30s)
FD = Path("/content/frames"); FD.mkdir(exist_ok=True)
NCAND = 64
for i in range(NCAND):
    t = dur * i / NCAND
    p = FD / f"f{i:03d}.jpg"
    if not p.exists():
        sh(f"ffmpeg -v error -y -ss {t:.2f} -i {VIDEO} -frames:v 1 -vf scale=448:-1 {p}")
AD = Path("/content/aud"); AD.mkdir(exist_ok=True)
WIN = 30; nwin = int(dur // WIN)
for j in range(nwin):
    p = AD / f"a{j:03d}.wav"
    if not p.exists():
        sh(f"ffmpeg -v error -y -ss {j*WIN} -i {VIDEO} -t {WIN} -vn -ac 1 -ar 16000 {p}")
print(f"candidates: {NCAND} frames, {nwin} audio windows", flush=True)

# 3. Gist scorers (real): CLIP frames + speech/sound dispatcher audio
from gist.vision.clip import HuggingFaceClipFrameScorer
from gist.audio.clap import HuggingFaceClapAudioScorer
from gist.audio.whisper import FasterWhisperTranscriber
from gist.audio.dispatcher import SpeechSoundDispatcher
from gist.media.models import ExtractedFrame, AudioWindow
from gist.core.schemas import Candidate, CompressionRequest
from gist.core.presets import CompressionPreset
from gist.core.token_estimation import TokenEstimatorProfile
from gist.core.compressor import GistCompressor
from gist.core.schemas import Modality

clip = HuggingFaceClipFrameScorer()
clap = HuggingFaceClapAudioScorer()
whisper = FasterWhisperTranscriber(model_size="tiny", device="cuda", compute_type="float16", beam_size=1)
disp = SpeechSoundDispatcher(clap=clap, transcriber=whisper)

frames = [ExtractedFrame(index=i, timestamp_seconds=dur*i/NCAND, path=FD/f"f{i:03d}.jpg")
          for i in range(NCAND) if (FD/f"f{i:03d}.jpg").exists()]
awins = [AudioWindow(index=j, start_seconds=j*WIN, duration_seconds=WIN, path=AD/f"a{j:03d}.wav")
         for j in range(nwin) if (AD/f"a{j:03d}.wav").exists()]

def uniform(paths_ts, k):
    if len(paths_ts) <= k: return paths_ts
    step = (len(paths_ts)-1)/(k-1) if k>1 else 0
    return [paths_ts[round(i*step)] for i in range(k)]

def gist_select(query):
    vscores = clip.score_frames(frames, query=query)
    ascores = disp.score_windows(awins, query)
    vc = [Candidate(id=f"v{f.index}", timestamp_seconds=f.timestamp_seconds,
                    saliency_score=vscores.get(f.path,0.0), asset_path=f.path) for f in frames]
    ac = [Candidate(id=f"a{w.index}", timestamp_seconds=w.start_seconds+WIN/2,
                    saliency_score=ascores.get(w.path,0.0), asset_path=w.path) for w in awins]
    resp = GistCompressor().compress(CompressionRequest(
        video_id=VID, query=query, duration_seconds=dur, preset=CompressionPreset.BALANCED,
        adaptive_budget=True, decompose_query=True, token_estimator=TokenEstimatorProfile.GENERIC,
        task_aware_selection=True, visual_candidates=vc, audio_candidates=ac))
    imgs = [str(s.asset_path) for s in resp.selected if s.modality==Modality.VISUAL and s.asset_path][:FRAME_BUDGET]
    auds = [str(s.asset_path) for s in resp.selected if s.modality==Modality.AUDIO and s.asset_path][:AUDIO_BUDGET]
    return imgs, auds

# 4. Omni model (load once)
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from PIL import Image
import soundfile as sf
print("loading Omni...", flush=True)
proc = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-3B", torch_dtype=torch.float16, device_map="cuda", enable_audio_output=False).eval()
print("Omni ready", flush=True)

def omni_answer(query, opts, imgs, auds):
    content = [{"type":"audio","audio":a} for a in auds] + [{"type":"image","image":i} for i in imgs]
    prompt = ("You are given frames and audio from a video. Using only what you see and hear, "
              "answer the multiple-choice question with ONLY the letter.\nQuestion: "+query+"\nOptions:\n"+"\n".join(opts))
    content.append({"type":"text","text":prompt})
    conv=[{"role":"user","content":content}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    image_in = [Image.open(i).convert("RGB") for i in imgs] or None
    audio_in = [sf.read(a)[0] for a in auds] or None
    inputs = proc(text=text, audio=audio_in, images=image_in, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False, return_audio=False)
    ans = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    m = re.search(r"[ABCD]", ans.upper())
    return m.group(0) if m else "?"

# 5. run
full_frames = uniform(frames, FRAME_BUDGET)
full_imgs = [str(f.path) for f in full_frames]
full_auds = [str(w.path) for w in uniform(awins, AUDIO_BUDGET)]

res = {"full": [], "gist": []}
for qi, item in enumerate(QUESTIONS):
    gi, ga = gist_select(item["q"])
    fa = omni_answer(item["q"], item["opts"], full_imgs, full_auds)
    ka = omni_answer(item["q"], item["opts"], gi, ga)
    res["full"].append(fa == item["gold"]); res["gist"].append(ka == item["gold"])
    print(f"Q{qi+1} gold={item['gold']} full={fa}({'Y' if fa==item['gold'] else 'n'}) "
          f"gist={ka}({'Y' if ka==item['gold'] else 'n'}) gist_frames={len(gi)} gist_auds={len(ga)}", flush=True)

def acc(x): return sum(x)/len(x)
print(f"RESULT full: {sum(res['full'])}/{len(res['full'])} ({acc(res['full']):.0%}) frames=8 auds=4", flush=True)
print(f"RESULT gist: {sum(res['gist'])}/{len(res['gist'])} ({acc(res['gist']):.0%}) frames<=8 auds<=4", flush=True)
print("AV_SMOKE_DONE", flush=True)
