"""Scaled AV benchmark: Gist-selected vs full frames+audio through Qwen2.5-Omni-3B.

Multiple Video-MME videos, real gold MC answers. Omni loaded once. Per question,
answered two ways:
  full  = uniform 8 frames + 4 audio windows
  gist  = Gist-selected frames (CLIP) + audio windows (speech/sound dispatcher)
Reports accuracy + avg frames/windows per condition. Larger n than the smoke.
"""
import os, re, json, subprocess, time
os.environ.setdefault("LD_LIBRARY_PATH", "/usr/lib64-nvidia")
from pathlib import Path
import torch

import sys
QFILE = sys.argv[1] if len(sys.argv)>1 else "/content/videomme_av_all.json"
RESF = "/content/av7b_all_results.jsonl"
FRAME_BUDGET, AUDIO_BUDGET, NCAND, WIN = 8, 4, 64, 30

def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True)

rows = json.load(open(QFILE))
by_vid = {}
for r in rows:
    by_vid.setdefault(r["videoID"], []).append(r)
print(f"{len(rows)} questions across {len(by_vid)} videos", flush=True)

# --- Gist scorers + Omni (load once) ---
from gist.vision.clip import HuggingFaceClipFrameScorer
from gist.audio.clap import HuggingFaceClapAudioScorer
from gist.audio.whisper import FasterWhisperTranscriber
from gist.audio.dispatcher import SpeechSoundDispatcher
from gist.media.models import ExtractedFrame, AudioWindow
from gist.core.schemas import Candidate, CompressionRequest, Modality
from gist.core.presets import CompressionPreset
from gist.core.token_estimation import TokenEstimatorProfile
from gist.core.compressor import GistCompressor
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, BitsAndBytesConfig
from PIL import Image
import soundfile as sf

clip = HuggingFaceClipFrameScorer()
clap = HuggingFaceClapAudioScorer()
whisper = FasterWhisperTranscriber(model_size="tiny", device="cuda", compute_type="float16", beam_size=1)
disp = SpeechSoundDispatcher(clap=clap, transcriber=whisper)
print("loading Omni...", flush=True)
MID="Qwen/Qwen2.5-Omni-7B"
bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
proc = Qwen2_5OmniProcessor.from_pretrained(MID)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MID, torch_dtype=torch.float16, device_map="cuda", quantization_config=bnb, enable_audio_output=False).eval()
print("Omni ready", flush=True)

def uniform(xs, k):
    if len(xs) <= k: return xs
    step = (len(xs)-1)/(k-1) if k>1 else 0
    return [xs[round(i*step)] for i in range(k)]

def omni_answer(query, opts, imgs, auds):
    content = [{"type":"audio","audio":a} for a in auds] + [{"type":"image","image":i} for i in imgs]
    prompt = ("You are given frames and audio from a video. Using only what you see and hear, "
              "answer the multiple-choice question with ONLY the letter.\nQuestion: "+query+"\nOptions:\n"+"\n".join(opts))
    content.append({"type":"text","text":prompt})
    text = proc.apply_chat_template([{"role":"user","content":content}], add_generation_prompt=True, tokenize=False)
    image_in = [Image.open(i).convert("RGB") for i in imgs] or None
    audio_in = [sf.read(a)[0] for a in auds] or None
    inputs = proc(text=text, audio=audio_in, images=image_in, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False, return_audio=False)
    ans = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    m = re.search(r"[ABCD]", ans.upper())
    return m.group(0) if m else "?"

def prep_video(vid):
    fd = Path(f"/content/fr_{vid}"); ad = Path(f"/content/au_{vid}")
    cached = fd.exists() and len(list(fd.glob("f*.jpg")))>=NCAND and ad.exists() and len(list(ad.glob("a*.wav")))>0
    video = f"/content/vids/videomme-{vid}.mp4"
    if cached:
        # infer duration from cached audio-window count (WIN sec each) as a fallback
        nwav = len(list(ad.glob("a*.wav")))
        dur = float(nwav*WIN) if nwav else 0.0
        frames = [ExtractedFrame(index=i, timestamp_seconds=dur*i/NCAND, path=fd/f"f{i:03d}.jpg") for i in range(NCAND) if (fd/f"f{i:03d}.jpg").exists()]
        awins = [AudioWindow(index=j, start_seconds=j*WIN, duration_seconds=WIN, path=ad/f"a{j:03d}.wav") for j in range(nwav) if (ad/f"a{j:03d}.wav").exists()]
        return dur, frames, awins
    if not Path(video).exists():
        return None
    r = sh(f"ffprobe -v error -show_entries format=duration -of csv=p=0 {video}").stdout.strip()
    if not r: return None
    dur = float(r)
    fd.mkdir(exist_ok=True)
    for i in range(NCAND):
        p = fd/f"f{i:03d}.jpg"
        if not p.exists(): sh(f"ffmpeg -v error -y -ss {dur*i/NCAND:.2f} -i {video} -frames:v 1 -vf scale=448:-1 {p}")
    ad = Path(f"/content/au_{vid}"); ad.mkdir(exist_ok=True)
    nwin = int(dur//WIN)
    for j in range(nwin):
        p = ad/f"a{j:03d}.wav"
        if not p.exists(): sh(f"ffmpeg -v error -y -ss {j*WIN} -i {video} -t {WIN} -vn -ac 1 -ar 16000 {p}")
    frames = [ExtractedFrame(index=i, timestamp_seconds=dur*i/NCAND, path=fd/f"f{i:03d}.jpg") for i in range(NCAND) if (fd/f"f{i:03d}.jpg").exists()]
    awins = [AudioWindow(index=j, start_seconds=j*WIN, duration_seconds=WIN, path=ad/f"a{j:03d}.wav") for j in range(nwin) if (ad/f"a{j:03d}.wav").exists()]
    return dur, frames, awins

def gist_select(query, dur, frames, awins):
    vs = clip.score_frames(frames, query=query)
    as_ = disp.score_windows(awins, query)
    vc = [Candidate(id=f"v{f.index}", timestamp_seconds=f.timestamp_seconds, saliency_score=vs.get(f.path,0.0), asset_path=f.path) for f in frames]
    ac = [Candidate(id=f"a{w.index}", timestamp_seconds=w.start_seconds+WIN/2, saliency_score=as_.get(w.path,0.0), asset_path=w.path) for w in awins]
    resp = GistCompressor().compress(CompressionRequest(
        video_id="v", query=query, duration_seconds=dur, preset=CompressionPreset.BALANCED,
        adaptive_budget=True, decompose_query=True, token_estimator=TokenEstimatorProfile.GENERIC,
        task_aware_selection=True, visual_candidates=vc, audio_candidates=ac))
    imgs = [str(s.asset_path) for s in resp.selected if s.modality==Modality.VISUAL and s.asset_path][:FRAME_BUDGET]
    auds = [str(s.asset_path) for s in resp.selected if s.modality==Modality.AUDIO and s.asset_path][:AUDIO_BUDGET]
    return imgs, auds

full_ok=gist_ok=n=0; gist_frames_tot=gist_auds_tot=0
for vid, qs in by_vid.items():
    prepped = prep_video(vid)
    if prepped is None:
        print(f"SKIP {vid} (no video)", flush=True); continue
    dur, frames, awins = prepped
    full_imgs = [str(f.path) for f in uniform(frames, FRAME_BUDGET)]
    full_auds = [str(w.path) for w in uniform(awins, AUDIO_BUDGET)]
    for item in qs:
        q, opts, gold = item["question"], item["options"], str(item["answer"]).strip()[:1].upper()
        gi, ga = gist_select(q, dur, frames, awins)
        fa = omni_answer(q, opts, full_imgs, full_auds)
        ka = omni_answer(q, opts, gi, ga)
        n+=1; full_ok+=int(fa==gold); gist_ok+=int(ka==gold)
        open(RESF,"a").write(json.dumps({"qid":item["question_id"],"gold":gold,"full":fa,"gist":ka,"gf":len(gi),"ga":len(ga)})+"\n")
        gist_frames_tot+=len(gi); gist_auds_tot+=len(ga)
        print(f"{item['question_id']} gold={gold} full={fa}{'Y' if fa==gold else 'n'} gist={ka}{'Y' if ka==gold else 'n'} gf={len(gi)} ga={len(ga)}", flush=True)

print(f"RESULT7B_FULL: {full_ok}/{n} ({full_ok/n:.0%}) frames=8 auds=4", flush=True)
print(f"RESULT7B_GIST: {gist_ok}/{n} ({gist_ok/n:.0%}) avg_frames={gist_frames_tot/n:.1f} avg_auds={gist_auds_tot/n:.1f}", flush=True)
print("AV_BENCH_7B_DONE", flush=True)
