"""Modality-knockout diagnostic: does each modality carry real signal?

For each question, Gist selects frames + audio, then Qwen2.5-Omni-3B answers
three ways from the SAME Gist evidence:
  full        = frames + audio
  frames_only = frames, audio removed
  audio_only  = audio, frames removed
If a modality is decorative, removing it wouldn't change accuracy.
"""
import os, re, json
os.environ.setdefault("LD_LIBRARY_PATH", "/usr/lib64-nvidia")
from pathlib import Path
import torch

QFILE = "/content/videomme_av6.json"
FRAME_BUDGET, AUDIO_BUDGET, WIN = 8, 4, 30

from gist.vision.clip import HuggingFaceClipFrameScorer
from gist.audio.clap import HuggingFaceClapAudioScorer
from gist.audio.whisper import FasterWhisperTranscriber
from gist.audio.dispatcher import SpeechSoundDispatcher
from gist.media.models import ExtractedFrame, AudioWindow
from gist.core.schemas import Candidate, CompressionRequest, Modality
from gist.core.presets import CompressionPreset
from gist.core.token_estimation import TokenEstimatorProfile
from gist.core.compressor import GistCompressor
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from PIL import Image
import soundfile as sf

clip = HuggingFaceClipFrameScorer()
clap = HuggingFaceClapAudioScorer()
whisper = FasterWhisperTranscriber(model_size="tiny", device="cuda", compute_type="float16", beam_size=1)
disp = SpeechSoundDispatcher(clap=clap, transcriber=whisper)
proc = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-3B", torch_dtype=torch.float16, device_map="cuda", enable_audio_output=False).eval()
print("ready", flush=True)

rows = json.load(open(QFILE)); by_vid = {}
for r in rows: by_vid.setdefault(r["videoID"], []).append(r)

def load_media(vid):
    fd = Path(f"/content/fr_{vid}"); ad = Path(f"/content/au_{vid}")
    fs = sorted(fd.glob("f*.jpg")); aw = sorted(ad.glob("a*.wav"))
    frames = [ExtractedFrame(index=i, timestamp_seconds=float(i), path=p) for i, p in enumerate(fs)]
    awins = [AudioWindow(index=j, start_seconds=j*WIN, duration_seconds=WIN, path=p) for j, p in enumerate(aw)]
    return frames, awins

def gist_select(query, frames, awins):
    vs = clip.score_frames(frames, query=query); as_ = disp.score_windows(awins, query)
    vc = [Candidate(id=f"v{f.index}", timestamp_seconds=f.timestamp_seconds, saliency_score=vs.get(f.path,0.0), asset_path=f.path) for f in frames]
    ac = [Candidate(id=f"a{w.index}", timestamp_seconds=w.start_seconds+WIN/2, saliency_score=as_.get(w.path,0.0), asset_path=w.path) for w in awins]
    resp = GistCompressor().compress(CompressionRequest(video_id="v", query=query, duration_seconds=float(len(frames)),
        preset=CompressionPreset.BALANCED, adaptive_budget=True, decompose_query=True,
        token_estimator=TokenEstimatorProfile.GENERIC, task_aware_selection=True, visual_candidates=vc, audio_candidates=ac))
    imgs = [str(s.asset_path) for s in resp.selected if s.modality==Modality.VISUAL and s.asset_path][:FRAME_BUDGET]
    auds = [str(s.asset_path) for s in resp.selected if s.modality==Modality.AUDIO and s.asset_path][:AUDIO_BUDGET]
    return imgs, auds

def answer(query, opts, imgs, auds):
    if not imgs and not auds: return "?"
    content = [{"type":"audio","audio":a} for a in auds] + [{"type":"image","image":i} for i in imgs]
    content.append({"type":"text","text":"Using only what you see and hear, answer with ONLY the letter.\nQuestion: "+query+"\nOptions:\n"+"\n".join(opts)})
    text = proc.apply_chat_template([{"role":"user","content":content}], add_generation_prompt=True, tokenize=False)
    inputs = proc(text=text, audio=[sf.read(a)[0] for a in auds] or None, images=[Image.open(i).convert("RGB") for i in imgs] or None, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False, return_audio=False)
    a = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    m = re.search(r"[ABCD]", a.upper()); return m.group(0) if m else "?"

acc = {"full":0, "frames_only":0, "audio_only":0}; n=0
for vid, qs in by_vid.items():
    frames, awins = load_media(vid)
    if not frames: print(f"SKIP {vid}", flush=True); continue
    for item in qs:
        q, opts, gold = item["question"], item["options"], str(item["answer"]).strip()[:1].upper()
        gi, ga = gist_select(q, frames, awins)
        r_full = answer(q, opts, gi, ga); r_v = answer(q, opts, gi, []); r_a = answer(q, opts, [], ga)
        acc["full"]+=int(r_full==gold); acc["frames_only"]+=int(r_v==gold); acc["audio_only"]+=int(r_a==gold); n+=1
        print(f"{item['question_id']} gold={gold} full={r_full} frames_only={r_v} audio_only={r_a}", flush=True)
for k in ["full","frames_only","audio_only"]:
    print(f"RESULT {k}: {acc[k]}/{n} ({acc[k]/n:.0%})", flush=True)
print("KNOCKOUT_DONE", flush=True)
