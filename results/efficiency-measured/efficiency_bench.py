"""Measured efficiency: wall clock and peak GPU memory per condition.

The capstone's claim is that selecting *before* the encoders run saves encoder
cost and peak memory. Every number supporting that so far has been analytic,
derived from the encoder configs in ``gist.eval.efficiency``. Analytic FLOPs
cannot speak to peak memory at all, and "measured" was the word the citation
audit had to walk back. This runner produces the real thing.

Mirrors ``results/runpod-omnizip-h2h/omnizip_h2h.py`` so the numbers stay
comparable: same 18 Video-MME AV questions, same six videos, same frame budget,
one model loaded once, each question answered under every condition.

Three conditions, chosen so the comparison isolates placement:

- ``dense``  — every candidate frame and every audio window, i.e. what a system
  with no compression at all would hand the encoders. On a 24 GB card this is
  *expected* to run out of memory on long videos. That is the result, not a bug:
  the harness records the OOM and keeps going.
- ``full``   — uniform 8 frames + 4 windows. The accuracy-matched baseline the
  existing results use, and the honest denominator for an efficiency ratio.
- ``gist``   — Gist's selection at the same total budget ceiling.

The selection cost is measured too, under stage ``select``. This is the number a
reviewer will reach for first: CLIP, CLAP and Whisper are not free, and if
scoring costs more than the encoding it saves then the claim collapses. Reporting
it unprompted is cheaper than being asked. Note that ``select`` is per question
while the candidate decode and transcription behind it are per video and cached,
so the amortised figure across a library is lower than one row here.

Usage on the pod:
    python efficiency_bench.py /root/videomme_av6.json
    GIST_SKIP_DENSE=1 python efficiency_bench.py ...   # cheap pass, no OOM arm
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("LD_LIBRARY_PATH", "/usr/lib64-nvidia")

import soundfile as sf
import torch
from PIL import Image
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from gist.audio.clap import HuggingFaceClapAudioScorer
from gist.audio.dispatcher import SpeechSoundDispatcher
from gist.audio.whisper import FasterWhisperTranscriber
from gist.core.compressor import GistCompressor
from gist.core.presets import CompressionPreset
from gist.core.schemas import Candidate, CompressionRequest, Modality
from gist.core.token_estimation import TokenEstimatorProfile
from gist.eval.profiling import DeviceInfo, MeasurementLog
from gist.media.models import AudioWindow, ExtractedFrame
from gist.vision.clip import HuggingFaceClipFrameScorer

QFILE = sys.argv[1] if len(sys.argv) > 1 else "/root/videomme_av6.json"
MEASF = os.environ.get("GIST_MEASUREMENTS", "/tmp/efficiency_measurements.jsonl")
RESF = os.environ.get("GIST_RESULTS", "/tmp/efficiency_answers.jsonl")
VIDDIR = os.environ.get("GIST_VIDEOS", "/content/vids")
SKIP_DENSE = os.environ.get("GIST_SKIP_DENSE") == "1"

FRAME_BUDGET, AUDIO_BUDGET, NCAND, WIN = 8, 4, 64, 30


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


rows = json.loads(Path(QFILE).read_text())
by_vid = {}
for row in rows:
    by_vid.setdefault(row["videoID"], []).append(row)
print(f"{len(rows)} questions across {len(by_vid)} videos", flush=True)

device = DeviceInfo.capture()
print(f"device: {device.describe()}", flush=True)
log = MeasurementLog(MEASF, device=device)

clip = HuggingFaceClipFrameScorer()
clap = HuggingFaceClapAudioScorer()
whisper = FasterWhisperTranscriber(
    model_size="tiny", device="cuda", compute_type="float16", beam_size=1
)
disp = SpeechSoundDispatcher(clap=clap, transcriber=whisper)

print("loading Omni...", flush=True)
MID = "Qwen/Qwen2.5-Omni-7B"
proc = Qwen2_5OmniProcessor.from_pretrained(MID)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MID, torch_dtype=torch.float16, device_map="auto", enable_audio_output=False
).eval()
print("Omni ready", flush=True)

# Everything below is compared against a card that already holds the weights.
# Recording the resting footprint once makes the per-condition activation deltas
# interpretable on their own, without the reader having to know the model size.
if torch.cuda.is_available():
    torch.cuda.synchronize()
    weights_gb = torch.cuda.memory_allocated() / 1e9
    print(f"resident after load: {weights_gb:.2f} GB", flush=True)


def uniform(xs, k):
    if len(xs) <= k:
        return xs
    step = (len(xs) - 1) / (k - 1) if k > 1 else 0
    return [xs[round(i * step)] for i in range(k)]


def omni_answer(query, opts, imgs, auds):
    content = [{"type": "audio", "audio": a} for a in auds]
    content += [{"type": "image", "image": i} for i in imgs]
    prompt = (
        "You are given frames and audio from a video. Using only what you see and hear, "
        "answer the multiple-choice question with ONLY the letter.\nQuestion: "
        + query
        + "\nOptions:\n"
        + "\n".join(opts)
    )
    content.append({"type": "text", "text": prompt})
    text = proc.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
    )
    image_in = [Image.open(i).convert("RGB") for i in imgs] or None
    audio_in = [sf.read(a)[0] for a in auds] or None
    inputs = proc(
        text=text, audio=audio_in, images=image_in, return_tensors="pt", padding=True
    ).to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False, return_audio=False)
    ans = proc.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]
    match = re.search(r"[ABCD]", ans.upper())
    return match.group(0) if match else "?"


def prep_video(vid):
    fd, ad = Path(f"/content/fr_{vid}"), Path(f"/content/au_{vid}")
    video = f"{VIDDIR}/videomme-{vid}.mp4"
    cached = (
        fd.exists()
        and len(list(fd.glob("f*.jpg"))) >= NCAND
        and ad.exists()
        and len(list(ad.glob("a*.wav"))) > 0
    )
    if cached:
        nwav = len(list(ad.glob("a*.wav")))
        dur = float(nwav * WIN)
    else:
        if not Path(video).exists():
            return None
        probe = sh(
            f"ffprobe -v error -show_entries format=duration -of csv=p=0 {video}"
        ).stdout.strip()
        if not probe:
            return None
        dur = float(probe)
        fd.mkdir(exist_ok=True)
        ad.mkdir(exist_ok=True)
        for i in range(NCAND):
            p = fd / f"f{i:03d}.jpg"
            if not p.exists():
                sh(
                    f"ffmpeg -v error -y -ss {dur * i / NCAND:.2f} -i {video} "
                    f"-frames:v 1 -vf scale=448:-1 {p}"
                )
        nwav = int(dur // WIN)
        for j in range(nwav):
            p = ad / f"a{j:03d}.wav"
            if not p.exists():
                sh(f"ffmpeg -v error -y -ss {j * WIN} -i {video} -t {WIN} -vn -ac 1 -ar 16000 {p}")

    frames = [
        ExtractedFrame(index=i, timestamp_seconds=dur * i / NCAND, path=fd / f"f{i:03d}.jpg")
        for i in range(NCAND)
        if (fd / f"f{i:03d}.jpg").exists()
    ]
    awins = [
        AudioWindow(index=j, start_seconds=j * WIN, duration_seconds=WIN, path=ad / f"a{j:03d}.wav")
        for j in range(nwav)
        if (ad / f"a{j:03d}.wav").exists()
    ]
    return dur, frames, awins


def gist_select(query, dur, frames, awins):
    vscores = clip.score_frames(frames, query=query)
    ascores = disp.score_windows(awins, query)
    vc = [
        Candidate(
            id=f"v{f.index}",
            timestamp_seconds=f.timestamp_seconds,
            saliency_score=vscores.get(f.path, 0.0),
            asset_path=f.path,
        )
        for f in frames
    ]
    ac = [
        Candidate(
            id=f"a{w.index}",
            timestamp_seconds=w.start_seconds + WIN / 2,
            saliency_score=ascores.get(w.path, 0.0),
            asset_path=w.path,
        )
        for w in awins
    ]
    resp = GistCompressor().compress(
        CompressionRequest(
            video_id="v",
            query=query,
            duration_seconds=dur,
            preset=CompressionPreset.BALANCED,
            adaptive_budget=True,
            decompose_query=True,
            token_estimator=TokenEstimatorProfile.GENERIC,
            task_aware_selection=True,
            visual_candidates=vc,
            audio_candidates=ac,
        )
    )
    imgs = [
        str(s.asset_path)
        for s in resp.selected
        if s.modality == Modality.VISUAL and s.asset_path
    ][:FRAME_BUDGET]
    auds = [
        str(s.asset_path)
        for s in resp.selected
        if s.modality == Modality.AUDIO and s.asset_path
    ][:AUDIO_BUDGET]
    return imgs, auds


correct = {"dense": 0, "full": 0, "gist": 0}
attempted = {"dense": 0, "full": 0, "gist": 0}

for vid, questions in by_vid.items():
    prepped = prep_video(vid)
    if prepped is None:
        print(f"SKIP {vid} (no video)", flush=True)
        continue
    dur, frames, awins = prepped

    arms = {
        "full": ([str(f.path) for f in uniform(frames, FRAME_BUDGET)],
                 [str(w.path) for w in uniform(awins, AUDIO_BUDGET)]),
    }
    if not SKIP_DENSE:
        arms["dense"] = ([str(f.path) for f in frames], [str(w.path) for w in awins])

    for item in questions:
        qid = item["question_id"]
        query, opts = item["question"], item["options"]
        gold = str(item["answer"]).strip()[:1].upper()

        try:
            with log.measure("select", condition="gist", qid=qid) as rec:
                imgs, auds = gist_select(query, dur, frames, awins)
                rec.extra["frames"] = len(imgs)
                rec.extra["audio_windows"] = len(auds)
        except Exception as exc:  # noqa: BLE001 - logged, then the run continues
            print(f"{qid} select: ERR {type(exc).__name__}: {exc}"[:200], flush=True)
            continue

        per_arm = dict(arms)
        per_arm["gist"] = (imgs, auds)

        for cond in ("dense", "full", "gist"):
            if cond not in per_arm:
                continue
            arm_imgs, arm_auds = per_arm[cond]
            answer = None
            # Hand every arm the same allocator state. Conditions always run in
            # the same order, so without this the dense arm's fragmentation (and
            # its OOM) would follow the other two around and inflate their
            # reserved figures.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                with log.measure("answer", condition=cond, qid=qid) as rec:
                    rec.extra["frames"] = len(arm_imgs)
                    rec.extra["audio_windows"] = len(arm_auds)
                    answer = omni_answer(query, opts, arm_imgs, arm_auds)
                    rec.extra["answer"] = answer
                    rec.extra["gold"] = gold
                    rec.extra["correct"] = answer == gold
            except Exception as exc:  # noqa: BLE001 - logged, then the run continues
                # The row is already on disk via the log's sink. Carrying on beats
                # losing the remaining questions of a paid pod hour to one of them.
                print(f"{qid} {cond}: ERR {type(exc).__name__}: {exc}"[:200], flush=True)
                continue
            if rec.oom:
                print(f"{qid} {cond}: OOM at {len(arm_imgs)}f/{len(arm_auds)}w", flush=True)
                continue
            attempted[cond] += 1
            correct[cond] += int(answer == gold)
            with open(RESF, "a") as fh:
                fh.write(json.dumps({
                    "qid": qid, "condition": cond, "gold": gold, "answer": answer,
                    "frames": len(arm_imgs), "audio_windows": len(arm_auds),
                    "wall_seconds": rec.wall_seconds, "activation_gb": rec.activation_gb,
                    "peak_reserved_gb": rec.peak_reserved_gb,
                }) + "\n")
            print(
                f"{qid} {cond}: {answer}{'Y' if answer == gold else 'n'} "
                f"{len(arm_imgs)}f/{len(arm_auds)}w {rec.wall_seconds:.2f}s "
                f"act={rec.activation_gb}GB",
                flush=True,
            )

for cond in ("dense", "full", "gist"):
    n = attempted[cond]
    if n:
        print(f"RESULT_{cond.upper()}: {correct[cond]}/{n} ({correct[cond] / n:.0%})", flush=True)
print(f"MEASUREMENTS: {MEASF}", flush=True)
print("EFFICIENCY_BENCH_DONE", flush=True)
