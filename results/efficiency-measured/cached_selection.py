"""Selection cost with per-video caching, against the live path the GPU run timed.

`efficiency_bench.py` timed Gist's selection through the live scorers
(`score_frames`, the dispatcher's `score_windows`), which re-encode every frame
and every audio window for every question; only Whisper transcripts are cached.
That reproduces the selections behind the accuracy results, but it is not how
Gist is meant to run. The query-independent towers (`embed_frames`,
`embed_windows`, transcript embeddings) can run once per video, after which a
question costs a few text embeddings and dot products. The web library already
works that way.

This script measures both paths on the same videos and questions:

  ingest     once per video: frame and window extraction excluded, then the
             CLIP image tower, the CLAP audio tower, Whisper on speech windows
             and MiniLM on their transcripts
  cached     per question: the text embeddings, the dot products, the speech
             routing and Gist's compressor
  live       per question: exactly what `efficiency_bench.gist_select` does

and checks that the cached path reproduces the live scores and the live
selection, so the cheaper number describes the same method, not a different one.

Run on whatever device the scorers pick (here, a MacBook); the answer stage is
not involved, so no GPU is needed. Timings on a laptop CPU are an upper bound on
what the RTX 4090 run would show.

    python results/efficiency-measured/cached_selection.py .gist/benchmark/videomme_av6.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from gist.audio.clap import HuggingFaceClapAudioScorer
from gist.audio.dispatcher import _AMBIENT_PROBE, _SPEECH_PROBE, SpeechSoundDispatcher
from gist.audio.whisper import FasterWhisperTranscriber
from gist.core.compressor import GistCompressor
from gist.core.presets import CompressionPreset
from gist.core.schemas import Candidate, CompressionRequest, Modality
from gist.core.token_estimation import TokenEstimatorProfile
from gist.media.models import AudioWindow, ExtractedFrame
from gist.vision.clip import HuggingFaceClipFrameScorer

QFILE = Path(sys.argv[1] if len(sys.argv) > 1 else ".gist/benchmark/videomme_av6.json")
VIDDIR = Path(os.environ.get("GIST_VIDEOS", ".gist/vids"))
WORK = Path(os.environ.get("GIST_WORK", ".gist/cached-selection-work"))
OUT = Path(os.environ.get("GIST_OUT", "results/efficiency-measured/cached_selection.jsonl"))
NCAND, WIN = 64, 30  # as efficiency_bench.py
FRAME_BUDGET, AUDIO_BUDGET = 8, 4


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def prep_video(vid: str):
    """Frames and windows exactly as efficiency_bench.prep_video makes them."""
    video = VIDDIR / f"videomme-{vid}.mp4"
    if not video.exists():
        return None
    dur = float(sh(f"ffprobe -v error -show_entries format=duration -of csv=p=0 {video}"))
    fd, ad = WORK / f"fr_{vid}", WORK / f"au_{vid}"
    fd.mkdir(parents=True, exist_ok=True)
    ad.mkdir(parents=True, exist_ok=True)
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
    windows = [
        AudioWindow(index=j, start_seconds=j * WIN, duration_seconds=WIN, path=ad / f"a{j:03d}.wav")
        for j in range(nwav)
        if (ad / f"a{j:03d}.wav").exists()
    ]
    return dur, frames, windows


def raw_clap_text(clap: HuggingFaceClapAudioScorer, prompts: list[str]) -> np.ndarray:
    """CLAP text embeddings for raw prompts, as `score_windows_against` uses them.

    `embed_text` prepends "the sound of:", which is right for the query but not
    for the dispatcher's speech and ambient probes.
    """
    from gist.audio.clap import _feature_tensor, _normalize

    clap._load()
    inputs = clap._processor(text=prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(clap._model.device) for k, v in inputs.items()}
    with clap._torch.no_grad():
        out = _normalize(_feature_tensor(clap._model.get_text_features(**inputs)), clap._torch)
    return out.cpu().numpy()


def compress(query, dur, frames, windows, vscores, ascores):
    """Gist's compressor on given scores; identical to efficiency_bench.gist_select."""
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
        for w in windows
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
        str(s.asset_path) for s in resp.selected if s.modality == Modality.VISUAL and s.asset_path
    ]
    auds = [
        str(s.asset_path) for s in resp.selected if s.modality == Modality.AUDIO and s.asset_path
    ]
    return imgs[:FRAME_BUDGET], auds[:AUDIO_BUDGET]


def main() -> int:
    rows = json.loads(QFILE.read_text())
    by_vid: dict[str, list] = {}
    for r in rows:
        by_vid.setdefault(r["videoID"], []).append(r)

    clip = HuggingFaceClipFrameScorer()
    clap = HuggingFaceClapAudioScorer()
    whisper = FasterWhisperTranscriber(
        model_size="tiny", device="cpu", compute_type="int8", beam_size=1
    )
    disp = SpeechSoundDispatcher(clap=clap, transcriber=whisper)
    probes = raw_clap_text(clap, [_SPEECH_PROBE, _AMBIENT_PROBE])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("")

    for vid, questions in by_vid.items():
        prepped = prep_video(vid)
        if prepped is None:
            print(f"SKIP {vid}", flush=True)
            continue
        dur, frames, windows = prepped

        # Once per video: every query-independent tower.
        t0 = time.perf_counter()
        fe = clip.embed_frames(frames)
        F = np.array([e.vector for e in fe])
        we = clap.embed_windows(windows)
        A = np.array([we[w.path] for w in windows])
        t_towers = time.perf_counter() - t0
        sp = A @ probes.T
        speech = [
            w for w, row in zip(windows, sp, strict=True) if row[0] - row[1] > disp.speech_threshold
        ]
        t1 = time.perf_counter()
        transcripts = whisper.transcribe_windows(speech)
        texts = [(transcripts.get(w.path) or "").strip() for w in speech]
        indexed = [(w, t) for w, t in zip(speech, texts, strict=True) if t]
        T = np.array(disp.embedder.encode([t for _w, t in indexed])) if indexed else None
        t_ingest = t_towers + (time.perf_counter() - t1)
        rec = {
            "record": "ingest",
            "video": vid,
            "frames": len(frames),
            "windows": len(windows),
            "speech_windows": len(speech),
            "towers_s": round(t_towers, 3),
            "ingest_s": round(t_ingest, 3),
        }
        print(json.dumps(rec), flush=True)
        with OUT.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

        for item in questions:
            q = item["question"]

            # Per question, cached: text embeddings, dot products, routing, compressor.
            t0 = time.perf_counter()
            qv = np.array(clip.embed_text(q))
            vscores = {f.path: float(v) for f, v in zip(frames, F @ qv, strict=True)}
            qa = np.array(clap.embed_text(q))
            ascores = {w.path: float(v) for w, v in zip(windows, A @ qa, strict=True)}
            if indexed:
                qt = np.array(disp.embedder.encode([q.strip()]))[0]
                for (w, _t), v in zip(indexed, T @ qt, strict=True):
                    ascores[w.path] = float(v)
            c_imgs, c_auds = compress(q, dur, frames, windows, vscores, ascores)
            t_cached = time.perf_counter() - t0

            # Per question, live: what the GPU run timed.
            t0 = time.perf_counter()
            lv = clip.score_frames(frames, query=q)
            la = disp.score_windows(windows, q)
            l_imgs, l_auds = compress(q, dur, frames, windows, lv, la)
            t_live = time.perf_counter() - t0

            vdiff = max(abs(lv[k] - vscores[k]) for k in lv)
            adiff = max(abs(la[k] - ascores[k]) for k in la)
            rec = {
                "record": "question",
                "qid": item["question_id"],
                "video": vid,
                "cached_s": round(t_cached, 4),
                "live_s": round(t_live, 3),
                "max_visual_score_diff": vdiff,
                "max_audio_score_diff": adiff,
                "same_selection": (c_imgs, c_auds) == (l_imgs, l_auds),
                "frames": len(c_imgs),
                "audio_windows": len(c_auds),
            }
            print(json.dumps(rec), flush=True)
            with OUT.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")

    print("CACHED_SELECTION_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
