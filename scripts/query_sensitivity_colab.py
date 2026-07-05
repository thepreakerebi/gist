"""Query-sensitivity diagnostic (Gist is genuinely query-conditional).

For each real question, Gist selects evidence for the true query and for an
unrelated distractor query over the SAME candidate pool. If Gist were
query-agnostic (just picking generically-salient frames), the two selections
would overlap heavily. Low overlap => selection is driven by the query.

Plan §8 success criterion: correct-query vs distractor-query retain <50% overlap.
Reuses frames already extracted by av_bench.py (cached at /content/fr_<vid>).
"""
import os, json, glob
os.environ.setdefault("LD_LIBRARY_PATH", "/usr/lib64-nvidia")
from pathlib import Path

QFILE = "/content/videomme_av6.json"
FRAME_BUDGET, WIN, NCAND = 8, 30, 64
DISTRACTOR = "a close-up of a birthday cake with lit candles indoors"

from gist.vision.clip import HuggingFaceClipFrameScorer
from gist.media.models import ExtractedFrame
from gist.core.schemas import Candidate, CompressionRequest, Modality
from gist.core.presets import CompressionPreset
from gist.core.token_estimation import TokenEstimatorProfile
from gist.core.compressor import GistCompressor

clip = HuggingFaceClipFrameScorer()
rows = json.load(open(QFILE))
by_vid = {}
for r in rows:
    by_vid.setdefault(r["videoID"], []).append(r)

def frames_for(vid):
    fd = Path(f"/content/fr_{vid}")
    fs = sorted(fd.glob("f*.jpg"))
    if not fs:
        return None
    n = len(fs)
    return [ExtractedFrame(index=i, timestamp_seconds=float(i), path=p) for i, p in enumerate(fs)]

def select_visual(query, frames, dur):
    vs = clip.score_frames(frames, query=query)
    vc = [Candidate(id=f"v{f.index}", timestamp_seconds=f.timestamp_seconds,
                    saliency_score=vs.get(f.path, 0.0), asset_path=f.path) for f in frames]
    resp = GistCompressor().compress(CompressionRequest(
        video_id="v", query=query, duration_seconds=dur, preset=CompressionPreset.BALANCED,
        adaptive_budget=True, decompose_query=True, token_estimator=TokenEstimatorProfile.GENERIC,
        task_aware_selection=True, visual_candidates=vc, audio_candidates=[]))
    return {s.id for s in resp.selected if s.modality == Modality.VISUAL}

overlaps = []
for vid, qs in by_vid.items():
    frames = frames_for(vid)
    if not frames:
        print(f"SKIP {vid}", flush=True); continue
    dur = float(len(frames))
    distract_sel = select_visual(DISTRACTOR, frames, dur)
    for item in qs:
        real_sel = select_visual(item["question"], frames, dur)
        if not real_sel and not distract_sel:
            continue
        union = real_sel | distract_sel
        jac = len(real_sel & distract_sel) / len(union) if union else 0.0
        overlaps.append(jac)
        print(f"{item['question_id']} real_n={len(real_sel)} distract_n={len(distract_sel)} jaccard={jac:.2f}", flush=True)

mean = sum(overlaps) / len(overlaps) if overlaps else 0.0
print(f"RESULT mean_jaccard_overlap={mean:.2f} over {len(overlaps)} questions "
      f"(< 0.50 => query-conditional)", flush=True)
print("QSENS_DONE", flush=True)
