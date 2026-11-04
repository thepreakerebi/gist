#!/usr/bin/env bash
# One RunPod session for the two outstanding GPU runs, budgeted at $10 total.
#
#   bash results/pod_session.sh setup        # ~10 min: envs, model, checks
#   bash results/pod_session.sh fetch        # fallback: videos from YouTube
#   bash results/pod_session.sh plumb        # ~5 min: both CUDA paths, one video
#   bash results/pod_session.sh efficiency   # results/efficiency-measured/
#   bash results/pod_session.sh omnipack     # results/omnipack-replication/
#   bash results/pod_session.sh h2h_plumb    # ~5 min: the Gist-vs-OmniPack runner on one video
#   bash results/pod_session.sh h2h          # results/gist-vs-omnipack/: timed run of record
#   bash results/pod_session.sh h2h_flops    # FLOP-counting pass, two videos
#   bash results/pod_session.sh h2h_long_plumb / h2h_long / h2h_long_flops
#                                            # the same on whole-video input (RUN.md, follow-up)
#   bash results/pod_session.sh collect      # tar everything worth keeping
#
# Run each step under nohup (see POD_SESSION.md) so a dropped web terminal does
# not kill a paid run. Every step is safe to re-run; finished downloads and
# installs are skipped.
#
# Two Python environments, on purpose. The efficiency run uses the system env
# on transformers 5.13, because that is what produced the fp16 accuracy numbers
# it is compared against. The OmniPack runner uses a side venv on 4.52.3, the
# pin OmniZip and OmniScope share, which is what its model code was written and
# CPU-tested against. The venv inherits torch from the system env, so only the
# version-sensitive packages are installed twice.
#
# The `av` and `librosa` pins are not cosmetic. With torch 2.6 / torchvision
# 0.21, PyAV 14+ breaks torchvision's video reader (`metadata_errors`), and
# librosa 1.0 can no longer read the audioread handle qwen-omni-utils passes it
# in the 4.52.3 env. Both failures were hit in the CPU smoke tests, not guessed.
#
# decord is not optional either. Without it qwen-vl-utils falls back to
# torchvision, which decodes the *whole* video into RAM before sampling 32
# frames; an hour-long Video-MME video exceeded the pod's 62 GB limit and the
# runner was killed. With decord the same video peaks at 3.2 GB. Found on the
# first pod session, 2026-10-28.

set -euo pipefail

REPO=${REPO:-/root/gist}
SIDE=${SIDE:-/root/op452}
DATA=${DATA:-/root/data}
VIDS=${VIDS:-/content/vids}
LOGS=${LOGS:-/root/logs}
mkdir -p "$DATA" "$VIDS" "$LOGS"

cudnn_path() {
  python - <<'PY'
import os
import nvidia.cublas
import nvidia.cudnn

print(f"{os.path.dirname(nvidia.cudnn.__file__)}/lib:{os.path.dirname(nvidia.cublas.__file__)}/lib")
PY
}

# faster-whisper (CTranslate2) needs cuDNN 9 and cuBLAS 12 on the loader path.
# torch's wheels already ship both; point the loader at them.
export LD_LIBRARY_PATH="$(cudnn_path 2>/dev/null || true):${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

setup() {
  # ffmpeg backs every video and audio decode in both runners; the PyTorch
  # images do not always ship it.
  command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null; }

  cd /root
  [ -d "$REPO" ] || git clone https://github.com/thepreakerebi/gist.git "$REPO"
  cd "$REPO"

  pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
  pip install -q -e '.[vision,audio,sound,sota]' \
    "transformers==5.13.0" soundfile librosa qwen-omni-utils "av==13.1.0" yt-dlp decord
  # torchaudio ships with the image, pinned to the image's torch; it would
  # fail to import against 2.6 and nothing here uses it.
  pip uninstall -y -q torchaudio || true

  # virtualenv rather than `python -m venv`: Ubuntu's system Python often lacks
  # ensurepip, and that failure would only show up on the pod.
  pip install -q virtualenv
  [ -d "$SIDE" ] || python -m virtualenv -q --system-site-packages "$SIDE"
  "$SIDE/bin/pip" install -q "transformers==4.52.3" "librosa==0.10.2.post1" \
    "av==13.1.0" qwen-omni-utils
  # NVML energy counter for gist.eval.profiling; both envs see it.
  pip install -q nvidia-ml-py

  python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('cuda ok:', torch.cuda.get_device_name(0))"
  python -c "from gist.eval.profiling import DeviceInfo; print(DeviceInfo.capture().describe())"
  "$SIDE/bin/python" -c "import transformers; assert transformers.__version__ == '4.52.3'; print('side env ok')"
  export LD_LIBRARY_PATH="$(cudnn_path):${LD_LIBRARY_PATH:-}"
  python -c "from faster_whisper import WhisperModel; WhisperModel('tiny', device='cuda', compute_type='float16'); print('whisper on cuda ok')"

  # ~16 GB. Shared by both envs through the default Hugging Face cache.
  python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-Omni-7B')"
  # Gist's scorers. CLIP's processor loads with local_files_only, so on a fresh
  # pod it fails unless the repo is already cached (found 2026-11-04).
  python -c "from huggingface_hub import snapshot_download as s; [s(m) for m in ('openai/clip-vit-base-patch32', 'laion/clap-htsat-unfused', 'sentence-transformers/all-MiniLM-L6-v2')]"

  for q in videomme_av6.json videomme_av_all.json; do
    [ -f "$DATA/$q" ] || echo "MISSING $DATA/$q: receive the bundle from the Mac (POD_SESSION.md step 3)"
  done
  [ -f "$DATA/videomme_av_all.json" ] || { echo "SETUP_DONE (videos not checked yet)"; return 0; }
  python - <<PY
import json, pathlib
rows = json.loads(pathlib.Path("$DATA/videomme_av_all.json").read_text())
vids = {r["videoID"] for r in rows}
have = {v for v in vids if (pathlib.Path("$VIDS") / f"videomme-{v}.mp4").exists()}
print(f"videos present: {len(have)}/{len(vids)}", "" if have == vids else f"missing {sorted(vids - have)}")
PY
  echo "SETUP_DONE"
}

plumb() {
  # ~5 min on one video: both runners down their CUDA paths, into separate
  # files so nothing here leaks into the measurements of record.
  cd "$REPO"
  python - <<PY
import json, pathlib
rows = json.loads(pathlib.Path("$DATA/videomme_av6.json").read_text())
first = rows[0]["videoID"]
pathlib.Path("$DATA/plumb.json").write_text(json.dumps([r for r in rows if r["videoID"] == first]))
PY
  GIST_SKIP_DENSE=1 GIST_MEASUREMENTS=/tmp/plumb_eff_m.jsonl GIST_RESULTS=/tmp/plumb_eff_r.jsonl \
    python results/efficiency-measured/efficiency_bench.py "$DATA/plumb.json" \
    2>&1 | tee "$LOGS/plumb_efficiency.log"
  OMNIPACK_RATIOS=0.25 GIST_MEASUREMENTS=/tmp/plumb_op_m.jsonl GIST_RESULTS=/tmp/plumb_op_r.jsonl \
    "$SIDE/bin/python" results/omnipack-replication/omnipack_h2h.py "$DATA/plumb.json" \
    2>&1 | tee "$LOGS/plumb_omnipack.log"
  grep -h "control:\|RESULT_\|FATAL\|ERR\|OOM" "$LOGS"/plumb_*.log || true
  echo "PLUMB_DONE"
}

efficiency() {
  # Fresh files: the measurement log appends, and a half-finished earlier
  # attempt must not be averaged into the run of record.
  rm -f /tmp/efficiency_measurements.jsonl /tmp/efficiency_answers.jsonl
  cd "$REPO"
  python results/efficiency-measured/efficiency_bench.py "$DATA/videomme_av6.json" \
    2>&1 | tee "$LOGS/efficiency.log"
  python -m gist.eval.efficiency \
    --measurements /tmp/efficiency_measurements.jsonl \
    --measured-baseline full \
    --measured-markdown results/efficiency-measured/measured.md \
    --measured-json results/efficiency-measured/measured.json \
    --markdown results/efficiency-measured/analytic.md \
    --json results/efficiency-measured/analytic.json \
    2>&1 | tee -a "$LOGS/efficiency.log"
  cp /tmp/efficiency_measurements.jsonl /tmp/efficiency_answers.jsonl results/efficiency-measured/
}

omnipack() {
  rm -f /tmp/omnipack_measurements.jsonl /tmp/omnipack_h2h_results.jsonl
  cd "$REPO"
  local qfile="${OMNIPACK_QFILE:-$DATA/videomme_av_all.json}"
  OMNIPACK_RATIOS="${OMNIPACK_RATIOS:-0.25,0.20,0.15,0.10}" \
    "$SIDE/bin/python" results/omnipack-replication/omnipack_h2h.py "$qfile" \
    2>&1 | tee "$LOGS/omnipack.log"
  cp /tmp/omnipack_measurements.jsonl /tmp/omnipack_h2h_results.jsonl results/omnipack-replication/
}

h2h_run() {
  # $1: output dir, rest: extra env assignments
  local out="$1"; shift
  rm -rf "$out"
  cd "$REPO"
  env H2H_OUT="$out" H2H_WORK=/root/h2h_work GIST_VIDEOS="$VIDS" "$@" \
    "$SIDE/bin/python" results/gist-vs-omnipack/h2h.py "${H2H_QFILE:-$DATA/videomme_av_all.json}"
}

h2h_plumb() {
  h2h_run /root/h2h_plumb H2H_MAX_VIDEOS=1 2>&1 | tee "$LOGS/h2h_plumb.log"
  grep -h "exactness\|control:\|FATAL\|ERR\|OOM\|H2H_DONE" "$LOGS/h2h_plumb.log" || true
}

h2h() {
  h2h_run /root/h2h 2>&1 | tee "$LOGS/h2h.log"
  mkdir -p results/gist-vs-omnipack
  cp /root/h2h/measurements.jsonl /root/h2h/answers.jsonl results/gist-vs-omnipack/
}

h2h_flops() {
  h2h_run /root/h2h_flops H2H_COUNT_FLOPS=1 H2H_MAX_VIDEOS=2 H2H_PRODUCT_WHISPER=0 2>&1 | tee "$LOGS/h2h_flops.log"
  cp /root/h2h_flops/measurements_flops.jsonl /root/h2h_flops/answers_flops.jsonl results/gist-vs-omnipack/
}

# Whole-video input: 128 frames, the whole audio track (Qwen's 300 s cap raised),
# OmniPack's stage 1 per 300 s of audio (A8), and a ratio matched to Gist's budget.
LONG_ENV=(OMNIPACK_FRAMES=128 H2H_AUDIO_CAP_SECONDS=4000 H2H_STAGE1_CHUNK_SECONDS=300
          H2H_RATIOS=0.25,0.10,0.04 H2H_VERIFY=0 H2H_PRODUCT_WHISPER=0)

h2h_long_plumb() {
  h2h_run /root/h2h_long_plumb "${LONG_ENV[@]}" H2H_MAX_VIDEOS=1 2>&1 | tee "$LOGS/h2h_long_plumb.log"
  grep -h "exactness\\|not run\\|FATAL\\|ERR\\|OOM\\|H2H_DONE" "$LOGS/h2h_long_plumb.log" || true
}

h2h_long() {
  h2h_run /root/h2h_long "${LONG_ENV[@]}" 2>&1 | tee "$LOGS/h2h_long.log"
  mkdir -p results/gist-vs-omnipack-long
  cp /root/h2h_long/measurements.jsonl /root/h2h_long/answers.jsonl results/gist-vs-omnipack-long/
}

h2h_long_flops() {
  h2h_run /root/h2h_long_flops "${LONG_ENV[@]}" H2H_COUNT_FLOPS=1 H2H_MAX_VIDEOS=2 2>&1 | tee "$LOGS/h2h_long_flops.log"
  mkdir -p results/gist-vs-omnipack-long
  cp /root/h2h_long_flops/measurements_flops.jsonl /root/h2h_long_flops/answers_flops.jsonl results/gist-vs-omnipack-long/
}

fetch() {
  # Fallback only. The bundle from the Mac is the primary route, because
  # YouTube may refuse a datacenter address.
  cd "$REPO"
  python results/fetch_videos.py "$DATA/videomme_av_all.json" "$VIDS"
}

collect() {
  cd "$REPO"
  tar czf /root/results.tgz \
    results/efficiency-measured results/omnipack-replication results/gist-vs-omnipack results/gist-vs-omnipack-long "$LOGS" 2>/dev/null || true
  ls -lh /root/results.tgz
  echo "Send it home:  runpodctl send /root/results.tgz"
}

case "${1:-}" in
  setup) setup ;;
  fetch) fetch ;;
  plumb) plumb ;;
  efficiency) efficiency ;;
  omnipack) omnipack ;;
  h2h_plumb) h2h_plumb ;;
  h2h) h2h ;;
  h2h_flops) h2h_flops ;;
  h2h_long_plumb) h2h_long_plumb ;;
  h2h_long) h2h_long ;;
  h2h_long_flops) h2h_long_flops ;;
  collect) collect ;;
  *) echo "usage: $0 {setup|fetch|plumb|efficiency|omnipack|h2h_plumb|h2h|h2h_flops|h2h_long_plumb|h2h_long|h2h_long_flops|collect}"; exit 2 ;;
esac
