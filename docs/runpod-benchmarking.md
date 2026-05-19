# RunPod Benchmarking

RunPod is the recommended first cloud target for Gist benchmark work. Use the MacBook for
development and unit tests, then use a RunPod GPU pod for Video-LLM benchmark runs.

## Recommended Pod

Start with:

- GPU: RTX 4090 24GB
- Image/template: PyTorch CUDA image
- Volume: 100-200GB persistent volume
- Disk: enough for model cache, videos, reports, and intermediate clips

If RTX 4090 VRAM is not enough, move to A100 40GB or A100 80GB. H100 is not necessary yet.

## First-Time Setup

Clone the repository on the pod:

```bash
git clone https://github.com/thepreakerebi/gist.git
cd gist
```

Run setup:

```bash
bash scripts/setup_runpod.sh
source .venv/bin/activate
```

The setup script installs FFmpeg when `apt-get` is available, creates `.venv`, installs Gist
with development, SOTA, audio, vision, and sound extras, prints CUDA status, and runs a small
test subset.

## Run A Practical Benchmark

Use the targeted frame-density benchmark first:

```bash
source .venv/bin/activate
bash scripts/run_runpod_videomme.sh
```

Default behavior:

- prepares a small Video-MME subset
- runs one strong Gist variant, `clip_scene + adaptive_budget`
- compares frame budgets `1,2`
- uses `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`
- writes reports under `reports/runpod-videomme`

The key summary files are:

```text
reports/runpod-videomme/frame-sweep-summary.md
reports/runpod-videomme/frame-sweep-summary.json
reports/runpod-videomme/frames-1/sota-report.html
reports/runpod-videomme/frames-2/sota-report.html
```

## Larger Run

After the small run completes cleanly, increase the sweep:

```bash
FRAME_COUNTS=1,2,4 \
VIDEO_COUNT=4 \
QUESTIONS_PER_VIDEO=3 \
SAMPLE_COUNT=32 \
MAX_NEW_TOKENS=24 \
bash scripts/run_runpod_videomme.sh
```

If you only want to rerun benchmarks using an already prepared dataset:

```bash
SKIP_PREPARE=1 bash scripts/run_runpod_videomme.sh
```

## Useful Environment Variables

```bash
MODEL=llava-hf/llava-onevision-qwen2-0.5b-ov-hf
DEVICE_MAP=cuda:0
TORCH_DTYPE=float16
FRAME_COUNTS=1,2
VIDEO_COUNT=2
QUESTIONS_PER_VIDEO=3
SAMPLE_COUNT=16
MAX_NEW_TOKENS=16
WHISPER_MODEL_SIZE=tiny
FRAME_SAMPLING=start
PROMPT_STRATEGY=default
OUTPUT_DIR=reports/runpod-videomme
WORK_DIR=data/videomme-runpod-subset
```

Use `DEVICE_MAP=cuda:0` and `TORCH_DTYPE=float16` on RunPod. Avoid `device_map=auto` unless
there is a specific reason, because local testing showed auto offload can become unstable.

## Long Runs

Use `tmux` so benchmark jobs survive browser/SSH disconnects:

```bash
tmux new -s gist
source .venv/bin/activate
bash scripts/run_runpod_videomme.sh
```

Detach with `Ctrl-b d`. Reattach with:

```bash
tmux attach -t gist
```

## Pull Reports Back

From your local machine:

```bash
scp -r root@YOUR_RUNPOD_HOST:/workspace/gist/reports/runpod-videomme ./reports/
```

Or zip reports on the pod:

```bash
tar -czf runpod-videomme-reports.tar.gz reports/runpod-videomme
```

## Cost Discipline

Do not leave the pod running after benchmarks finish. Stop or terminate it after copying the
reports. Keep persistent storage only if you want to reuse model and dataset caches.
