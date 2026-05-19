#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${WORK_DIR:-data/videomme-runpod-subset}"
OUTPUT_DIR="${OUTPUT_DIR:-reports/runpod-videomme}"
VIDEO_COUNT="${VIDEO_COUNT:-2}"
QUESTIONS_PER_VIDEO="${QUESTIONS_PER_VIDEO:-3}"
FRAME_COUNTS="${FRAME_COUNTS:-1,2}"
SAMPLE_COUNT="${SAMPLE_COUNT:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
MODEL="${MODEL:-llava-hf/llava-onevision-qwen2-0.5b-ov-hf}"
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
WHISPER_MODEL_SIZE="${WHISPER_MODEL_SIZE:-tiny}"
FRAME_SAMPLING="${FRAME_SAMPLING:-start}"
PROMPT_STRATEGY="${PROMPT_STRATEGY:-default}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

ARGS=(
  --work-dir "$WORK_DIR"
  --output-dir "$OUTPUT_DIR"
  --video-count "$VIDEO_COUNT"
  --questions-per-video "$QUESTIONS_PER_VIDEO"
  --frame-counts "$FRAME_COUNTS"
  --sample-count "$SAMPLE_COUNT"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --model "$MODEL"
  --device-map "$DEVICE_MAP"
  --torch-dtype "$TORCH_DTYPE"
  --whisper-model-size "$WHISPER_MODEL_SIZE"
  --frame-sampling "$FRAME_SAMPLING"
  --prompt-strategy "$PROMPT_STRATEGY"
  --single-config
  --preset balanced
  --visual-scorer clip_scene
  --audio-scorer baseline
  --adaptive-budget
)

if [[ "$SKIP_PREPARE" == "1" ]]; then
  ARGS+=(--skip-prepare)
fi

python scripts/run_videomme_frame_sweep.py "${ARGS[@]}"
