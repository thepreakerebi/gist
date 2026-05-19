#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_AUDIO="${INSTALL_AUDIO:-1}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends ffmpeg git git-lfs curl ca-certificates
  else
    echo "ffmpeg is required. Install ffmpeg, then rerun this script." >&2
    exit 1
  fi
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
if [[ "$INSTALL_AUDIO" == "1" ]]; then
  pip install -e ".[dev,sota,audio,vision,sound]"
else
  pip install -e ".[dev,sota,vision,sound]"
fi

python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
    print(f"cuda_capability={torch.cuda.get_device_capability(0)}")
PY

pytest tests/test_answers.py tests/test_frame_sweep.py tests/test_hf_vision_gateway.py

echo "RunPod setup complete. Activate with: source ${VENV_DIR}/bin/activate"
