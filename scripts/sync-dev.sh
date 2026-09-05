#!/usr/bin/env bash
# Sync the dev environment WITH the model extras.
#
# A bare `uv sync` prunes every optional extra, which removes torch and
# transformers and breaks CLIP, CLAP and Whisper without any error until the
# next model load. uv's `default-extras` setting does not prevent this
# (verified on uv 0.10.12), so the extras are passed explicitly here.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv sync "$@" \
  --extra dev --extra vision --extra sota --extra audio --extra sound
