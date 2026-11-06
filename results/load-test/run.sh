#!/usr/bin/env bash
# Step the load up and keep one k6 summary per level in raw/.
#
#   VIDEO=<ready video id> ./results/load-test/run.sh ask    "1 5 10 25 50"
#   VIDEO=<ready video id> ./results/load-test/run.sh browse "1 10 50 100 200"
#   VIDEO=<id> CLIP_DIR=<library>/<id>/clips ./results/load-test/run.sh varied "1 5 10 25 50"
#
# Each level starts from an empty conversation, so the detail payload and the
# message table do not grow from one level to the next.
set -euo pipefail
MODE=${1:-ask}
LEVELS=${2:-"1 5 10 25 50"}
API=${API:-http://127.0.0.1:8100}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$HERE/raw"
: "${VIDEO:?set VIDEO to the id of a ready video}"

for vus in $LEVELS; do
  curl -s -o /dev/null -X DELETE "$API/v1/library/videos/$VIDEO/conversation"
  # CLIP_DIR, when set, is the video's cut-clip cache; emptying it makes every
  # level start cold, so a clip cut at one level is not reused at the next.
  if [ -n "${CLIP_DIR:-}" ]; then rm -f "$CLIP_DIR"/*.mp4; fi
  echo "== $MODE, $vus virtual users"
  k6 run --quiet --no-color \
    -e API="$API" -e VIDEO="$VIDEO" -e MODE="$MODE" -e VUS="$vus" \
    -e DURATION="${DURATION:-60s}" -e THINK="${THINK:-1}" \
    --summary-export "$HERE/raw/$MODE-$vus.json" \
    "$HERE/load.js" > "$HERE/raw/$MODE-$vus.log" 2>&1 || true
  tail -n 3 "$HERE/raw/$MODE-$vus.log"
done
