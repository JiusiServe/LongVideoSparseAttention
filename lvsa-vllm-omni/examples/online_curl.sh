#!/usr/bin/env bash
# Bash + curl + jq equivalent of online_client.py — submit, poll, download.
#
# Usage:
#   PROMPT="A dog running in the forest." \
#   OUTPUT=out.mp4 \
#   examples/online_curl.sh
#
# Env overrides:
#   HOST=localhost:8098
#   SIZE=832x480           (WIDTHxHEIGHT)
#   FPS=16
#   STEPS=40
#   GUIDANCE=4.0           (Wan 2.2 high-noise stage / HunyuanVideo)
#   GUIDANCE2=4.0          (Wan 2.2 low-noise stage; set empty to omit)
#   FLOW_SHIFT=5.0         (HV 480p typical; set empty to omit)
#   BOUNDARY_RATIO=0.875   (Wan 2.2 only; set empty to omit)
#   SEED=42
#   NUM_FRAMES=81          (Wan training horizon)
#   POLL_INTERVAL=2
#   TIMEOUT=3600

set -euo pipefail

HOST=${HOST:-localhost:8098}
PROMPT=${PROMPT:?"PROMPT env var required"}
OUTPUT=${OUTPUT:-out.mp4}
SIZE=${SIZE:-832x480}
FPS=${FPS:-16}
STEPS=${STEPS:-40}
GUIDANCE=${GUIDANCE:-4.0}
GUIDANCE2=${GUIDANCE2:-}
FLOW_SHIFT=${FLOW_SHIFT:-}
BOUNDARY_RATIO=${BOUNDARY_RATIO:-}
SEED=${SEED:-42}
NUM_FRAMES=${NUM_FRAMES:-81}
POLL_INTERVAL=${POLL_INTERVAL:-2}
TIMEOUT=${TIMEOUT:-3600}

if ! command -v jq >/dev/null; then
  echo "This script needs jq (apt install jq, brew install jq)." >&2
  exit 2
fi

# ── 1. Submit ────────────────────────────────────────────────────────────────
form_args=(
  -F "prompt=$PROMPT"
  -F "num_frames=$NUM_FRAMES"
  -F "size=$SIZE"
  -F "fps=$FPS"
  -F "num_inference_steps=$STEPS"
  -F "guidance_scale=$GUIDANCE"
  -F "seed=$SEED"
)
[ -n "$GUIDANCE2" ]      && form_args+=( -F "guidance_scale_2=$GUIDANCE2" )
[ -n "$FLOW_SHIFT" ]     && form_args+=( -F "flow_shift=$FLOW_SHIFT" )
[ -n "$BOUNDARY_RATIO" ] && form_args+=( -F "boundary_ratio=$BOUNDARY_RATIO" )

echo "[curl] POST http://$HOST/v1/videos"
resp=$(curl -sS -X POST "http://$HOST/v1/videos" "${form_args[@]}")
job_id=$(echo "$resp" | jq -r '.id // .video_id // .request_id')
if [ -z "$job_id" ] || [ "$job_id" = "null" ]; then
  echo "FAIL: no job id in response:" >&2
  echo "$resp" >&2
  exit 1
fi
echo "[curl] job_id = $job_id"

# ── 2. Poll ──────────────────────────────────────────────────────────────────
t0=$(date +%s)
last_status=""
while true; do
  status=$(curl -sS "http://$HOST/v1/videos/$job_id" | jq -r '.status')
  if [ "$status" != "$last_status" ]; then
    elapsed=$(( $(date +%s) - t0 ))
    echo "[curl] t=${elapsed}s  status=$status"
    last_status="$status"
  fi
  case "$status" in
    completed) break ;;
    failed)    echo "FAIL: server reported failure" >&2; exit 1 ;;
    queued|in_progress|*) ;;
  esac
  if [ $(( $(date +%s) - t0 )) -gt $TIMEOUT ]; then
    echo "FAIL: timeout after ${TIMEOUT}s" >&2
    exit 1
  fi
  sleep "$POLL_INTERVAL"
done

# ── 3. Download ──────────────────────────────────────────────────────────────
echo "[curl] GET http://$HOST/v1/videos/$job_id/content"
curl -sS -L "http://$HOST/v1/videos/$job_id/content" -o "$OUTPUT"
ls -la "$OUTPUT"
echo "[curl] done"
