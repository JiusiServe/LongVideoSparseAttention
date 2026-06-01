#!/usr/bin/env bash
# Launch a vllm-omni server with LVSA enabled for Wan 2.x.
#
# Usage:
#   examples/serve_wan.sh /path/to/Wan2.1-T2V-1.3B-Diffusers
#   PORT=8200 examples/serve_wan.sh /path/to/Wan...
#
# Env overrides (defaults shown):
#   PORT=8098
#   DTYPE=bfloat16
#   TP=1                   # tensor-parallel-size (multi-GPU)
#   LVSA_REFERENCE_LATENT_FRAMES=21
#   LVSA_TOTAL_LATENT_FRAMES=21   # set per generation length / 4
#   LVSA_AUTO_KEYFRAMES=1
#   LVSA_ROTATE_KEYFRAMES=1
#   LVSA_SPARSITY_SCALE=1.0
#
# After it starts, hit http://localhost:$PORT/v1/videos.  See
# examples/online_client.py for a complete request flow.

set -euo pipefail

MODEL_PATH=${1:?"usage: $0 MODEL_PATH"}
PORT=${PORT:-8098}
DTYPE=${DTYPE:-bfloat16}
TP=${TP:-1}

# ── LVSA backend selection ───────────────────────────────────────────────────
export DIFFUSION_ATTENTION_BACKEND=LVSA
export LVSA_WAN_HOOK=1
export LVSA_REFERENCE_LATENT_FRAMES=${LVSA_REFERENCE_LATENT_FRAMES:-21}
export LVSA_TOTAL_LATENT_FRAMES=${LVSA_TOTAL_LATENT_FRAMES:-21}
export LVSA_AUTO_KEYFRAMES=${LVSA_AUTO_KEYFRAMES:-1}
export LVSA_ROTATE_KEYFRAMES=${LVSA_ROTATE_KEYFRAMES:-1}
export LVSA_SPARSITY_SCALE=${LVSA_SPARSITY_SCALE:-1.0}

# Pick a Python binary: honour $PYTHON if caller passed one, else fall back
# to the LVSA-bundled vllm-omni venv (at <repo>/.venv-vllm), else system
# `python` / `python3` on PATH.
PYTHON_BIN=${PYTHON:-}
if [ -z "$PYTHON_BIN" ]; then
  SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
  CAND="$SCRIPT_DIR/../../.venv-vllm/bin/python"
  if [ -x "$CAND" ]; then
    PYTHON_BIN="$CAND"
  else
    PYTHON_BIN=$(command -v python 2>/dev/null || command -v python3 2>/dev/null) || true
  fi
fi
[ -n "$PYTHON_BIN" ] || { echo "ERROR: no usable python found" >&2; exit 1; }

echo "[serve_wan] model=$MODEL_PATH port=$PORT dtype=$DTYPE tp=$TP python=$PYTHON_BIN"
echo "[serve_wan] LVSA env:"
env | grep '^LVSA_\|DIFFUSION_ATTENTION_BACKEND' | sort

exec "$PYTHON_BIN" -m lvsa_vllm_omni.serve "$MODEL_PATH" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --tensor-parallel-size "$TP"
