#!/usr/bin/env bash
# Launch a vllm-omni server with LVSA enabled for HunyuanVideo 1.5.
#
# Usage:
#   examples/serve_hunyuan.sh /path/to/HunyuanVideo-1.5-Diffusers-480p_t2v
#   PORT=8200 examples/serve_hunyuan.sh /path/to/HV...
#
# Env overrides (defaults shown):
#   PORT=8098
#   DTYPE=bfloat16
#   TP=1                   # tensor-parallel-size (multi-GPU)
#   LVSA_REFERENCE_LATENT_FRAMES=33
#   LVSA_TOTAL_LATENT_FRAMES=33   # set per generation length / 4
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
# vllm-omni 0.22 dropped the DIFFUSION_ATTENTION_BACKEND env var; the
# `lvsa_vllm_omni.serve` wrapper below injects the per-role AttentionConfig
# (--diffusion-attention-config) automatically. The hook drives the sparse
# attention; LVSA_HUNYUAN_HOOK=1 keeps it active.
export LVSA_HUNYUAN_HOOK=${LVSA_HUNYUAN_HOOK:-1}
export LVSA_REFERENCE_LATENT_FRAMES=${LVSA_REFERENCE_LATENT_FRAMES:-33}
export LVSA_TOTAL_LATENT_FRAMES=${LVSA_TOTAL_LATENT_FRAMES:-33}
export LVSA_AUTO_KEYFRAMES=${LVSA_AUTO_KEYFRAMES:-1}
export LVSA_ROTATE_KEYFRAMES=${LVSA_ROTATE_KEYFRAMES:-1}
export LVSA_SPARSITY_SCALE=${LVSA_SPARSITY_SCALE:-1.0}

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

echo "[serve_hunyuan] model=$MODEL_PATH port=$PORT dtype=$DTYPE tp=$TP python=$PYTHON_BIN"
echo "[serve_hunyuan] LVSA env:"
env | grep '^LVSA_' | sort

# --vae-use-tiling / --vae-use-slicing: HunyuanVideo 1.5's VAE decode OOMs at
# longer clips (~65+ frames) on an 80 GB GPU without them — independent of LVSA
# (the sparse attention completes; the dense VAE decode is the ceiling). They
# are effectively lossless. Override VAE_OPT="" to disable.
VAE_OPT=${VAE_OPT:---vae-use-tiling --vae-use-slicing}

exec "$PYTHON_BIN" -m lvsa_vllm_omni.serve "$MODEL_PATH" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --tensor-parallel-size "$TP" \
    $VAE_OPT
