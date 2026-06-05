#!/usr/bin/env bash
# Minimal vllm-omni serving recipe for LVSA-accelerated video generation.
#
# Usage:
#   examples/vllm_omni_serve.sh wan       /path/to/Wan2.1-T2V-1.3B-Diffusers
#   examples/vllm_omni_serve.sh hunyuan   /path/to/HunyuanVideo-1.5-Diffusers-480p_t2v
#
# After it starts, hit the OpenAI-compatible endpoint at http://localhost:8100/v1/...
#
# Prerequisites: pip install -e . && pip install -e lvsa-vllm-omni/

set -euo pipefail

MODEL_FAMILY=${1:?"usage: $0 {wan|hunyuan} MODEL_PATH"}
MODEL_PATH=${2:?"usage: $0 {wan|hunyuan} MODEL_PATH"}
PORT=${PORT:-8100}
DTYPE=${DTYPE:-bfloat16}

case "$MODEL_FAMILY" in
  wan)
    export LVSA_WAN_HOOK=1
    export LVSA_REFERENCE_LATENT_FRAMES=21
    ;;
  hunyuan|hunyuanvideo|hv)
    export LVSA_REFERENCE_LATENT_FRAMES=33
    ;;
  *)
    echo "Unknown model family: $MODEL_FAMILY (expected wan|hunyuan)" >&2
    exit 2
    ;;
esac

# vllm-omni 0.22 dropped the DIFFUSION_ATTENTION_BACKEND env var; the
# `lvsa_vllm_omni.serve` wrapper injects the per-role AttentionConfig
# (--diffusion-attention-config) automatically.
export LVSA_AUTO_KEYFRAMES=1
export LVSA_ROTATE_KEYFRAMES=1

echo "[serve] family=$MODEL_FAMILY model=$MODEL_PATH port=$PORT dtype=$DTYPE"
echo "[serve] LVSA env:"
env | grep '^LVSA_' | sort

exec python -m lvsa_vllm_omni.serve "$MODEL_PATH" --port "$PORT" --dtype "$DTYPE"
