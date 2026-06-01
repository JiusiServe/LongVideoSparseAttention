#!/usr/bin/env bash
# Reproduce the SotA comparison table from the paper.
#
# Grid: 5 prompts × 3 horizons (2×/3×/4× = 165/249/333 frames) × 4 methods
#       (Dense / RIFLEx / LVSA-SDPA / LVSA-FI). UltraViCo runs separately —
#       see scripts/paper_results/sota_ultravico_rerun.sh in the dev repo
#       or run UltraViCo's `ultra-wan` branch directly.
#
# Defaults match the paper exactly: Wan 2.1 1.3B, 480×832, CFG 5.0, seed 16,
# 50 denoising steps.
#
# Usage:
#   MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers \
#       benchmarks/sota_comparison.sh
#
# Outputs: $OUTDIR/<model>__<backend>__<horizon>__<prompt>.{mp4,log}
# To score: benchmarks/score_vqeval.sh && benchmarks/score_vbench.sh
# To aggregate: python benchmarks/aggregate.py --outdir $OUTDIR

set -euo pipefail

# ── Config (override via env) ──────────────────────────────────────────────
MODEL_PATH=${MODEL_PATH:?"MODEL_PATH must point at a Wan 2.1 1.3B Diffusers checkpoint"}
OUTDIR=${OUTDIR:-out/sota_comparison}
PROMPTS_FILE=${PROMPTS_FILE:-benchmarks/long_prompts.json}
MODEL_TAG=${MODEL_TAG:-wan13b}
STEPS=${STEPS:-50}
SEED=${SEED:-16}
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-832}
CUDA_DEVICE=${CUDA_DEVICE:-0}

mkdir -p "$OUTDIR"

# Horizons: (label num_frames)  — UltraViCo's 84r-3 parameterization
HORIZONS=("2x:165" "3x:249" "4x:333")

# Methods and their wan_generate.py flags
declare -A BACKEND_FLAGS=(
  ["dense"]=""
  ["riflex"]="--riflex --riflex-s {RATIO}"
  ["lvsa"]="--lvsa --rotate-keyframes --auto-keyframes"
  ["lvsa-fi"]="--lvsa --flashinfer --rotate-keyframes --auto-keyframes"
)

# Map horizon label to riflex-s value
declare -A RIFLEX_S=(
  ["2x"]="2.0"
  ["3x"]="3.0"
  ["4x"]="4.0"
)

# Read the 5 prompt tags from the JSON manifest
mapfile -t TAGS < <(python -c "
import json, sys
data = json.load(open('$PROMPTS_FILE'))
for p in data['prompts']:
    print(p['tag'])
")

# Helper: extract prompt text by tag
prompt_for() {
  python -c "
import json
data = json.load(open('$PROMPTS_FILE'))
for p in data['prompts']:
    if p['tag'] == '$1':
        print(p['prompt'])
        break
"
}

# Total job count for progress
TOTAL=$((${#HORIZONS[@]} * ${#BACKEND_FLAGS[@]} * ${#TAGS[@]}))
COUNT=0

for horizon_pair in "${HORIZONS[@]}"; do
  HORIZON_LABEL="${horizon_pair%%:*}"
  NUM_FRAMES="${horizon_pair##*:}"
  for BACKEND in dense riflex lvsa lvsa-fi; do
    FLAGS_TEMPLATE="${BACKEND_FLAGS[$BACKEND]}"
    # Substitute {RATIO} for the riflex case
    FLAGS=$(echo "$FLAGS_TEMPLATE" | sed "s/{RATIO}/${RIFLEX_S[$HORIZON_LABEL]}/")
    for TAG in "${TAGS[@]}"; do
      COUNT=$((COUNT + 1))
      STEM="${MODEL_TAG}__${BACKEND}__${HORIZON_LABEL}__${TAG}"
      MP4="$OUTDIR/${STEM}.mp4"
      LOG="$OUTDIR/${STEM}.log"

      if [ -s "$MP4" ]; then
        echo "[$COUNT/$TOTAL] skip $STEM (exists)"
        continue
      fi

      PROMPT="$(prompt_for "$TAG")"
      echo "[$COUNT/$TOTAL] $STEM  flags=$FLAGS  frames=$NUM_FRAMES"

      CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python examples/wan_generate.py \
        --model "$MODEL_PATH" \
        --prompt "$PROMPT" \
        --num-frames "$NUM_FRAMES" \
        --height "$HEIGHT" --width "$WIDTH" \
        --steps "$STEPS" --seed "$SEED" \
        --output-dir "$(dirname "$MP4")" --output-name "$(basename "$MP4")" \
        $FLAGS \
        2>&1 | tee "$LOG"
    done
  done
done

echo
echo "[done] $TOTAL jobs"
echo "Next: bash benchmarks/score_vqeval.sh $OUTDIR && bash benchmarks/score_vbench.sh $OUTDIR"
echo "Then: python benchmarks/aggregate.py --outdir $OUTDIR"
