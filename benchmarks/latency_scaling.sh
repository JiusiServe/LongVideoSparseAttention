#!/usr/bin/env bash
# Reproduce the latency-scaling sweep from the paper.
#
# For a single prompt, sweep frame counts to show wall-time scaling for
# Dense vs LVSA-FI on Wan 2.1 1.3B.
#
# Usage:
#   MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers \
#       benchmarks/latency_scaling.sh
#
# Outputs: $OUTDIR/wan13b__{dense,lvsa-fi}__{1x,2x,4x,6x}__dog_forest.{mp4,log}

set -euo pipefail

MODEL_PATH=${MODEL_PATH:?"MODEL_PATH must point at a Wan 2.1 1.3B Diffusers checkpoint"}
OUTDIR=${OUTDIR:-out/latency_scaling}
STEPS=${STEPS:-50}
SEED=${SEED:-16}
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-832}
CUDA_DEVICE=${CUDA_DEVICE:-0}
PROMPTS_FILE=${PROMPTS_FILE:-benchmarks/long_prompts.json}
TAG=${TAG:-dog_forest}

mkdir -p "$OUTDIR"

# (label num_frames) — paper's reported sweep
HORIZONS=("1x:81" "2x:161" "4x:321" "6x:481")

PROMPT=$(python -c "
import json
data = json.load(open('$PROMPTS_FILE'))
for p in data['prompts']:
    if p['tag'] == '$TAG':
        print(p['prompt'])
        break
")

for horizon_pair in "${HORIZONS[@]}"; do
  HORIZON_LABEL="${horizon_pair%%:*}"
  NUM_FRAMES="${horizon_pair##*:}"
  for BACKEND in dense lvsa-fi; do
    case "$BACKEND" in
      dense)   FLAGS="" ;;
      lvsa-fi) FLAGS="--lvsa --flashinfer --rotate-keyframes --auto-keyframes" ;;
    esac
    STEM="wan13b__${BACKEND}__${HORIZON_LABEL}__${TAG}"
    MP4="$OUTDIR/${STEM}.mp4"
    LOG="$OUTDIR/${STEM}.log"
    if [ -s "$MP4" ]; then
      echo "skip $STEM (exists)"
      continue
    fi
    echo "=== $STEM  frames=$NUM_FRAMES ==="
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

echo
echo "Done. Aggregate: python benchmarks/aggregate.py --outdir $OUTDIR"
