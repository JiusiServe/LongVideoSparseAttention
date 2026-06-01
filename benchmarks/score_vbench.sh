#!/usr/bin/env bash
# Score all mp4s in a directory with VBench-Long and write `<stem>.vbench.json` next to each.
#
# Usage:
#   VBENCH_REPO=/path/to/VBench \
#   VBENCH_PYTHON=/path/to/vbench-venv/bin/python \
#       bash benchmarks/score_vbench.sh [OUTDIR]
#
# Requires: clone of [VBench](https://github.com/Vchitect/VBench) with the
# `VBenchLong` long-video sub-suite installed in its own venv (it pins old
# diffusers / transformers versions that conflict with LVSA's env).

set -euo pipefail

OUTDIR=${1:-${OUTDIR:-out/sota_comparison}}
[ -d "$OUTDIR" ] || { echo "OUTDIR not found: $OUTDIR" >&2; exit 1; }

VBENCH_REPO=${VBENCH_REPO:?"VBENCH_REPO must point at a clone of github.com/Vchitect/VBench"}
VBENCH_PYTHON=${VBENCH_PYTHON:-python}

# VBench dimensions to compute (the 5 used in the paper for long video)
DIMENSIONS=${DIMENSIONS:-"subject_consistency temporal_flickering motion_smoothness background_consistency imaging_quality"}

count=0
for mp4 in "$OUTDIR"/*.mp4; do
  [ -s "$mp4" ] || continue
  stem=$(basename "$mp4" .mp4)
  json_out="$OUTDIR/${stem}.vbench.json"
  if [ -s "$json_out" ]; then
    echo "skip $stem (vbench.json exists)"
    continue
  fi
  count=$((count + 1))
  echo "[$count] vbench $stem"
  pushd "$VBENCH_REPO" > /dev/null
    "$VBENCH_PYTHON" -m vbench evaluate \
      --videos_path "$mp4" \
      --dimension $DIMENSIONS \
      --output_path "$json_out" \
      2>&1 | tail -3
  popd > /dev/null
done

echo
echo "[done] scored $count videos with VBench-Long."
