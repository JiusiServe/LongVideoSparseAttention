#!/usr/bin/env bash
# Score all mp4s in a directory with VQeval and write `<stem>.vqeval.json` next to each.
#
# Usage:
#   bash benchmarks/score_vqeval.sh [OUTDIR]
#
# Requires: `pip install -e vqeval/` (the bundled subpackage).

set -euo pipefail

OUTDIR=${1:-${OUTDIR:-out/sota_comparison}}
[ -d "$OUTDIR" ] || { echo "OUTDIR not found: $OUTDIR" >&2; exit 1; }

PROMPTS_FILE=${PROMPTS_FILE:-benchmarks/long_prompts.json}

# Helper to extract prompt text by tag from the canonical mp4 stem
prompt_for_stem() {
  local stem="$1"
  local tag="${stem##*__}"   # last __-separated field
  python -c "
import json
data = json.load(open('$PROMPTS_FILE'))
for p in data['prompts']:
    if p['tag'] == '$tag':
        print(p['prompt'])
        break
"
}

count=0
for mp4 in "$OUTDIR"/*.mp4; do
  [ -s "$mp4" ] || continue
  stem=$(basename "$mp4" .mp4)
  json_out="$OUTDIR/${stem}.vqeval.json"
  if [ -s "$json_out" ]; then
    echo "skip $stem (vqeval.json exists)"
    continue
  fi
  prompt=$(prompt_for_stem "$stem")
  count=$((count + 1))
  echo "[$count] vqeval $stem"
  vqeval evaluate "$mp4" \
    --prompt "$prompt" \
    --output "$json_out" \
    2>&1 | tail -3
done

echo
echo "[done] scored $count videos with VQeval."
