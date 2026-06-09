#!/usr/bin/env bash
# Integration smoke-sweep for the LVSA vllm-omni plugin.
#
# Mirrors the standalone-engine sweep (`examples/wan_generate.py`-based) but
# exercises the plugin path: each cell either invokes the OFFLINE Python API
# (vllm_omni.entrypoints.omni.Omni) or spins up an ONLINE HTTP server +
# client.
#
# Each cell uses few inference steps (default 4) — the goal is "does the
# plugin engage correctly and produce an mp4?", not quality.
#
# Idempotent: skips cells whose output mp4 already exists. Crash-and-resume.
#
# Usage:
#   lvsa-vllm-omni/scripts/integration_sweep.sh                 # all cells
#   lvsa-vllm-omni/scripts/integration_sweep.sh wan13b          # by label substring
#   ONLY="01 02 09" lvsa-vllm-omni/scripts/integration_sweep.sh # by IDs
#
# Env overrides:
#   STEPS=4
#   SEED=16
#   PROMPT="A dog running in the forest."
#   OUTDIR=/tmp/lvsa_vllm_omni_sweep
#   CUDA_DEVICE=0
#   ONLINE_PORT=8198      # base port for ONLINE cells
#   SKIP_ONLINE=0         # set =1 to skip the HTTP-based cells

set -uo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

# This sweep targets the vllm-omni-pinned venv (`.venv-vllm`), not the
# standalone-engine venv (`.venv`).  The two venvs have incompatible torch
# pins: vllm-omni 0.22.0 (+ vllm 0.22.0) require torch 2.11 / CUDA 13 while
# the standalone engine uses torch 2.12.  See lvsa-vllm-omni/examples/README.md.
PYTHON_VENV=${PYTHON_VENV:-$ROOT/.venv-vllm}
PYTHON="$PYTHON_VENV/bin/python"
EXAMPLES="$ROOT/lvsa-vllm-omni/examples"

if [ ! -x "$PYTHON" ]; then
  echo "ERROR: $PYTHON not found." >&2
  echo "Create the vllm-omni venv first:" >&2
  echo "  uv venv .venv-vllm --python 3.12" >&2
  echo "  source .venv-vllm/bin/activate" >&2
  echo "  uv pip install -e . -e lvsa-vllm-omni/ \"vllm==0.22.0\"" >&2
  echo "  uv pip install --no-build-isolation \\" >&2
  echo "      'vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@v0.22.0'" >&2
  exit 1
fi

OUTDIR=${OUTDIR:-$ROOT/out/lvsa_vllm_omni_sweep}
STEPS=${STEPS:-4}
SEED=${SEED:-16}
PROMPT=${PROMPT:-"A dog running in the forest."}
CUDA_DEVICE=${CUDA_DEVICE:-0}
ONLINE_PORT=${ONLINE_PORT:-8198}
SKIP_ONLINE=${SKIP_ONLINE:-0}

# ─── Model checkpoints (override via env) ────────────────────────────────────
# Defaults match `/data/...` (the layout on the maintainer box).  External
# users should override these via env vars to point at their own checkpoints.
# A cell whose model path is empty or doesn't exist on disk is skipped
# gracefully — you can run a partial sweep with just one model.
WAN13B_PATH=${WAN13B_PATH:-/data/Wan2.1-T2V-1.3B-Diffusers}
WAN14B_PATH=${WAN14B_PATH:-/data/Wan2.1-T2V-14B-Diffusers}
HV15_PATH=${HV15_PATH:-/data/HunyuanVideo-1.5-Diffusers-480p_t2v}

mkdir -p "$OUTDIR"
# SUMMARY can be overridden so multiple parallel orchestrators (one per GPU)
# don't clobber each other's per-instance status file.
SUMMARY=${SUMMARY:-$OUTDIR/_summary.txt}

# ─── Sweep matrix ────────────────────────────────────────────────────────────
# Each entry: ID|LABEL|MODE|FAMILY|MODEL_PATH|SCRIPT|NUM_FRAMES|EXTRA_FLAGS|EXTRA_ENV
#   MODE:   offline | online
#   FAMILY: wan | hunyuan | cosmos — required by offline_lvsa.py; ignored for
#           online cells (the serve_*.sh wrappers set their own family env).
#   EXTRA_FLAGS: offline_lvsa.py flags. Backend: omit = flashinfer (LVSA),
#           "--backend sdpa" = LVSA via SDPA, "--backend dense" = NO LVSA
#           (baseline). Rotation is ON by default; "--no-rotate" disables it.
#           "--sparsity-scale F" sets LVSA_SPARSITY_SCALE (<1 = more sparse).
#   EXTRA_ENV: pipe-separated KEY=VAL pairs applied just for this cell
# NOTE: migrated from the retired per-family offline_wan.py/offline_hunyuan.py to
# the unified offline_lvsa.py. Cell 05 adds SDPA-backend coverage (the other LVSA
# path), which the per-family scripts didn't smoke-test.
ROWS=(
  # ─ Offline cells (all via the unified offline_lvsa.py) ─
  "01|Wan 1.3B offline LVSA no-rotate     |offline|wan    |$WAN13B_PATH|offline_lvsa.py| 81|--no-rotate        |"
  "02|Wan 1.3B offline LVSA + rotate      |offline|wan    |$WAN13B_PATH|offline_lvsa.py| 81|                   |"
  "03|Wan 1.3B offline DENSE baseline     |offline|wan    |$WAN13B_PATH|offline_lvsa.py| 81|--backend dense    |"
  "04|Wan 1.3B offline LVSA + sparsity 0.5|offline|wan    |$WAN13B_PATH|offline_lvsa.py| 81|--sparsity-scale 0.5|"
  "05|Wan 1.3B offline LVSA SDPA backend  |offline|wan    |$WAN13B_PATH|offline_lvsa.py| 81|--backend sdpa     |"
  "06|Wan 1.3B offline LVSA 2x extension  |offline|wan    |$WAN13B_PATH|offline_lvsa.py|161|                   |"
  "07|Wan 14B  offline LVSA               |offline|wan    |$WAN14B_PATH|offline_lvsa.py| 81|                   |"
  "08|HV 1.5   offline DENSE baseline     |offline|hunyuan|$HV15_PATH  |offline_lvsa.py|129|--backend dense    |"
  "09|HV 1.5   offline LVSA               |offline|hunyuan|$HV15_PATH  |offline_lvsa.py|129|                   |"
  # ─ Online cells (real HTTP server + client roundtrip) ─
  "10|Wan 1.3B ONLINE LVSA (HTTP roundtrip)|online|wan    |$WAN13B_PATH|serve_wan.sh    | 81|                   |"
  "11|HV 1.5   ONLINE LVSA (HTTP roundtrip)|online|hunyuan|$HV15_PATH  |serve_hunyuan.sh|129|                   |"
)

# ─── Filtering ───────────────────────────────────────────────────────────────
FILTER="${1:-}"
ONLY=${ONLY:-}
should_run() {
  local id="$1" label="$2"
  if [ -n "$ONLY" ]; then
    for x in $ONLY; do [ "$id" = "$x" ] && return 0; done
    return 1
  fi
  if [ -n "$FILTER" ]; then
    [[ "$label" == *"$FILTER"* ]] && return 0 || return 1
  fi
  return 0
}

# ─── Header ──────────────────────────────────────────────────────────────────
{
  echo "=== LVSA vllm-omni integration sweep started $(date '+%F %T') ==="
  echo "steps=$STEPS seed=$SEED outdir=$OUTDIR cuda=$CUDA_DEVICE online_port=$ONLINE_PORT"
  echo "skip_online=$SKIP_ONLINE"
  echo "models:"
  echo "  WAN13B_PATH=${WAN13B_PATH:-<unset>}"
  echo "  WAN14B_PATH=${WAN14B_PATH:-<unset>}"
  echo "  HV15_PATH=${HV15_PATH:-<unset>}"
  echo
} | tee "$SUMMARY"

# Warn if no models are configured at all
if [ -z "$WAN13B_PATH" ] && [ -z "$WAN14B_PATH" ] && [ -z "$HV15_PATH" ]; then
  echo "WARNING: no model paths configured (WAN13B_PATH / WAN14B_PATH / HV15_PATH)." | tee -a "$SUMMARY"
  echo "All cells will be skipped. See lvsa-vllm-omni/scripts/README.md." | tee -a "$SUMMARY"
fi

# ─── Helper: wait for HTTP server ready ──────────────────────────────────────
wait_for_server() {
  local host="$1" port="$2" timeout="$3"
  local t0=$(date +%s)
  while true; do
    if curl -sS -o /dev/null -m 2 "http://$host:$port/v1/models" 2>/dev/null; then
      return 0
    fi
    if [ $(( $(date +%s) - t0 )) -gt $timeout ]; then
      echo "TIMEOUT waiting for $host:$port"
      return 1
    fi
    sleep 5
  done
}

# ─── Run loop ────────────────────────────────────────────────────────────────
for row in "${ROWS[@]}"; do
  IFS='|' read -r ID LABEL MODE FAMILY MODEL_PATH SCRIPT NUM_FRAMES FLAGS EXTRA_ENV <<< "$row"
  ID=$(echo "$ID"|tr -d ' '); LABEL=$(echo "$LABEL"|sed 's/  *$//')
  MODE=$(echo "$MODE"|tr -d ' '); FAMILY=$(echo "$FAMILY"|tr -d ' ')
  MODEL_PATH=$(echo "$MODEL_PATH"|tr -d ' ')
  SCRIPT=$(echo "$SCRIPT"|tr -d ' '); NUM_FRAMES=$(echo "$NUM_FRAMES"|tr -d ' ')
  FLAGS=$(echo "$FLAGS" | sed -e 's/^ *//' -e 's/ *$//')

  should_run "$ID" "$LABEL" || { echo "[$ID] SKIP filtered: $LABEL"; continue; }
  if [ "$MODE" = "online" ] && [ "$SKIP_ONLINE" = "1" ]; then
    echo "[$ID] SKIP online (SKIP_ONLINE=1): $LABEL"; continue
  fi

  STEM=$(echo "${ID}_$LABEL" | tr ' /+()' '______' | tr -d '[:punct:]' | tr -s '_')
  MP4="$OUTDIR/${STEM}.mp4"
  LOG="$OUTDIR/${STEM}.log"

  if [ -s "$MP4" ]; then
    echo "[$ID] SKIP (mp4 exists): $LABEL" | tee -a "$SUMMARY"; continue
  fi
  if [ ! -d "$MODEL_PATH" ]; then
    echo "[$ID] SKIP (model missing): $MODEL_PATH" | tee -a "$SUMMARY"; continue
  fi

  t0=$(date +%s)
  echo "[$ID] START $(date '+%T')  ($MODE)  $LABEL"
  echo "       model=$MODEL_PATH frames=$NUM_FRAMES flags=[$FLAGS]"

  rc=0
  case "$MODE" in
    offline)
      CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON" "$EXAMPLES/$SCRIPT" \
        --family "$FAMILY" \
        --model "$MODEL_PATH" \
        --prompt "$PROMPT" \
        --num-frames "$NUM_FRAMES" \
        --steps "$STEPS" --seed "$SEED" \
        --output-dir "$(dirname "$MP4")" --output-name "$(basename "$MP4")" \
        $FLAGS \
        > "$LOG" 2>&1
      rc=$?
      ;;

    online)
      PORT=$ONLINE_PORT
      SERVE_LOG="$OUTDIR/${STEM}_server.log"
      # 1. spawn server — pass PYTHON explicitly so the serve_*.sh wrapper
      # uses the venv interpreter (system has only `python3`; the wrapper's
      # `exec python` would otherwise fail with "command not found").
      CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PORT="$PORT" PYTHON="$PYTHON" \
        bash "$EXAMPLES/$SCRIPT" "$MODEL_PATH" > "$SERVE_LOG" 2>&1 &
      SERVER_PID=$!
      # 2. wait for server ready (max 15 min)
      if wait_for_server localhost "$PORT" 900; then
        # 3. POST + poll + download
        "$PYTHON" "$EXAMPLES/online_client.py" \
          --host "localhost:$PORT" \
          --prompt "$PROMPT" \
          --num-frames "$NUM_FRAMES" \
          --steps "$STEPS" --seed "$SEED" \
          --output "$MP4" \
          > "$LOG" 2>&1
        rc=$?
      else
        echo "Server failed to come up (see $SERVE_LOG)" > "$LOG"
        rc=99
      fi
      # 4. shutdown server
      kill "$SERVER_PID" 2>/dev/null || true
      wait "$SERVER_PID" 2>/dev/null || true
      # bump port so consecutive online cells don't race
      ONLINE_PORT=$(( ONLINE_PORT + 1 ))
      ;;

    *)
      echo "[$ID] FAIL unknown mode '$MODE'" | tee -a "$SUMMARY"
      continue
      ;;
  esac

  dt=$(( $(date +%s) - t0 ))

  if [ $rc -ne 0 ]; then
    msg="[$ID] FAIL rc=$rc dt=${dt}s — $LABEL  (see $LOG)"
  elif [ ! -s "$MP4" ]; then
    msg="[$ID] FAIL no mp4 dt=${dt}s — $LABEL  (see $LOG)"
  else
    # Confirm LVSA dispatch line in log if LVSA expected
    if echo "$FLAGS" | grep -q -- '--backend dense'; then
      msg="[$ID] OK   dt=${dt}s  (dense)            $LABEL"
    else
      # Pull the n_blocks value from the LVSA calibration line (the backend prints
      # "[LVSA] Step counter auto-calibrated: n_blocks=N"; the cosmos hook emits a
      # similar n_blocks=N). Multiple files can contain it — take the first match
      # from the cell log, or the server log for online cells.
      files=("$LOG")
      [ -n "${SERVE_LOG:-}" ] && [ -f "$SERVE_LOG" ] && files+=("$SERVE_LOG")
      n_blocks=$(grep -hE 'n_blocks=[0-9]' "${files[@]}" 2>/dev/null \
                 | head -n 1 | sed -nE 's/.*n_blocks=([0-9]+).*/\1/p')
      [ -z "$n_blocks" ] && n_blocks="?"
      msg="[$ID] OK   dt=${dt}s  lvsa_blocks=$n_blocks  $LABEL"
    fi
  fi
  echo "$msg" | tee -a "$SUMMARY"
done

echo | tee -a "$SUMMARY"
echo "=== sweep finished $(date '+%F %T') ===" | tee -a "$SUMMARY"
echo "Logs + mp4s in $OUTDIR"
