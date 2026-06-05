# `lvsa-vllm-omni/scripts/`

Maintainer / contributor scripts for verifying the vllm-omni plugin against
real model checkpoints on real hardware. These are **not** required for
day-to-day use of the plugin — they exist so that anyone (you, a contributor,
or a CI runner with the right machine spec) can reproduce the integration
QA we run before tagging a release.

## Files

| Script | What it does |
|---|---|
| [`integration_sweep.sh`](integration_sweep.sh) | 10-cell smoke sweep across Wan 1.3B / Wan 14B / HunyuanVideo 1.5, offline + online modes, dense baseline + LVSA variants. |

## Prerequisites

- The `lvsa-vllm-omni` plugin venv with `vllm==0.22.0` (PyPI) and
  `vllm-omni==0.22.0rc1` (built from git — not on PyPI; versions intentionally
  differ). See [`../examples/README.md`](../examples/README.md) for the setup.
  The sweep expects `$ROOT/.venv-vllm/bin/python` by default; override with
  `PYTHON_VENV=...`.
- A GPU with ≥ 24 GB VRAM for Wan 1.3B; ≥ 80 GB for Wan 14B or HunyuanVideo at ≥ 1.5×.
- Local model checkpoints (HuggingFace `*-Diffusers` snapshots). Provide them via
  the env vars below — cells whose model path is empty or missing are skipped
  gracefully, so you can run a partial sweep with just one model.

## Usage

```bash
# Full sweep (all 10 cells)
WAN13B_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers \
WAN14B_PATH=/path/to/Wan2.1-T2V-14B-Diffusers \
HV15_PATH=/path/to/HunyuanVideo-1.5-Diffusers-480p_t2v \
    bash lvsa-vllm-omni/scripts/integration_sweep.sh

# Subset by label substring (matches the LABEL column in the matrix)
WAN13B_PATH=... bash lvsa-vllm-omni/scripts/integration_sweep.sh wan13b
WAN13B_PATH=... bash lvsa-vllm-omni/scripts/integration_sweep.sh HV

# Specific cell IDs
WAN13B_PATH=... ONLY="01 03" bash lvsa-vllm-omni/scripts/integration_sweep.sh

# Skip the (slow) online HTTP cells
SKIP_ONLINE=1 WAN13B_PATH=... bash lvsa-vllm-omni/scripts/integration_sweep.sh

# Run on GPU 1 instead of 0
CUDA_DEVICE=1 WAN13B_PATH=... bash lvsa-vllm-omni/scripts/integration_sweep.sh

# More steps per cell (default 4 — just a smoke). Bump for real generation.
STEPS=20 WAN13B_PATH=... bash lvsa-vllm-omni/scripts/integration_sweep.sh
```

## Env-var contract

| Var | Default | Meaning |
|---|---|---|
| `WAN13B_PATH` | `/data/Wan2.1-T2V-1.3B-Diffusers` | Local path to a `Wan2.1-T2V-1.3B-Diffusers` checkpoint |
| `WAN14B_PATH` | `/data/Wan2.1-T2V-14B-Diffusers` | Local path to a `Wan2.1-T2V-14B-Diffusers` checkpoint |
| `HV15_PATH` | `/data/HunyuanVideo-1.5-Diffusers-480p_t2v` | Local path to a `HunyuanVideo-1.5-Diffusers-480p_t2v` checkpoint |

> The defaults match the maintainer-box layout. External users almost certainly
> need to override them.
| `PYTHON_VENV` | `$REPO/.venv-vllm` | Path to the vllm-omni venv root |
| `OUTDIR` | `$REPO/out/lvsa_vllm_omni_sweep` | Where mp4s + logs land (gitignored) |
| `STEPS` | `4` | Denoising steps per cell. 4 is "does it run?"; 40-50 is real generation. |
| `SEED` | `16` | RNG seed |
| `PROMPT` | `"A dog running in the forest."` | Generation prompt |
| `CUDA_DEVICE` | `0` | GPU index for offline cells |
| `ONLINE_PORT` | `8198` | Base port for online HTTP cells (bumped per cell to avoid collisions) |
| `SKIP_ONLINE` | `0` | Set to `1` to skip the 2 online HTTP cells |
| `ONLY` | _unset_ | Space-separated cell IDs (e.g. `"01 03 05"`); takes precedence over the label filter |

## What gets produced

```
$OUTDIR/
├── _summary.txt                    one line per cell with status + timing
├── 01_Wan13Bofflinelvsabasic.mp4   per-cell mp4 output
├── 01_Wan13Bofflinelvsabasic.log   per-cell stdout/stderr
├── 09_Wan13BONLINELVSAHTTProundtripserver.log   server log for online cells
└── …
```

Idempotent: rerunning skips cells whose mp4 already exists. Safe to ctrl-C and resume.

## Wall-time expectations

At default `STEPS=4`, on a single A100 80GB:

| Cells | Approx wall |
|---|---|
| Wan 1.3B (offline, first cell) | ~3-4 min (model load) |
| Wan 1.3B (subsequent offline) | ~30-60 s |
| Wan 14B (offline) | ~12-15 min (load + 4 steps) |
| HunyuanVideo 1.5 (offline) | ~10-12 min |
| Online HTTP cells | ~10-15 min each (server startup + request) |
| **Full 10-cell sweep** | **~1-2 h** |

At `STEPS=50` (real generation), multiply step times by ~12×.

## Interpreting the summary

Each line in `_summary.txt` is one of:

- `[NN] OK   dt=Xs lvsa_blocks=N  LABEL` — cell completed; LVSA dispatched on N attention blocks
- `[NN] OK   dt=Xs  (dense)        LABEL` — cell completed; LVSA disabled (dense baseline)
- `[NN] FAIL rc=N dt=Xs — LABEL  (see /path/to/log)` — cell exited non-zero
- `[NN] SKIP filtered: LABEL` — filtered out by `$ONLY` or label-substring arg
- `[NN] SKIP (model missing): /path/to/model` — `WAN13B_PATH` etc. unset or doesn't exist
- `[NN] SKIP (mp4 exists): LABEL` — idempotent skip

The most important check: every `OK` line with a non-`(dense)` config should have `lvsa_blocks=N > 0`. If a cell reports `OK` but `lvsa_blocks=?` or `lvsa_blocks=0`, that means the mp4 was produced but the LVSA hook silently didn't engage — investigate `[LVSA-FALLBACK]` warnings in the log.

## Not in this sweep

- **Wan 2.2 A14B (MoE)** is excluded — at ~118 GB checkpoint it doesn't fit on a single A100 and needs multi-GPU sharding to run. Add a custom cell or use the offline script with `--tensor-parallel-size 2` for that case.
- **Non-default resolutions** (720p, etc.) — the sweep uses the defaults of the example scripts. For 720p coverage, drive the offline scripts directly with `--height 720 --width 1280` and set `LVSA_VIDEO_HEIGHT/WIDTH` env vars.

## Adding a cell

Edit the `ROWS=(...)` array in `integration_sweep.sh`. Each row is pipe-separated:

```
ID|LABEL|MODE|MODEL_PATH|SCRIPT|NUM_FRAMES|EXTRA_FLAGS|EXTRA_ENV
```

`SCRIPT` is the basename of a file in `lvsa-vllm-omni/examples/` (e.g. `offline_wan.py`, `serve_hunyuan.sh`).
