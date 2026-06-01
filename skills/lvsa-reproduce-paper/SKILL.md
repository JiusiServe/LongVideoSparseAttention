---
name: lvsa-reproduce-paper
description: Reproduce LVSA paper headline numbers using the bundled benchmarks/ scripts. Use when running the SotA comparison (5 prompts × 3 horizons × 4 methods), the latency-scaling sweep, scoring videos with VQeval + VBench-Long, or regenerating the figures embedded in the README.
---

# Reproducing LVSA paper numbers

## Setup

```bash
git clone https://github.com/JiusiServe/LongVideoSparseAttention
cd LVSA

uv venv --python 3.12
source .venv/bin/activate

# Install LVSA + scoring deps
uv pip install -e ".[diffusers,hunyuan,flashinfer,dev]"
uv pip install -e vqeval/

# For VBench-Long, you need a separate venv (it pins old diffusers/transformers)
git clone https://github.com/Vchitect/VBench /path/to/VBench
python3 -m venv /path/to/vbench-venv
source /path/to/vbench-venv/bin/activate
pip install -e /path/to/VBench
deactivate

source .venv/bin/activate  # back to LVSA venv

# Model weights (downloaded separately)
# Wan 2.1 1.3B: huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers
export MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers
```

## SotA grid (5 prompts × 3 horizons × 4 methods)

Generate 60 videos: Dense / RIFLEx / LVSA-SDPA / LVSA-FI × 165/249/333 frames × 5 prompts.

```bash
export OUTDIR=out/sota_comparison
bash benchmarks/sota_comparison.sh
```

Expected wall time per cell (single A100, 50 steps, seed 16):

| Method | 2× (165f) | 3× (249f) | 4× (333f) |
|---|---|---|---|
| Dense | 566 s | 1145 s | 1930 s |
| RIFLEx | 564 s | 1149 s | 1931 s |
| LVSA (SDPA) | 502 s | 796 s | 1021 s |
| **LVSA-FI** | **395 s** | **621 s** | **802 s** |

Total: ~9 hours on single A100, ~70 min on 8×A100 via GNU parallel.

**UltraViCo is excluded** — it lives in a separate repo (`thu-ml/DiT-Extrapolation`, branch `ultra-wan`) with a different CLI. See the paper appendix for the UltraViCo recipe.

## Latency scaling (single-prompt frame sweep)

For the headline 3.14× claim and the latency figure on the README:

```bash
export OUTDIR=out/latency_scaling
bash benchmarks/latency_scaling.sh
```

Frame counts swept: 81 (1×) / 161 (2×) / 321 (4×) / 481 (6×). Methods: Dense + LVSA-FI.

## Score with VQeval

```bash
bash benchmarks/score_vqeval.sh out/sota_comparison
# Writes <stem>.vqeval.json next to each mp4
```

VQeval scores 6 dimensions + composite. Single A100 + the bundled `vqeval/` subpackage. Expect ~15 min for 60 videos.

## Score with VBench-Long

```bash
VBENCH_REPO=/path/to/VBench \
VBENCH_PYTHON=/path/to/vbench-venv/bin/python \
    bash benchmarks/score_vbench.sh out/sota_comparison
# Writes <stem>.vbench.json next to each mp4
```

VBench-Long scores 5 dimensions: `subject_consistency`, `temporal_flickering`, `motion_smoothness`, `background_consistency`, `imaging_quality`.

## Aggregate

```bash
python benchmarks/aggregate.py --outdir out/sota_comparison
# Writes _summary.csv (60 rows) and _summary_means.csv (12 cells)
```

The aggregator walks the output directory, parses tags (`<model>__<backend>__<horizon>__<prompt>`), loads the per-video JSONs, and emits tidy + means CSVs.

## Regenerate figures

```bash
python benchmarks/generate_figures.py \
    --sota-csv     out/sota_comparison/_summary_means.csv \
    --scaling-csv  out/latency_scaling/_summary_means.csv \
    --outdir       docs/figures/
```

Produces 4 PNGs at 300 DPI:
- `latency_scaling.png` — Wan 1.3B Dense vs LVSA wall-time scaling
- `crossmodel_speedup.png` — speedup-vs-Dense bar chart
- `hv_latency_scaling.png` — HunyuanVideo wall-time scaling
- `sparsity_vs_frames.png` — per-query attended fraction by model

## Expected results (5-prompt mean)

### Wall time and speedup vs Dense

| Horizon | LVSA (SDPA) | LVSA-FI | LVSA-FI vs Dense |
|---|---|---|---|
| 2× | 502 s | 395 s | **1.43×** |
| 3× | 796 s | 621 s | **1.84×** |
| 4× | 1021 s | 802 s | **2.41×** |

### Speedup vs UltraViCo (from paper appendix data)

| Horizon | LVSA-FI vs UltraViCo |
|---|---|
| 2× | **1.88×** |
| 3× | **2.49×** |
| 4× | **3.27×** |

### Quality (VQeval composite, Δ vs Dense)

| Horizon | LVSA-FI Δ |
|---|---|
| 2× | +6.5 |
| 3× | +11.2 |
| 4× | +9.9 |

### Quality (VBench-Long imaging_quality, Δ vs Dense)

| Horizon | LVSA-FI Δ |
|---|---|
| 2× | +0.09 |
| 3× | +0.04 |
| 4× | +0.10 |

## Tips

- **Idempotency**: all scripts skip cells whose `.mp4` (or `.vqeval.json`, `.vbench.json`) already exists. Crash-and-resume works.
- **Seed**: 16 matches the paper. Change `SEED=<n>` to get a different RNG roll.
- **Resolution**: 480×832 is the default and matches the paper. Higher resolutions need `LVSA_PATCHES_PER_FRAME` set or VIDEO_HEIGHT/WIDTH for the vllm-omni plugin.
- **8× A100 parallel**: the dev repo at `scripts/paper_results/sota_job_runner.sh` ships a `flock`-queued GNU-parallel orchestrator. The pruned recipe in `benchmarks/` is single-GPU sequential.

## Adapting to other models

Change the example invocation in `benchmarks/sota_comparison.sh` from `examples/wan_generate.py` to `examples/hunyuan_generate.py` (and the `HORIZONS` arrays to HunyuanVideo's range: 65/129/193/257). The aggregator and figure scripts handle any model tag.
