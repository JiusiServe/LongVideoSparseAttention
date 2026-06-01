# Benchmarks — paper reproduction recipe

Recipe for reproducing the headline numbers from the paper. Outputs land under `out/` (gitignored).

## Pipeline

```
sota_comparison.sh / latency_scaling.sh    →  *.mp4 + *.log
score_vqeval.sh                            →  *.vqeval.json
score_vbench.sh                            →  *.vbench.json
python aggregate.py --outdir <OUTDIR>      →  _summary.csv + _summary_means.csv
python generate_figures.py                 →  PNGs in docs/figures/
```

## Hardware + setup

- The paper numbers were measured on **8× A100 80GB** for the SotA grid (each cell parallel-distributed) and **single A100** for the latency-scaling sweep. Single-GPU reproduction is supported by all scripts; expect proportionally longer wall times.
- LVSA-FI requires CUDA (FlashInfer is CUDA-only). The SDPA path works on CUDA and Ascend NPU.
- VBench-Long requires a separate Python environment because of pinned diffusers/transformers versions that conflict with LVSA's. See `score_vbench.sh` for the env vars.

## Files in this directory

| File | Purpose |
|---|---|
| [`long_prompts.json`](long_prompts.json) | 5 canonical descriptive prompts (~470 tokens each) — the SotA prompt set |
| [`sota_comparison.sh`](sota_comparison.sh) | 5 prompts × 3 horizons (165 / 249 / 333 frames) × 4 methods (Dense / RIFLEx / LVSA-SDPA / LVSA-FI). Wan 1.3B, seed 16. |
| [`latency_scaling.sh`](latency_scaling.sh) | Single-prompt frame-count sweep (81 / 161 / 321 / 481 frames). Used for the headline 3.14× claim. |
| [`score_vqeval.sh`](score_vqeval.sh) | VQeval scoring wrapper (uses the bundled `vqeval/` subpackage) |
| [`score_vbench.sh`](score_vbench.sh) | VBench-Long scoring wrapper (requires external VBench install) |
| [`aggregate.py`](aggregate.py) | Walks an output dir, parses `<tag>.{vqeval,vbench}.json` + logs, writes tidy + means CSVs |
| [`generate_figures.py`](generate_figures.py) | Regenerates the 4 PNG figures in `docs/figures/` from aggregated CSVs |

## SotA comparison (5 methods × 3 horizons × 5 prompts)

The headline grid. **UltraViCo is excluded from this script** because it lives in a separate repo (`thu-ml/DiT-Extrapolation`, branch `ultra-wan`) and the calling convention is different — see the paper's appendix for the UltraViCo recipe.

```bash
export MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers
export OUTDIR=out/sota_comparison

# Generate 60 videos (~9 hours on single A100; ~70 min on 8×A100 via GNU parallel)
bash benchmarks/sota_comparison.sh

# Score with VQeval (single-GPU; ~15 min for 60 videos)
bash benchmarks/score_vqeval.sh $OUTDIR

# Score with VBench-Long (needs separate VBench venv)
VBENCH_REPO=/path/to/VBench \
VBENCH_PYTHON=/path/to/vbench-venv/bin/python \
    bash benchmarks/score_vbench.sh $OUTDIR

# Aggregate
python benchmarks/aggregate.py --outdir $OUTDIR

# Regenerate figures
python benchmarks/generate_figures.py \
    --sota-csv $OUTDIR/_summary_means.csv \
    --outdir docs/figures/
```

Expected wall times per cell (single A100):

| Method | 2× (165f) | 3× (249f) | 4× (333f) |
|---|---|---|---|
| Dense | 566 s | 1145 s | 1930 s |
| RIFLEx | 564 s | 1149 s | 1931 s |
| LVSA (SDPA) | 502 s | 796 s | 1021 s |
| **LVSA-FI** | **395 s** | **621 s** | **802 s** |

## Latency scaling (single prompt, frame sweep)

For the 3.14× headline claim and the figure on the README:

```bash
export MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers
export OUTDIR=out/latency_scaling
bash benchmarks/latency_scaling.sh
python benchmarks/aggregate.py --outdir $OUTDIR
```

## Reproducing the figures

The figures embedded in the README and the paper are regenerated from the aggregated CSVs:

```bash
python benchmarks/generate_figures.py \
    --sota-csv     out/sota_comparison/_summary_means.csv \
    --scaling-csv  out/latency_scaling/_summary_means.csv \
    --outdir       docs/figures/
```

Each figure is skipped if its source data isn't available; both CSVs are optional.

## Adapting to other models

The scripts assume Wan 2.1 1.3B. To reproduce on HunyuanVideo 1.5 or Wan 14B, change the per-method invocation to call `examples/hunyuan_generate.py` or `examples/wan_generate.py` with the 14B path, and update the `HORIZONS` arrays to match the model's training reference (33 for HV, 21 for Wan). The aggregator and figure scripts handle any model tag.

## Notes

- All scripts are **idempotent**: rerunning skips cells whose `.mp4` (or `.vqeval.json` / `.vbench.json`) already exists.
- Seed 16 matches the paper. Change `SEED=<n>` to reproduce with different randomness.
- For 8-GPU parallel sweeps (paper config), use GNU parallel with `flock`-based queue management — the reference orchestrator lives in the dev repo under `scripts/paper_results/sota_job_runner.sh`. The pruned recipe here is single-GPU sequential.
