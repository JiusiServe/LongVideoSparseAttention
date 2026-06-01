# Figures

Publication-quality figures referenced from the project README and paper. All PNGs are 300 DPI.

| File | Topic |
|---|---|
| `latency_scaling.png` | Wan 2.1 1.3B wall-time vs frame count; Dense vs LVSA |
| `crossmodel_speedup.png` | Cross-model speedup summary (Wan + HunyuanVideo) |
| `hv_latency_scaling.png` | HunyuanVideo 1.5 wall-time vs frame count; Dense vs LVSA |
| `sparsity_vs_frames.png` | Per-query sparsity as a function of `T_lat`, by model |

## Regenerating

Figures are regenerated from `_summary_means.csv` files produced by the benchmark pipeline:

```bash
python benchmarks/generate_figures.py --outdir docs/figures/
```

See [`../../benchmarks/README.md`](../../benchmarks/README.md) for the data flow.
