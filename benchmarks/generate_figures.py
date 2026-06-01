#!/usr/bin/env python3
"""Regenerate the four headline PNG figures from `_summary_means.csv` files.

Reads aggregated CSVs produced by `aggregate.py` and emits:
  - latency_scaling.png      Wan 1.3B wall-time vs frame count
  - crossmodel_speedup.png   Cross-model speedup summary
  - hv_latency_scaling.png   HunyuanVideo wall-time vs frame count
  - sparsity_vs_frames.png   Per-query attended fraction vs T_lat

Usage:
    python benchmarks/generate_figures.py \\
        --sota-csv  out/sota_comparison/_summary_means.csv \\
        --scaling-csv out/latency_scaling/_summary_means.csv \\
        --outdir docs/figures/

Both CSVs are optional individually — figures that don't have their data
source will be skipped with a note.
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib required. Install with `pip install matplotlib`.", file=sys.stderr)
    sys.exit(1)


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def f1(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def fig_latency_scaling(rows, model_tag, title, outpath):
    """Wall-time per video vs frame count, by method."""
    if not rows:
        print(f"  skip {outpath.name}: no data")
        return
    # Pivot: backend -> [(num_frames, wall_time_s)]
    horizon_to_frames = {  # canonical mapping
        "0.5x": 41, "1x": 81, "1.5x": 121, "2x": 161, "2x_sota": 165,
        "3x": 241, "3x_sota": 249, "4x": 321, "4x_sota": 333, "6x": 481,
    }
    # Use the canonical mapping where possible; fall back to numeric extraction.
    def horizon_to_n(h, model_tag):
        if model_tag.startswith("hv"):
            return {"0.5x": 65, "1x": 129, "1.5x": 193, "2x": 257}.get(h)
        return {"1x": 81, "2x": 161, "4x": 321, "6x": 481,
                "2x_sota": 165, "3x_sota": 249, "4x_sota": 333}.get(h)
    series = {}
    for r in rows:
        if r["model"] != model_tag:
            continue
        b = r["backend"]
        n = horizon_to_n(r["horizon"], model_tag)
        w = f1(r.get("wall_time_s_mean"))
        if n is None or w is None:
            continue
        series.setdefault(b, []).append((n, w))
    if not series:
        print(f"  skip {outpath.name}: no rows match model='{model_tag}'")
        return
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    style = {
        "dense": dict(marker="o", color="#444"),
        "riflex": dict(marker="s", color="#888"),
        "lvsa": dict(marker="^", color="#1f77b4"),
        "lvsa-fi": dict(marker="D", color="#d62728"),
    }
    for b in ["dense", "riflex", "lvsa", "lvsa-fi"]:
        if b not in series:
            continue
        pts = sorted(series[b])
        xs, ys = zip(*pts)
        ax.plot(xs, ys, label=b, linewidth=2, markersize=6, **style.get(b, {}))
    ax.set_xlabel("Frames")
    ax.set_ylabel("Wall time (s) / video")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


def fig_crossmodel_speedup(rows_sota, rows_scaling, outpath):
    """Speedup vs Dense, one subplot per model, grouped bars by horizon.

    Layout: three side-by-side panels (Wan 1.3B / Wan 14B / HunyuanVideo)
    sharing a single Y-axis ("Speedup vs Dense"). Each panel has groups of
    three bars per horizon: Dense (1.0× reference), LVSA (SDPA), LVSA-FI.
    Legend lives above the panels so it can't collide with bars.

    Special case: when a horizon has LVSA / LVSA-FI measurements but no
    Dense baseline (Dense OOMs at that length), the Dense bar is drawn
    full-height with diagonal hatching and an "OOM" label. The LVSA bars
    at that horizon are drawn at the global y-max, annotated with their
    absolute wall time (in seconds) since speedup-vs-Dense is undefined.
    """
    # Collect wall-time means keyed by (model, backend, horizon)
    by_cell = {}
    for r in (rows_sota or []) + (rows_scaling or []):
        w = f1(r.get("wall_time_s_mean"))
        if w is None or w <= 0:
            continue
        by_cell[(r["model"], r["backend"], r["horizon"])] = w

    # Build {model: {horizon: {"dense_oom": bool, "lvsa": (speedup_or_None, wall),
    #                          "lvsa-fi": (speedup_or_None, wall)}}}
    per_model = {}
    # Pre-pass: discover all (model, horizon) where lvsa or lvsa-fi has data
    cells_with_lvsa = set()
    for (model, backend, horizon), w in by_cell.items():
        if backend in ("lvsa", "lvsa-fi"):
            cells_with_lvsa.add((model, horizon))

    for (model, horizon) in cells_with_lvsa:
        dense_w = by_cell.get((model, "dense", horizon))
        entry = {"dense_oom": dense_w is None or dense_w <= 0}
        for b in ("lvsa", "lvsa-fi"):
            wall = by_cell.get((model, b, horizon))
            if wall is None or wall <= 0:
                entry[b] = (None, None)
                continue
            if entry["dense_oom"]:
                entry[b] = (None, wall)  # no speedup, just wall time
            else:
                entry[b] = (dense_w / wall, wall)
        per_model.setdefault(model, {})[horizon] = entry

    if not per_model:
        print(f"  skip {outpath.name}: no dense/lvsa pairs found")
        return

    # Canonical model display order + titles
    model_order = [
        ("wan13b", "Wan 2.1 1.3B"),
        ("wan14b", "Wan 2.1 14B"),
        ("hv15",   "HunyuanVideo 1.5"),
    ]
    models = [(k, t) for (k, t) in model_order if k in per_model]
    n = len(models)
    if n == 0:
        print(f"  skip {outpath.name}: no recognized models")
        return

    # Canonical horizon order (sorted by numeric ratio)
    def horizon_key(h: str) -> float:
        try:
            return float(h.rstrip("x"))
        except ValueError:
            return 0.0

    # Compute global ylim across all panels for a shared scale.
    all_speedups = []
    for d in per_model.values():
        for h in d.values():
            for b in ("lvsa", "lvsa-fi"):
                s = h[b][0]
                if s is not None:
                    all_speedups.append(s)
    ymax = max([*all_speedups, 1.0]) * 1.25

    fig, axes = plt.subplots(1, n, figsize=(3.6 * n + 1, 4.6), dpi=300,
                             sharey=True, gridspec_kw={"wspace": 0.18})
    if n == 1:
        axes = [axes]

    colors = {"dense": "#888888", "lvsa": "#1f77b4", "lvsa-fi": "#d62728"}
    labels = {"dense": "Dense", "lvsa": "LVSA (SDPA)", "lvsa-fi": "LVSA-FI"}
    bar_width = 0.27

    for ax, (mkey, mtitle) in zip(axes, models):
        horizons = sorted(per_model[mkey].keys(), key=horizon_key)
        x = list(range(len(horizons)))

        for i, h in enumerate(horizons):
            cell = per_model[mkey][h]
            xc = i
            # Dense bar
            if cell["dense_oom"]:
                ax.bar(xc - bar_width, ymax, bar_width, color=colors["dense"],
                       hatch="///", alpha=0.55, edgecolor=colors["dense"], linewidth=0)
                ax.text(xc - bar_width, ymax * 0.5, "OOM",
                        ha="center", va="center", fontsize=8, rotation=90,
                        color="black", fontweight="bold")
            else:
                ax.bar(xc - bar_width, 1.0, bar_width, color=colors["dense"])
            # LVSA bar
            sv, wv = cell.get("lvsa", (None, None))
            if sv is not None:
                ax.bar(xc, sv, bar_width, color=colors["lvsa"])
                ax.text(xc, sv + ymax * 0.012, f"{sv:.2f}×",
                        ha="center", va="bottom", fontsize=8, rotation=90,
                        color=colors["lvsa"])
            elif wv is not None and cell["dense_oom"]:
                # No comparable speedup — show "ran" at full height + wall time
                ax.bar(xc, ymax, bar_width, color=colors["lvsa"], alpha=0.45,
                       edgecolor=colors["lvsa"], linewidth=0)
                ax.text(xc, ymax * 0.5, f"{wv:.0f} s",
                        ha="center", va="center", fontsize=8, rotation=90,
                        color="white", fontweight="bold")
            # LVSA-FI bar
            sv, wv = cell.get("lvsa-fi", (None, None))
            if sv is not None:
                ax.bar(xc + bar_width, sv, bar_width, color=colors["lvsa-fi"])
                ax.text(xc + bar_width, sv + ymax * 0.012, f"{sv:.2f}×",
                        ha="center", va="bottom", fontsize=8, rotation=90,
                        color=colors["lvsa-fi"])
            elif wv is not None and cell["dense_oom"]:
                ax.bar(xc + bar_width, ymax, bar_width, color=colors["lvsa-fi"], alpha=0.45,
                       edgecolor=colors["lvsa-fi"], linewidth=0)
                ax.text(xc + bar_width, ymax * 0.5, f"{wv:.0f} s",
                        ha="center", va="center", fontsize=8, rotation=90,
                        color="white", fontweight="bold")

        ax.axhline(1.0, color="#aaa", linestyle=":", linewidth=0.7, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(horizons, fontsize=10)
        ax.set_xlabel("Horizon", fontsize=10)
        ax.set_title(mtitle, fontsize=11)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Speedup vs Dense (×)", fontsize=11)

    # Shared legend at top, horizontal
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[b]) for b in ("dense", "lvsa", "lvsa-fi")]
    fig.legend(handles, [labels[b] for b in ("dense", "lvsa", "lvsa-fi")],
               loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=3, frameon=False, fontsize=10)

    # Footnote explaining the OOM rendering, only if any OOM cell exists
    any_oom = any(c["dense_oom"] for d in per_model.values() for c in d.values())
    if any_oom:
        fig.text(0.5, -0.04,
                 "Hatched bar = Dense OOM on single A100 80GB. "
                 "LVSA / LVSA-FI bars in that column show absolute wall time (seconds).",
                 ha="center", va="top", fontsize=8, style="italic", color="#444")

    fig.suptitle("Cross-model speedup", y=1.09, fontsize=12, fontweight="bold")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


def fig_sparsity_vs_frames(outpath):
    """Per-query attended fraction vs T_lat for both models. This is
    derivable from the auto-keyframe scheduler — no measurement needed."""
    try:
        from lvsa.sparse_attention import compute_auto_kfi, compute_global_indices, get_window_bounds
    except ImportError:
        print(f"  skip {outpath.name}: lvsa not importable")
        return

    def sparsity_curve(ref, W=4, n_first=1):
        Ts = list(range(ref - 4, ref * 6 + 1, 4))
        out = []
        for T in Ts:
            if T <= 0:
                continue
            kfi = compute_auto_kfi(T, W, n_first, reference_frames=ref, sparsity_scale=1.0)
            gset = set(compute_global_indices(T, n_first, kfi))
            atts = []
            for f in range(T):
                lo, hi = get_window_bounds(f, W, T, expand=True,
                                           global_set=gset, global_count=len(gset))
                win = set(range(lo, hi + 1)) if lo <= hi else set()
                atts.append(len(gset | win))
            mean_att = sum(atts) / len(atts)
            sparsity = 1.0 - mean_att / T
            out.append((T, sparsity * 100))
        return out

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    for ref, label, color in [(21, "Wan (ref=21)", "#1f77b4"),
                               (33, "HunyuanVideo (ref=33)", "#d62728")]:
        pts = sparsity_curve(ref)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, label=label, linewidth=2, color=color)
    ax.set_xlabel("T_lat (latent frames)")
    ax.set_ylabel("Per-query sparsity (%)")
    ax.set_title("Sparsity vs sequence length (W=4, n_first=1, scale=1.0)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outpath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sota-csv", type=Path,
                    default=Path("out/sota_comparison/_summary_means.csv"))
    ap.add_argument("--scaling-csv", type=Path,
                    default=Path("out/latency_scaling/_summary_means.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("docs/figures"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    sota = load_csv(args.sota_csv)
    scaling = load_csv(args.scaling_csv)
    if not sota and not scaling:
        print(f"WARN: neither {args.sota_csv} nor {args.scaling_csv} has data.",
              file=sys.stderr)
        print("Run benchmarks/sota_comparison.sh and/or latency_scaling.sh first.",
              file=sys.stderr)

    print(f"Writing to {args.outdir}/ ...")
    fig_latency_scaling(scaling or sota, "wan13b",
                        "Wan 2.1 1.3B — wall-time vs frame count",
                        args.outdir / "latency_scaling.png")
    fig_latency_scaling(scaling, "hv15",
                        "HunyuanVideo 1.5 — wall-time vs frame count",
                        args.outdir / "hv_latency_scaling.png")
    fig_crossmodel_speedup(sota, scaling,
                           args.outdir / "crossmodel_speedup.png")
    fig_sparsity_vs_frames(args.outdir / "sparsity_vs_frames.png")


if __name__ == "__main__":
    main()
