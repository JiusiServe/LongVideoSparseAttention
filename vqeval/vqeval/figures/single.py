"""Single-video figure generation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from vqeval.core.report import Report
from vqeval.figures.style import (
    COLORS, DISPLAY_NAMES, apply_style, get_verdict_color, save_fig,
)


def plot_radar_scores(report: Report, output_dir: Path, fmt: str = "png") -> Path:
    """Spider/radar chart of all dimension scores."""
    apply_style()

    dims = report.dimension_results
    names = list(dims.keys())
    scores = [dims[n].get("score", 0) for n in names]
    labels = [DISPLAY_NAMES.get(n, n) for n in names]
    n = len(names)

    if n < 3:
        return _plot_fallback_bars(report, output_dir, fmt)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    scores_closed = scores + [scores[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection="polar"))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles_closed, scores_closed, "o-", linewidth=2, color="#0072B2")
    ax.fill(angles_closed, scores_closed, alpha=0.15, color="#0072B2")

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7, color="gray")

    # Composite score in center
    ax.text(
        0, 0, f"{report.composite_score:.0f}",
        ha="center", va="center", fontsize=22, fontweight="bold",
        color=get_verdict_color(report.composite_score),
        transform=ax.transData,
    )

    ax.set_title("Quality Dimensions", pad=20, fontsize=13)
    fig.tight_layout()

    path = output_dir / f"radar_scores.{fmt}" if fmt != "both" else output_dir / "radar_scores.png"
    save_fig(fig, path, fmt)
    return path


def _plot_fallback_bars(report: Report, output_dir: Path, fmt: str) -> Path:
    """Fallback bar chart when fewer than 3 dimensions."""
    return plot_dimension_bars(report, output_dir, fmt)


def plot_dimension_bars(report: Report, output_dir: Path, fmt: str = "png") -> Path:
    """Horizontal bar chart of dimension scores, color-coded by verdict."""
    apply_style()

    dims = report.dimension_results
    names = list(dims.keys())
    scores = [dims[n].get("score", 0) for n in names]
    labels = [DISPLAY_NAMES.get(n, n) for n in names]
    colors = [get_verdict_color(s) for s in scores]

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.6 * len(names) + 1)))
    y_pos = range(len(names))

    bars = ax.barh(y_pos, scores, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Score")
    ax.set_title("Dimension Scores")

    # Score labels on bars
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{score:.0f}", va="center", fontsize=9, fontweight="bold",
        )

    # Composite score as vertical line
    ax.axvline(
        x=report.composite_score, color="#333", linestyle="--",
        linewidth=1.5, label=f"Composite: {report.composite_score:.0f}",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()

    path = output_dir / f"dimension_scores.{fmt}" if fmt != "both" else output_dir / "dimension_scores.png"
    save_fig(fig, path, fmt)
    return path


def plot_temporal_profile(report: Report, output_dir: Path, fmt: str = "png") -> Optional[Path]:
    """Per-frame spatial quality metrics over time."""
    apply_style()

    traces = _get_traces(report, "spatial_quality")
    if traces is None:
        return None

    frame_idx = traces.get("frame_indices")
    brisque = traces.get("brisque_per_frame")
    clip_iqa = traces.get("clip_iqa_per_frame")
    sharpness = traces.get("sharpness_per_frame")

    # Count how many traces we have
    panels = []
    if brisque is not None:
        panels.append(("BRISQUE (lower=better)", brisque, True))
    if clip_iqa is not None:
        panels.append(("CLIP-IQA", clip_iqa, False))
    if sharpness is not None:
        panels.append(("Sharpness (Laplacian var.)", sharpness, False))

    if not panels:
        return None

    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 2.5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    x = frame_idx if frame_idx is not None else np.arange(len(panels[0][1]))

    for ax, (title, data, invert) in zip(axes, panels):
        if data is None:
            continue
        ax.plot(x[:len(data)], data, linewidth=1.5, color="#0072B2")
        ax.fill_between(x[:len(data)], data, alpha=0.1, color="#0072B2")
        ax.set_ylabel(title, fontsize=9)
        if invert:
            ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frame Index")
    fig.suptitle("Spatial Quality Over Time", fontsize=12)
    fig.tight_layout()

    path = output_dir / f"temporal_profile.{fmt}" if fmt != "both" else output_dir / "temporal_profile.png"
    save_fig(fig, path, fmt)
    return path


def plot_self_similarity_heatmap(report: Report, output_dir: Path, fmt: str = "png") -> Optional[Path]:
    """NxN CLIP embedding self-similarity heatmap."""
    apply_style()

    traces = _get_traces(report, "loop_quality")
    if traces is None:
        return None

    sim_matrix = traces.get("self_similarity_matrix")
    if sim_matrix is None or not hasattr(sim_matrix, "shape"):
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim_matrix, cmap="RdYlBu_r", vmin=0.5, vmax=1.0, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cosine Similarity")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Frame Index")
    ax.set_title("CLIP Embedding Self-Similarity")

    # Annotate if cycle detected
    loop_data = report.dimension_results.get("loop_quality", {})
    if loop_data.get("cycle_detected"):
        period = loop_data.get("cycle_period_frames", 0)
        ax.set_title(f"CLIP Self-Similarity (cycle period: ~{period} frames)")

    fig.tight_layout()

    path = output_dir / f"self_similarity_heatmap.{fmt}" if fmt != "both" else output_dir / "self_similarity_heatmap.png"
    save_fig(fig, path, fmt)
    return path


def plot_coherence_profile(report: Report, output_dir: Path, fmt: str = "png") -> Optional[Path]:
    """Temporal coherence: consecutive CLIP similarities and flow magnitudes."""
    apply_style()

    traces = _get_traces(report, "temporal_coherence")
    if traces is None:
        return None

    clip_sims = traces.get("clip_consecutive_sims")
    flow_mags = traces.get("flow_magnitudes")
    flow_pair_indices = traces.get("flow_pair_indices")

    panels = []
    if clip_sims is not None and len(clip_sims) > 0:
        # CLIP sims are computed for ALL consecutive sampled frames (no sub-sampling)
        panels.append(("CLIP Frame-to-Frame Similarity", clip_sims, "#E69F00", None))
    if flow_mags is not None and len(flow_mags) > 0:
        # Flow is sub-sampled — use stored indices for accurate x-axis
        panels.append(("Optical Flow Magnitude", flow_mags, "#009E73", flow_pair_indices))

    if not panels:
        return None

    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 2.5 * len(panels)), sharex=False)
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, data, color, indices) in zip(axes, panels):
        data = np.array(data)
        if indices is not None and len(indices) == len(data):
            x = np.array(indices)
        else:
            x = np.arange(len(data))
        ax.plot(x, data, linewidth=1.5, color=color)
        ax.fill_between(x, data, alpha=0.1, color=color)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(True, alpha=0.3)

        # Highlight anomalous transitions
        if len(data) > 3:
            mean, std = data.mean(), data.std()
            outliers = np.where(np.abs(data - mean) > 2 * std)[0]
            if len(outliers) > 0:
                ax.scatter(x[outliers], data[outliers], color="#D55E00", s=30, zorder=5, label="Outliers")
                ax.legend(fontsize=8)

    axes[-1].set_xlabel("Sampled Frame Index")
    fig.suptitle("Temporal Coherence Profile", fontsize=12)
    fig.tight_layout()

    path = output_dir / f"coherence_profile.{fmt}" if fmt != "both" else output_dir / "coherence_profile.png"
    save_fig(fig, path, fmt)
    return path


def plot_text_alignment_drift(report: Report, output_dir: Path, fmt: str = "png") -> Optional[Path]:
    """Per-frame CLIP text-video similarity with trend line."""
    apply_style()

    traces = _get_traces(report, "text_alignment")
    if traces is None:
        return None

    sims = traces.get("clip_similarity_per_frame")
    if sims is None or len(sims) < 3:
        return None

    sims = np.array(sims)
    x = np.arange(len(sims))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, sims, "o-", markersize=3, linewidth=1.5, color="#56B4E9", label="CLIP Similarity")

    # Trend line
    coeffs = np.polyfit(x, sims, 1)
    trend = np.polyval(coeffs, x)
    slope = coeffs[0]
    ax.plot(x, trend, "--", linewidth=1.5, color="#D55E00",
            label=f"Trend (slope={slope:.4f})")

    # Min/max band
    ax.fill_between(x, sims.min(), sims.max(), alpha=0.05, color="#56B4E9")

    ax.set_xlabel("Frame Index")
    ax.set_ylabel("CLIP Cosine Similarity")
    ax.set_title("Text-Video Alignment Over Time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = output_dir / f"text_alignment_drift.{fmt}" if fmt != "both" else output_dir / "text_alignment_drift.png"
    save_fig(fig, path, fmt)
    return path


def plot_motion_profile(report: Report, output_dir: Path, fmt: str = "png") -> Optional[Path]:
    """Flow magnitude over time from coherence or loop evaluator."""
    apply_style()

    # Try temporal_coherence first, then loop_quality
    traces = _get_traces(report, "temporal_coherence")
    flow_mags = traces.get("flow_magnitudes") if traces else None
    flow_indices = traces.get("flow_pair_indices") if traces else None

    if flow_mags is None or len(flow_mags) < 2:
        traces = _get_traces(report, "loop_quality")
        flow_mags = traces.get("flow_magnitude_series") if traces else None
        flow_indices = traces.get("flow_pair_indices") if traces else None

    if flow_mags is None or len(flow_mags) < 2:
        return None

    flow_mags = np.array(flow_mags)
    if flow_indices is not None and len(flow_indices) == len(flow_mags):
        x = np.array(flow_indices)
    else:
        x = np.arange(len(flow_mags))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(x, flow_mags, linewidth=1.5, color="#009E73")
    ax.fill_between(x, flow_mags, alpha=0.1, color="#009E73")
    ax.set_xlabel("Sampled Frame Index")
    ax.set_ylabel("Mean Flow Magnitude")
    ax.set_title("Motion Profile")
    ax.grid(True, alpha=0.3)

    # Annotate dynamic degree if available
    dyn = report.dimension_results.get("dynamic_quality", {})
    dd = dyn.get("dynamic_degree")
    if dd is not None:
        ax.text(
            0.98, 0.95, f"Dynamic degree: {dd:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    fig.tight_layout()

    path = output_dir / f"motion_profile.{fmt}" if fmt != "both" else output_dir / "motion_profile.png"
    save_fig(fig, path, fmt)
    return path


def _get_traces(report: Report, dimension: str) -> Optional[dict]:
    """Get traces dict for a dimension, or None if unavailable."""
    raw = getattr(report, "raw_results", {})
    if not raw or dimension not in raw:
        return None
    result = raw[dimension]
    if hasattr(result, "traces") and result.traces:
        return result.traces
    return None
