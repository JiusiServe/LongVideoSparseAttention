"""Batch-mode figure generation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from vqeval.core.report import Report
from vqeval.figures.style import (
    COLORS, DISPLAY_NAMES, apply_style, get_verdict_color, save_fig,
)


def plot_batch_ranking(reports: list[Report], output_dir: Path, fmt: str = "png") -> Path:
    """Horizontal bar chart of videos ranked by composite score."""
    apply_style()

    # Sort by score descending
    sorted_reports = sorted(reports, key=lambda r: r.composite_score, reverse=True)
    names = [_short_name_from_report(r) for r in sorted_reports]
    scores = [r.composite_score for r in sorted_reports]
    colors = [get_verdict_color(s) for s in scores]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(names) + 1)))
    y_pos = range(len(names))

    bars = ax.barh(y_pos, scores, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Composite Score")
    ax.set_title(f"Video Quality Ranking ({len(reports)} videos)")

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{score:.0f}", va="center", fontsize=8, fontweight="bold",
        )

    ax.invert_yaxis()
    fig.tight_layout()

    path = output_dir / f"batch_ranking.{fmt}" if fmt != "both" else output_dir / "batch_ranking.png"
    save_fig(fig, path, fmt)
    return path


def plot_batch_dimension_comparison(reports: list[Report], output_dir: Path, fmt: str = "png") -> Path:
    """Grouped bar chart comparing dimension scores across videos."""
    apply_style()

    if not reports:
        return output_dir / f"batch_dimension_comparison.{fmt}"

    # Get all dimensions present across all reports
    all_dims = set()
    for r in reports:
        all_dims.update(r.dimension_results.keys())
    dims = sorted(all_dims)

    video_names = [_short_name_from_report(r) for r in reports]
    n_videos = len(reports)
    n_dims = len(dims)

    fig, ax = plt.subplots(figsize=(max(6, n_videos * 0.8 + 2), 5))

    x = np.arange(n_videos)
    width = 0.8 / n_dims

    for i, dim in enumerate(dims):
        scores = [
            r.dimension_results.get(dim, {}).get("score", 0) for r in reports
        ]
        offset = (i - n_dims / 2 + 0.5) * width
        color = COLORS.get(dim, f"C{i}")
        ax.bar(x + offset, scores, width, label=DISPLAY_NAMES.get(dim, dim), color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(video_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 105)
    ax.set_title("Dimension Scores by Video")
    ax.legend(fontsize=8, ncol=min(3, n_dims), loc="upper right")
    fig.tight_layout()

    path = output_dir / f"batch_dimension_comparison.{fmt}" if fmt != "both" else output_dir / "batch_dimension_comparison.png"
    save_fig(fig, path, fmt)
    return path


def plot_batch_distributions(reports: list[Report], output_dir: Path, fmt: str = "png") -> Path:
    """Boxplots showing score distributions per dimension across the batch."""
    apply_style()

    if not reports:
        return output_dir / f"batch_distributions.{fmt}"

    all_dims = set()
    for r in reports:
        all_dims.update(r.dimension_results.keys())
    dims = sorted(all_dims)

    data = []
    labels = []
    colors = []
    for dim in dims:
        scores = [
            r.dimension_results.get(dim, {}).get("score", 0) for r in reports
        ]
        data.append(scores)
        labels.append(DISPLAY_NAMES.get(dim, dim))
        colors.append(COLORS.get(dim, "#999"))

    # Also add composite
    composite_scores = [r.composite_score for r in reports]
    data.append(composite_scores)
    labels.append("Composite")
    colors.append("#333333")

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.8 + 2), 5))

    bp = ax.boxplot(
        data, patch_artist=True, tick_labels=labels,
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_ylabel("Score")
    ax.set_ylim(0, 105)
    ax.set_title(f"Score Distributions ({len(reports)} videos)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    path = output_dir / f"batch_distributions.{fmt}" if fmt != "both" else output_dir / "batch_distributions.png"
    save_fig(fig, path, fmt)
    return path


def plot_batch_grouped_comparison(
    reports: list[Report], output_dir: Path, fmt: str = "png",
    group_col: str = "model", hue_col: str = "backend",
) -> Optional[Path]:
    """Grouped bar chart: group_col on x-axis, hue_col as color, composite score on y.

    Only generated when reports have extra_meta with the required columns.
    """
    apply_style()

    # Check that at least some reports have the required columns
    valid = [r for r in reports if group_col in r.extra_meta and hue_col in r.extra_meta]
    if len(valid) < 2:
        return None

    # Collect unique groups and hues
    groups = sorted(set(r.extra_meta[group_col] for r in valid))
    hues = sorted(set(r.extra_meta[hue_col] for r in valid))

    # Build score matrix: for each (group, hue) take the mean composite score
    score_map: dict[tuple[str, str], list[float]] = {}
    for r in valid:
        key = (r.extra_meta[group_col], r.extra_meta[hue_col])
        score_map.setdefault(key, []).append(r.composite_score)

    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.2 + 2), 5))
    x = np.arange(len(groups))
    n_hues = len(hues)
    width = 0.8 / max(n_hues, 1)

    palette = plt.cm.tab20(np.linspace(0, 1, max(n_hues, 1)))

    for i, hue in enumerate(hues):
        means = []
        for g in groups:
            vals = score_map.get((g, hue), [])
            means.append(np.mean(vals) if vals else 0)
        offset = (i - n_hues / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=hue, color=palette[i])

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Composite Score")
    ax.set_ylim(0, 105)
    ax.set_title(f"Quality by {group_col.replace('_', ' ').title()} and {hue_col.replace('_', ' ').title()}")
    ax.legend(fontsize=7, ncol=min(4, n_hues), loc="upper right", title=hue_col)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = output_dir / f"batch_grouped_{group_col}_{hue_col}.{fmt}" if fmt != "both" \
        else output_dir / f"batch_grouped_{group_col}_{hue_col}.png"
    save_fig(fig, path, fmt)
    return path


def _short_name(path: str, max_len: int = 40) -> str:
    """Shorten a video path for display."""
    name = Path(path).stem
    if len(name) > max_len:
        return name[:max_len - 3] + "..."
    return name


def _short_name_from_report(report: Report, max_len: int = 40) -> str:
    """Use extra_meta for a readable label, falling back to filename."""
    meta = report.extra_meta
    if meta:
        # Try to build a label from common metadata columns
        parts = []
        for key in ("model", "backend", "frame_label"):
            if key in meta and meta[key]:
                parts.append(str(meta[key]))
        if parts:
            label = "/".join(parts)
            if len(label) > max_len:
                return label[:max_len - 3] + "..."
            return label
    return _short_name(report.video_path, max_len)
