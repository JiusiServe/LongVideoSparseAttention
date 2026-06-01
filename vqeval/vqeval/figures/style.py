"""Shared matplotlib styling for publication-quality figures."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Colorblind-friendly palette (Okabe-Ito inspired)
COLORS = {
    "spatial_quality": "#0072B2",
    "temporal_coherence": "#E69F00",
    "loop_quality": "#CC79A7",
    "artifact_detection": "#D55E00",
    "dynamic_quality": "#009E73",
    "text_alignment": "#56B4E9",
}

# Verdict colors
VERDICT_COLORS = {
    "excellent": "#22c55e",
    "good": "#86efac",
    "fair": "#fbbf24",
    "poor": "#f97316",
    "bad": "#ef4444",
}

# Short display names for axes labels
DISPLAY_NAMES = {
    "spatial_quality": "Spatial",
    "temporal_coherence": "Temporal",
    "loop_quality": "Repetition",
    "artifact_detection": "Artifacts",
    "dynamic_quality": "Dynamic",
    "text_alignment": "Text Align.",
}

DPI = 300
FIG_FORMAT = "png"


def apply_style():
    """Set global matplotlib style for clean academic figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (6, 4),
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def get_verdict_color(score: float) -> str:
    """Map a 0-100 score to a verdict color."""
    if score >= 90:
        return VERDICT_COLORS["excellent"]
    if score >= 70:
        return VERDICT_COLORS["good"]
    if score >= 50:
        return VERDICT_COLORS["fair"]
    if score >= 30:
        return VERDICT_COLORS["poor"]
    return VERDICT_COLORS["bad"]


def save_fig(fig, path, fmt="png"):
    """Save a figure, supporting 'both' for png+pdf."""
    if fmt == "both":
        fig.savefig(str(path).replace(".png", ".png"))
        fig.savefig(str(path).replace(".png", ".pdf"))
    else:
        fig.savefig(str(path))
    plt.close(fig)
