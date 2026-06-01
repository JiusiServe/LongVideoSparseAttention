"""Figure generator orchestrating all plot creation."""

from __future__ import annotations

import logging
from pathlib import Path

from vqeval.core.report import Report

logger = logging.getLogger(__name__)


class FigureGenerator:
    """Generates publication-quality figures from evaluation reports."""

    def __init__(self):
        # Validate matplotlib is available
        try:
            import matplotlib
        except ImportError:
            raise ImportError(
                "Figure generation requires matplotlib. "
                "Install with: pip install matplotlib>=3.7.0"
            )

    def generate_single(
        self, report: Report, output_dir: Path, fmt: str = "png"
    ) -> list[Path]:
        """Generate all applicable figures for a single video report.

        Args:
            report: Evaluation report with raw_results containing traces
            output_dir: Directory to save figures
            fmt: "png", "pdf", or "both"

        Returns:
            List of generated file paths (verified to exist on disk)
        """
        from vqeval.figures.single import (
            plot_radar_scores,
            plot_dimension_bars,
            plot_temporal_profile,
            plot_self_similarity_heatmap,
            plot_coherence_profile,
            plot_text_alignment_drift,
            plot_motion_profile,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        generated = []

        # Always generate score overview charts
        generators = [
            plot_radar_scores,
            plot_dimension_bars,
            # Trace-dependent figures (return None if traces unavailable)
            plot_temporal_profile,
            plot_self_similarity_heatmap,
            plot_coherence_profile,
            plot_text_alignment_drift,
            plot_motion_profile,
        ]

        for gen_func in generators:
            try:
                result = gen_func(report, output_dir, fmt)
                if result is not None:
                    generated.extend(_collect_paths(result, fmt))
            except Exception as e:
                logger.debug("Figure %s failed: %s", gen_func.__name__, e)

        return generated

    def generate_batch(
        self, reports: list[Report], output_dir: Path, fmt: str = "png"
    ) -> list[Path]:
        """Generate batch comparison figures.

        Args:
            reports: List of evaluation reports
            output_dir: Directory to save figures
            fmt: "png", "pdf", or "both"

        Returns:
            List of generated file paths (verified to exist on disk)
        """
        from vqeval.figures.batch import (
            plot_batch_ranking,
            plot_batch_dimension_comparison,
            plot_batch_distributions,
            plot_batch_grouped_comparison,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        generated = []

        generators = [
            plot_batch_ranking,
            plot_batch_dimension_comparison,
            plot_batch_distributions,
            plot_batch_grouped_comparison,
        ]

        for gen_func in generators:
            try:
                result = gen_func(reports, output_dir, fmt)
                if result is not None:
                    generated.extend(_collect_paths(result, fmt))
            except Exception as e:
                logger.debug("Figure %s failed: %s", gen_func.__name__, e)

        # Also generate per-video figures in subdirectories
        used_names: dict[str, int] = {}
        for report in reports:
            video_name = _unique_dir_name(report, used_names)
            video_dir = output_dir / video_name
            try:
                per_video = self.generate_single(report, video_dir, fmt)
                generated.extend(per_video)
            except Exception as e:
                logger.debug("Per-video figures for %s failed: %s", video_name, e)

        return generated


def _unique_dir_name(report: Report, used: dict[str, int]) -> str:
    """Create a unique, filesystem-safe directory name for a video.

    Uses extra_meta keys (model/backend/frame_label) when available for
    readable names like 'wan_1.3b__baseline__1x'. Falls back to the
    video's parent directory structure, then to a truncated filename stem.
    Appends _2, _3, ... on collision.
    """
    meta = report.extra_meta

    # Strategy 1: build from metadata columns
    if meta:
        parts = []
        for key in ("model", "backend", "frame_label"):
            if key in meta and meta[key]:
                parts.append(str(meta[key]))
        if parts:
            stem = "__".join(parts)
            return _deduplicate(stem, used)

    # Strategy 2: use parent directory structure from video path
    # e.g. "weekend/wan_1.3b/baseline/1x/long_name.mp4" -> "wan_1.3b__baseline__1x"
    video_path = Path(report.video_path)
    parents = list(video_path.parents)
    if len(parents) >= 4:
        # Take the 3 deepest parent dirs (skip the file's immediate directory name collision)
        parts = [video_path.parts[i] for i in range(-4, -1)]
        stem = "__".join(parts)
        return _deduplicate(stem, used)

    # Strategy 3: fallback to filename stem with middle truncation
    stem = video_path.stem
    max_len = 80
    if len(stem) > max_len:
        half = (max_len - 3) // 2
        stem = stem[:half] + "___" + stem[-half:]

    return _deduplicate(stem, used)


def _deduplicate(stem: str, used: dict[str, int]) -> str:
    """Append _2, _3, ... on name collision."""
    base = stem
    count = used.get(base, 0) + 1
    used[base] = count
    if count > 1:
        return f"{base}_{count}"
    return stem


def _collect_paths(path: Path, fmt: str) -> list[Path]:
    """Return list of files that actually exist on disk for a generated figure.

    When fmt='both', save_fig writes both .png and .pdf but only the .png
    path is returned by the plot function. This helper collects both.
    """
    result = []
    if path.exists():
        result.append(path)
    if fmt == "both":
        # save_fig also writes .pdf alongside .png
        pdf_path = path.with_suffix(".pdf")
        if pdf_path.exists():
            result.append(pdf_path)
    return result
