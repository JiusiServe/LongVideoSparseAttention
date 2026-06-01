"""Normalization and calibration utilities for mapping raw metrics to 0-100 scores."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np


class MetricNormalizer:
    """Normalizes raw metric values to 0-100 scores using calibration data.

    Supports multiple normalization strategies:
    - linear: simple min-max scaling
    - sigmoid: sigmoid mapping centered on median
    - percentile: empirical percentile mapping
    """

    def __init__(self, presets: Optional[dict] = None):
        self.presets = presets or {}

    def normalize(
        self,
        metric_name: str,
        raw_value: float,
        lower_is_better: bool = False,
        strategy: str = "linear",
    ) -> float:
        """Normalize a raw metric value to 0-100.

        Args:
            metric_name: Name of the metric (used to look up calibration data)
            raw_value: The raw metric value
            lower_is_better: If True, lower raw values map to higher scores
            strategy: Normalization strategy (linear, sigmoid, percentile)

        Returns:
            Normalized score in [0, 100]
        """
        preset = self.presets.get(metric_name)
        if preset:
            return self._normalize_with_preset(raw_value, preset, lower_is_better)

        # Fallback: use default ranges
        return self._normalize_default(metric_name, raw_value, lower_is_better)

    def _normalize_with_preset(
        self, value: float, preset: dict, lower_is_better: bool
    ) -> float:
        strategy = preset.get("strategy", "linear")

        if strategy == "linear":
            lo = preset["min"]
            hi = preset["max"]
            if hi == lo:
                return 50.0
            normalized = (value - lo) / (hi - lo)
            if lower_is_better:
                normalized = 1.0 - normalized
            return max(0.0, min(100.0, normalized * 100.0))

        elif strategy == "sigmoid":
            center = preset["center"]
            scale = preset["scale"]
            x = (value - center) / scale
            if lower_is_better:
                x = -x
            return max(0.0, min(100.0, 100.0 / (1.0 + math.exp(-x))))

        elif strategy == "percentile":
            percentiles = preset["percentiles"]  # list of (raw_value, percentile)
            return self._interpolate_percentile(
                value, percentiles, lower_is_better
            )

        return 50.0

    def _interpolate_percentile(
        self, value: float, percentiles: list[list[float]], lower_is_better: bool
    ) -> float:
        """Interpolate a score based on empirical percentile data."""
        sorted_p = sorted(percentiles, key=lambda x: x[0])
        raw_vals = [p[0] for p in sorted_p]
        pct_vals = [p[1] for p in sorted_p]

        if value <= raw_vals[0]:
            score = pct_vals[0]
        elif value >= raw_vals[-1]:
            score = pct_vals[-1]
        else:
            score = float(np.interp(value, raw_vals, pct_vals))

        if lower_is_better:
            score = 100.0 - score
        return max(0.0, min(100.0, score))

    def _normalize_default(
        self, name: str, value: float, lower_is_better: bool
    ) -> float:
        """Fallback normalization using built-in default ranges."""
        defaults = _DEFAULT_RANGES.get(name)
        if defaults:
            lo, hi = defaults
            if hi == lo:
                return 50.0
            normalized = (value - lo) / (hi - lo)
            normalized = max(0.0, min(1.0, normalized))
            if lower_is_better:
                normalized = 1.0 - normalized
            return normalized * 100.0

        # If no range known, assume value is already 0-1 and scale
        if lower_is_better:
            value = 1.0 - value
        return max(0.0, min(100.0, value * 100.0))


# Default metric ranges (empirically estimated for AI-generated video)
_DEFAULT_RANGES = {
    "brisque": (0.0, 80.0),         # lower is better quality
    "niqe": (2.0, 10.0),            # lower is better
    "clip_iqa": (0.0, 1.0),         # higher is better
    "laplacian_var": (0.0, 1000.0), # higher is better (sharper)
    "ssim": (0.0, 1.0),             # higher is better
    "lpips": (0.0, 1.0),            # lower is better
    "cosine_sim": (0.0, 1.0),       # higher is better
    "flow_magnitude": (0.0, 50.0),  # context-dependent
    "dynamic_degree": (0.0, 1.0),   # higher is better
    "aesthetic": (1.0, 10.0),       # higher is better
    "clip_text_sim": (0.0, 0.45),   # higher is better
}


class Calibrator:
    """Builds normalization presets from a calibration dataset."""

    def calibrate(
        self, metric_name: str, raw_values: list[float], strategy: str = "percentile"
    ) -> dict:
        """Compute normalization parameters from a set of raw values.

        Args:
            metric_name: Name of the metric
            raw_values: List of raw metric values from calibration set
            strategy: Normalization strategy to use

        Returns:
            Preset dict suitable for MetricNormalizer
        """
        arr = np.array(raw_values)

        if strategy == "linear":
            return {
                "strategy": "linear",
                "min": float(np.percentile(arr, 2)),
                "max": float(np.percentile(arr, 98)),
            }

        elif strategy == "sigmoid":
            return {
                "strategy": "sigmoid",
                "center": float(np.median(arr)),
                "scale": float(np.std(arr)) if np.std(arr) > 0 else 1.0,
            }

        elif strategy == "percentile":
            pcts = [0, 5, 10, 25, 50, 75, 90, 95, 100]
            percentiles = [
                [float(np.percentile(arr, p)), float(p)] for p in pcts
            ]
            return {
                "strategy": "percentile",
                "percentiles": percentiles,
            }

        raise ValueError(f"Unknown strategy: {strategy}")

    def save_presets(self, presets: dict, path: str):
        """Save calibration presets to JSON file."""
        with open(path, "w") as f:
            json.dump(presets, f, indent=2)
