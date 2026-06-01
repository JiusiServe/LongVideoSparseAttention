"""Configuration and default parameters for VQeval."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# Default dimension weights (must sum to 1.0 when all active dimensions included)
DEFAULT_WEIGHTS = {
    "spatial_quality": 0.20,
    "temporal_coherence": 0.25,
    "loop_quality": 0.15,
    "artifact_detection": 0.20,
    "dynamic_quality": 0.10,
    "text_alignment": 0.10,
}

# Verdict thresholds
VERDICT_THRESHOLDS = {
    "excellent": 90,
    "good": 70,
    "fair": 50,
    "poor": 30,
    "bad": 0,
}

# Frame sampling configuration
SAMPLE_ALL_THRESHOLD_SEC = 5.0
MIN_SAMPLE_FPS = 2.0

# Blur detection
LAPLACIAN_BLUR_THRESHOLD = 100.0

# Normalization preset defaults
DEFAULT_PRESETS_PATH = Path(__file__).parent.parent / "normalization" / "presets.json"


def score_to_verdict(score: float) -> str:
    """Convert a 0-100 score to a human-readable verdict."""
    for verdict, threshold in VERDICT_THRESHOLDS.items():
        if score >= threshold:
            return verdict
    return "bad"


@dataclass
class EvalConfig:
    """Configuration for a single evaluation run."""

    video_path: str = ""
    loop: bool = False
    prompt: Optional[str] = None
    reference_image: Optional[str] = None
    dimensions: Optional[list[str]] = None
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    presets_path: Optional[str] = None
    export_frames_dir: Optional[str] = None
    html_report: Optional[str] = None
    csv_output: Optional[str] = None
    device: str = "cuda"
    sample_fps: Optional[float] = None  # None = auto (2 fps for >5s videos, all frames for ≤5s)

    def get_active_dimensions(self) -> list[str]:
        """Return the list of dimensions to evaluate."""
        all_dims = [
            "spatial_quality",
            "temporal_coherence",
            "loop_quality",
            "artifact_detection",
            "dynamic_quality",
        ]
        if self.prompt:
            all_dims.append("text_alignment")

        if self.dimensions:
            return [d for d in self.dimensions if d in all_dims]
        return all_dims

    def get_effective_weights(self) -> dict[str, float]:
        """Compute effective weights, redistributing inactive dimensions."""
        active = self.get_active_dimensions()
        raw = {k: v for k, v in self.weights.items() if k in active}
        total = sum(raw.values())
        if total == 0:
            n = len(raw)
            return {k: 1.0 / n for k in raw} if n > 0 else {}
        return {k: v / total for k, v in raw.items()}

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    input_dir: str = ""
    from_csv: Optional[str] = None
    video_col: str = "video_path"
    prompt_col: Optional[str] = None
    status_col: Optional[str] = "status"
    status_ok: str = "ok"
    output_csv: Optional[str] = None
    html_report: Optional[str] = None
    eval_config: EvalConfig = field(default_factory=EvalConfig)
    max_workers: int = 1


def load_presets(path: Optional[str] = None) -> dict:
    """Load normalization presets from a JSON file."""
    preset_path = Path(path) if path else DEFAULT_PRESETS_PATH
    if preset_path.exists():
        with open(preset_path, "r") as f:
            return json.load(f)
    return {}
