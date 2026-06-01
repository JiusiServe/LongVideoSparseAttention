"""Report generation: JSON, CSV, and HTML outputs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape


TEMPLATE_DIR = Path(__file__).parent.parent / "reports" / "templates"


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class Report:
    """Complete evaluation report for a single video."""

    video_path: str
    video_meta: dict[str, Any]
    dimension_results: dict[str, dict[str, Any]]
    composite_score: float
    composite_verdict: str
    weights: dict[str, float]
    config: dict[str, Any]
    elapsed_seconds: float
    raw_results: dict[str, Any] = field(default_factory=dict)
    extra_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "video_path": self.video_path,
            "video_meta": self.video_meta,
            "composite_score": round(self.composite_score, 1),
            "composite_verdict": self.composite_verdict,
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "dimensions": self.dimension_results,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }
        if self.extra_meta:
            d["extra_meta"] = self.extra_meta
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, cls=_NumpyEncoder)


class ReportGenerator:
    """Generates output reports in various formats."""

    def to_json_file(self, report: Report, path: str):
        Path(path).write_text(report.to_json())

    def to_csv(self, reports: list[Report], path: str):
        """Write batch reports to CSV."""
        if not reports:
            return

        # Flatten report data for CSV
        rows = []
        extra_keys: list[str] = []
        for r in reports:
            row = {}
            # Prepend extra_meta columns (from --from-csv) before quality columns
            if r.extra_meta:
                for k, v in r.extra_meta.items():
                    row[k] = v
                    if k not in extra_keys:
                        extra_keys.append(k)

            row["video_path"] = r.video_path
            row["composite_score"] = round(r.composite_score, 1)
            row["composite_verdict"] = r.composite_verdict
            row["duration"] = r.video_meta.get("duration", 0)
            row["resolution"] = f"{r.video_meta.get('width', 0)}x{r.video_meta.get('height', 0)}"

            for dim_name, dim_data in r.dimension_results.items():
                row[f"{dim_name}_score"] = dim_data.get("score", "")
                row[f"{dim_name}_verdict"] = dim_data.get("verdict", "")
            rows.append(row)

        # Collect all columns: extra_meta first, then quality columns
        quality_keys = [
            k for row in rows for k in row.keys()
            if k not in extra_keys
        ]
        quality_keys = list(dict.fromkeys(quality_keys))
        all_keys = extra_keys + quality_keys

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(rows)

    def to_html(self, report: Report, path: str):
        """Generate a single-video HTML report."""
        env = self._get_jinja_env()
        template = env.get_template("single_report.html")
        html = template.render(report=report.to_dict())
        Path(path).write_text(html)

    def to_html_batch(self, reports: list[Report], path: str):
        """Generate a batch HTML report."""
        env = self._get_jinja_env()
        template = env.get_template("batch_report.html")
        html = template.render(
            reports=[r.to_dict() for r in reports],
            total=len(reports),
            avg_score=sum(r.composite_score for r in reports) / len(reports)
            if reports
            else 0,
        )
        Path(path).write_text(html)

    def _get_jinja_env(self) -> Environment:
        return Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=select_autoescape(["html"]),
        )
