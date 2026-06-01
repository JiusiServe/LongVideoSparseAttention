"""Tests for normalization and calibration module."""

import json
import pytest
import numpy as np

from vqeval.normalization.calibration import MetricNormalizer, Calibrator


class TestMetricNormalizer:
    def test_linear_higher_is_better(self):
        presets = {
            "test_metric": {"strategy": "linear", "min": 0.0, "max": 100.0}
        }
        norm = MetricNormalizer(presets)
        assert norm.normalize("test_metric", 50.0) == pytest.approx(50.0)
        assert norm.normalize("test_metric", 0.0) == pytest.approx(0.0)
        assert norm.normalize("test_metric", 100.0) == pytest.approx(100.0)

    def test_linear_lower_is_better(self):
        presets = {
            "test_metric": {"strategy": "linear", "min": 0.0, "max": 100.0}
        }
        norm = MetricNormalizer(presets)
        assert norm.normalize("test_metric", 0.0, lower_is_better=True) == pytest.approx(100.0)
        assert norm.normalize("test_metric", 100.0, lower_is_better=True) == pytest.approx(0.0)

    def test_linear_clamps(self):
        presets = {
            "test_metric": {"strategy": "linear", "min": 10.0, "max": 90.0}
        }
        norm = MetricNormalizer(presets)
        assert norm.normalize("test_metric", -100.0) == 0.0
        assert norm.normalize("test_metric", 200.0) == 100.0

    def test_sigmoid(self):
        presets = {
            "test_metric": {"strategy": "sigmoid", "center": 50.0, "scale": 10.0}
        }
        norm = MetricNormalizer(presets)
        score_center = norm.normalize("test_metric", 50.0)
        score_high = norm.normalize("test_metric", 80.0)
        score_low = norm.normalize("test_metric", 20.0)
        assert score_center == pytest.approx(50.0, abs=1.0)
        assert score_high > score_center
        assert score_low < score_center

    def test_percentile(self):
        presets = {
            "test_metric": {
                "strategy": "percentile",
                "percentiles": [[0, 0], [50, 50], [100, 100]],
            }
        }
        norm = MetricNormalizer(presets)
        assert norm.normalize("test_metric", 25.0) == pytest.approx(25.0, abs=1.0)
        assert norm.normalize("test_metric", 75.0) == pytest.approx(75.0, abs=1.0)

    def test_default_ranges_fallback(self):
        norm = MetricNormalizer()
        # BRISQUE: lower is better, range 0-80
        score = norm.normalize("brisque", 40.0, lower_is_better=True)
        assert 0 <= score <= 100

    def test_unknown_metric_fallback(self):
        norm = MetricNormalizer()
        score = norm.normalize("unknown_metric", 0.5)
        assert 0 <= score <= 100


class TestCalibrator:
    def test_linear_calibration(self):
        cal = Calibrator()
        values = list(range(100))
        preset = cal.calibrate("test", values, strategy="linear")
        assert preset["strategy"] == "linear"
        assert "min" in preset
        assert "max" in preset

    def test_sigmoid_calibration(self):
        cal = Calibrator()
        values = np.random.randn(100).tolist()
        preset = cal.calibrate("test", values, strategy="sigmoid")
        assert preset["strategy"] == "sigmoid"
        assert "center" in preset
        assert "scale" in preset

    def test_percentile_calibration(self):
        cal = Calibrator()
        values = list(range(200))
        preset = cal.calibrate("test", values, strategy="percentile")
        assert preset["strategy"] == "percentile"
        assert len(preset["percentiles"]) > 0

    def test_save_presets(self, tmp_path):
        cal = Calibrator()
        presets = {"metric1": {"strategy": "linear", "min": 0, "max": 100}}
        path = str(tmp_path / "presets.json")
        cal.save_presets(presets, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == presets
