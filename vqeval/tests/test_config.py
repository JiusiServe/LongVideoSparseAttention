"""Tests for configuration module."""

import pytest
from vqeval.core.config import EvalConfig, score_to_verdict, DEFAULT_WEIGHTS


class TestScoreToVerdict:
    def test_excellent(self):
        assert score_to_verdict(95) == "excellent"

    def test_good(self):
        assert score_to_verdict(75) == "good"

    def test_fair(self):
        assert score_to_verdict(55) == "fair"

    def test_poor(self):
        assert score_to_verdict(35) == "poor"

    def test_bad(self):
        assert score_to_verdict(10) == "bad"

    def test_boundary_excellent(self):
        assert score_to_verdict(90) == "excellent"

    def test_boundary_good(self):
        assert score_to_verdict(70) == "good"

    def test_zero(self):
        assert score_to_verdict(0) == "bad"

    def test_hundred(self):
        assert score_to_verdict(100) == "excellent"


class TestEvalConfig:
    def test_default_dimensions_no_prompt(self):
        config = EvalConfig()
        dims = config.get_active_dimensions()
        assert "spatial_quality" in dims
        assert "temporal_coherence" in dims
        assert "artifact_detection" in dims
        assert "dynamic_quality" in dims
        assert "loop_quality" in dims
        assert "text_alignment" not in dims

    def test_prompt_adds_dimension(self):
        config = EvalConfig(prompt="a cat walking")
        dims = config.get_active_dimensions()
        assert "text_alignment" in dims

    def test_custom_dimensions(self):
        config = EvalConfig(
            dimensions=["spatial_quality", "temporal_coherence"]
        )
        dims = config.get_active_dimensions()
        assert dims == ["spatial_quality", "temporal_coherence"]

    def test_effective_weights_normalize(self):
        config = EvalConfig()
        weights = config.get_effective_weights()
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_effective_weights_include_loop(self):
        config = EvalConfig()
        weights = config.get_effective_weights()
        assert "loop_quality" in weights
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_to_dict(self):
        config = EvalConfig(video_path="test.mp4")
        d = config.to_dict()
        assert d["video_path"] == "test.mp4"
