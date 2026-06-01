"""Tests for evaluator base class and registration."""

import pytest

from vqeval.core.config import EvalConfig
from vqeval.evaluators.base import (
    BaseEvaluator,
    EvalResult,
    register_evaluator,
    get_evaluator_class,
    get_all_evaluator_names,
    _EVALUATOR_REGISTRY,
)

# Import all evaluators to trigger registration
import vqeval.evaluators.spatial_quality
import vqeval.evaluators.temporal_coherence
import vqeval.evaluators.loop_quality
import vqeval.evaluators.artifact_detection
import vqeval.evaluators.dynamic_quality
import vqeval.evaluators.text_alignment


class TestEvalResult:
    def test_to_dict(self):
        result = EvalResult(
            dimension="test_dim",
            score=75.5,
            verdict="good",
            metrics={"metric_a": 0.123456789, "metric_b": 42},
        )
        d = result.to_dict()
        assert "test_dim" in d
        assert d["test_dim"]["score"] == 75.5
        assert d["test_dim"]["verdict"] == "good"
        assert d["test_dim"]["metric_a"] == pytest.approx(0.1235, abs=0.001)
        assert d["test_dim"]["metric_b"] == 42

    def test_score_clamping(self):
        # Scores should be representable as-is in the result
        result = EvalResult(dimension="test", score=150.0, verdict="excellent", metrics={})
        d = result.to_dict()
        assert d["test"]["score"] == 150.0  # EvalResult doesn't clamp; _make_result does

    def test_list_metrics(self):
        result = EvalResult(
            dimension="test",
            score=50.0,
            verdict="fair",
            metrics={"frames": [1, 2, 3], "nested": [0.1, 0.2]},
        )
        d = result.to_dict()
        assert d["test"]["frames"] == [1, 2, 3]

    def test_traces_default_empty(self):
        result = EvalResult(dimension="test", score=50.0, verdict="fair", metrics={})
        assert result.traces == {}

    def test_traces_stored(self):
        traces = {"per_frame_scores": [0.1, 0.2, 0.3], "heatmap": [[1, 2], [3, 4]]}
        result = EvalResult(
            dimension="test", score=50.0, verdict="fair", metrics={}, traces=traces
        )
        assert result.traces == traces
        assert result.traces["per_frame_scores"] == [0.1, 0.2, 0.3]

    def test_traces_excluded_from_to_dict(self):
        traces = {"internal_data": [1, 2, 3]}
        result = EvalResult(
            dimension="test", score=60.0, verdict="fair", metrics={"m": 1}, traces=traces
        )
        d = result.to_dict()
        assert "internal_data" not in d["test"]
        assert "traces" not in d["test"]


class TestEvaluatorRegistry:
    def test_all_evaluators_registered(self):
        names = get_all_evaluator_names()
        expected = {
            "spatial_quality",
            "temporal_coherence",
            "loop_quality",
            "artifact_detection",
            "dynamic_quality",
            "text_alignment",
        }
        assert expected.issubset(set(names))

    def test_get_evaluator_class(self):
        cls = get_evaluator_class("spatial_quality")
        assert issubclass(cls, BaseEvaluator)
        assert cls.dimension_name == "spatial_quality"

    def test_get_unknown_evaluator(self):
        with pytest.raises(ValueError, match="Unknown evaluator"):
            get_evaluator_class("nonexistent_dimension")

    def test_evaluator_creation(self):
        config = EvalConfig(device="cpu")
        cls = get_evaluator_class("spatial_quality")
        evaluator = cls(config=config)
        assert evaluator.dimension_name == "spatial_quality"
        assert evaluator.device == "cpu"

    def test_loop_evaluator_always_applicable(self):
        config = EvalConfig(device="cpu")
        cls = get_evaluator_class("loop_quality")
        evaluator = cls(config=config)
        import numpy as np
        from vqeval.core.video_loader import VideoData, VideoMeta

        meta = VideoMeta("test.mp4", 320, 240, 30.0, 30, 1.0, "mp4v", False)
        frames = np.zeros((10, 240, 320, 3), dtype=np.uint8)
        video = VideoData(meta=meta, frames=frames, frame_indices=np.arange(10))
        assert evaluator.is_applicable(video)

    def test_text_evaluator_not_applicable_without_prompt(self):
        config = EvalConfig(prompt=None, device="cpu")
        cls = get_evaluator_class("text_alignment")
        evaluator = cls(config=config)
        import numpy as np
        from vqeval.core.video_loader import VideoData, VideoMeta

        meta = VideoMeta("test.mp4", 320, 240, 30.0, 30, 1.0, "mp4v", False)
        frames = np.zeros((10, 240, 320, 3), dtype=np.uint8)
        video = VideoData(meta=meta, frames=frames, frame_indices=np.arange(10))
        assert not evaluator.is_applicable(video)

    def test_text_evaluator_applicable_with_prompt(self):
        config = EvalConfig(prompt="a cat", device="cpu")
        cls = get_evaluator_class("text_alignment")
        evaluator = cls(config=config)
        import numpy as np
        from vqeval.core.video_loader import VideoData, VideoMeta

        meta = VideoMeta("test.mp4", 320, 240, 30.0, 30, 1.0, "mp4v", False)
        frames = np.zeros((10, 240, 320, 3), dtype=np.uint8)
        video = VideoData(meta=meta, frames=frames, frame_indices=np.arange(10))
        assert evaluator.is_applicable(video)


class TestBaseEvaluator:
    def test_make_result_clamps_score(self):
        config = EvalConfig(device="cpu")
        cls = get_evaluator_class("spatial_quality")
        evaluator = cls(config=config)

        result = evaluator._make_result(150.0, {"test": 1})
        assert result.score == 100.0

        result = evaluator._make_result(-10.0, {"test": 1})
        assert result.score == 0.0

    def test_make_result_verdict(self):
        config = EvalConfig(device="cpu")
        cls = get_evaluator_class("spatial_quality")
        evaluator = cls(config=config)

        result = evaluator._make_result(95.0, {})
        assert result.verdict == "excellent"

        result = evaluator._make_result(25.0, {})
        assert result.verdict == "bad"

        result = evaluator._make_result(35.0, {})
        assert result.verdict == "poor"

    def test_make_result_traces_default_empty(self):
        config = EvalConfig(device="cpu")
        cls = get_evaluator_class("spatial_quality")
        evaluator = cls(config=config)

        result = evaluator._make_result(75.0, {"m": 1})
        assert result.traces == {}

    def test_make_result_traces_passed_through(self):
        config = EvalConfig(device="cpu")
        cls = get_evaluator_class("spatial_quality")
        evaluator = cls(config=config)

        traces = {"per_frame": [10, 20, 30]}
        result = evaluator._make_result(75.0, {"m": 1}, traces=traces)
        assert result.traces == traces
