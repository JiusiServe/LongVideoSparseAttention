"""Abstract base class for evaluator plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from vqeval.core.config import EvalConfig, score_to_verdict
from vqeval.core.video_loader import VideoData


@dataclass
class EvalResult:
    """Result from a single evaluator."""

    dimension: str
    score: float  # 0-100
    verdict: str
    metrics: dict[str, Any] = field(default_factory=dict)
    traces: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        # traces deliberately excluded — used only by figure generator
        return {
            self.dimension: {
                "score": round(self.score, 1),
                "verdict": self.verdict,
                **{k: _round_value(v) for k, v in self.metrics.items()},
            }
        }


def _round_value(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, list):
        return [_round_value(x) for x in v]
    return v


class BaseEvaluator(ABC):
    """Base class for all quality dimension evaluators."""

    # Subclasses must set these
    dimension_name: str = ""
    requires_gpu: bool = True

    def __init__(self, config: EvalConfig, model_registry=None):
        self.config = config
        self.model_registry = model_registry
        self.device = config.device

    @abstractmethod
    def evaluate(self, video: VideoData) -> EvalResult:
        """Run evaluation on the video data and return scored result."""
        ...

    def is_applicable(self, video: VideoData) -> bool:
        """Check whether this evaluator should run for the given video/config."""
        return True

    def _make_result(
        self, score: float, metrics: dict, traces: dict | None = None
    ) -> EvalResult:
        """Helper to create a result with automatic verdict."""
        return EvalResult(
            dimension=self.dimension_name,
            score=max(0.0, min(100.0, score)),
            verdict=score_to_verdict(score),
            metrics=metrics,
            traces=traces or {},
        )


# Evaluator registry
_EVALUATOR_REGISTRY: dict[str, type[BaseEvaluator]] = {}


def register_evaluator(cls: type[BaseEvaluator]) -> type[BaseEvaluator]:
    """Decorator to register an evaluator class."""
    _EVALUATOR_REGISTRY[cls.dimension_name] = cls
    return cls


def get_evaluator_class(name: str) -> type[BaseEvaluator]:
    if name not in _EVALUATOR_REGISTRY:
        raise ValueError(
            f"Unknown evaluator: {name}. Available: {list(_EVALUATOR_REGISTRY.keys())}"
        )
    return _EVALUATOR_REGISTRY[name]


def get_all_evaluator_names() -> list[str]:
    return list(_EVALUATOR_REGISTRY.keys())
