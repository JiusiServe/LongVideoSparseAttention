"""Dimension 6: Text-video alignment evaluation (optional, requires prompt)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from vqeval.core.config import EvalConfig
from vqeval.core.video_loader import VideoData
from vqeval.evaluators.base import BaseEvaluator, EvalResult, register_evaluator
from vqeval.normalization.calibration import MetricNormalizer


@register_evaluator
class TextAlignmentEvaluator(BaseEvaluator):
    """Evaluates semantic alignment between the generation prompt and video content."""

    dimension_name = "text_alignment"
    requires_gpu = True

    def __init__(self, config: EvalConfig, model_registry=None):
        super().__init__(config, model_registry)
        self.normalizer = MetricNormalizer()

    def is_applicable(self, video: VideoData) -> bool:
        return self.config.prompt is not None and len(self.config.prompt.strip()) > 0

    def evaluate(self, video: VideoData) -> EvalResult:
        prompt = self.config.prompt
        tensors = video.get_tensors(self.device)

        # Get CLIP text embedding
        text_embed = self.model_registry.compute_clip_text_embedding(prompt)

        # Get CLIP image embeddings (use cache)
        clip_embeds = video.cache_get("clip_embeddings")
        if clip_embeds is None:
            clip_embeds = self.model_registry.compute_clip_image_embeddings(tensors)
            video.cache_set("clip_embeddings", clip_embeds)

        # Compute per-frame similarities
        # text_embed: (1, D), clip_embeds: (N, D)
        similarities = F.cosine_similarity(
            text_embed.expand(clip_embeds.shape[0], -1),
            clip_embeds,
            dim=-1,
        ).cpu().numpy()

        sim_mean = float(np.mean(similarities))
        sim_min = float(np.min(similarities))
        sim_max = float(np.max(similarities))

        # Detect drift: fit a linear trend to the similarity curve
        drift_slope, drift_significant = self._compute_drift(similarities)

        # Normalize to 0-100 score
        score = self.normalizer.normalize("clip_text_sim", sim_mean, lower_is_better=False)

        # Penalize drift (declining alignment over time)
        if drift_significant and drift_slope < 0:
            drift_penalty = min(15, abs(drift_slope) * 1000)
            score -= drift_penalty

        metrics = {
            "clip_similarity_mean": sim_mean,
            "clip_similarity_min": sim_min,
            "clip_similarity_max": sim_max,
            "drift_slope": drift_slope,
            "drift_significant": drift_significant,
        }

        traces = {
            "clip_similarity_per_frame": similarities,
        }

        return self._make_result(score, metrics, traces)

    def _compute_drift(
        self, similarities: np.ndarray
    ) -> tuple[float, bool]:
        """Fit a linear trend to the similarity curve to detect temporal drift."""
        n = len(similarities)
        if n < 3:
            return 0.0, False

        x = np.arange(n, dtype=float)
        x_norm = x / (n - 1)  # Normalize to [0, 1]

        # Linear regression
        coeffs = np.polyfit(x_norm, similarities, 1)
        slope = float(coeffs[0])

        # Significance: is the slope meaningfully different from 0?
        residuals = similarities - np.polyval(coeffs, x_norm)
        std_residual = float(np.std(residuals))

        # Slope is significant if it's larger than residual noise
        significant = abs(slope) > 2 * std_residual / np.sqrt(n)

        return slope, significant
