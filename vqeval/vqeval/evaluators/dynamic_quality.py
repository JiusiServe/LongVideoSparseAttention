"""Dimension 5: Dynamic quality - motion and visual interest assessment."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from vqeval.core.config import EvalConfig
from vqeval.core.video_loader import VideoData
from vqeval.evaluators.base import BaseEvaluator, EvalResult, register_evaluator
from vqeval.normalization.calibration import MetricNormalizer


@register_evaluator
class DynamicQualityEvaluator(BaseEvaluator):
    """Evaluates whether the video has meaningful motion and visual interest."""

    dimension_name = "dynamic_quality"
    requires_gpu = True

    def __init__(self, config: EvalConfig, model_registry=None):
        super().__init__(config, model_registry)
        self.normalizer = MetricNormalizer()
        self.flow_threshold = 1.0  # Minimum flow magnitude to count as "moving"

    def evaluate(self, video: VideoData) -> EvalResult:
        tensors = video.get_tensors(self.device)
        n = video.num_frames

        if n < 2:
            return self._make_result(50.0, {"note": "single frame"})

        # 1. Dynamic degree
        dynamic_degree = self._compute_dynamic_degree(tensors)

        # 2. Motion diversity
        motion_diversity = self._compute_motion_diversity(tensors)

        # 3. Aesthetic quality
        aesthetic_mean, aesthetic_std = self._compute_aesthetic(tensors, video)

        # Composite score
        score = (
            0.35 * (dynamic_degree * 100)
            + 0.25 * (motion_diversity * 100)
            + 0.40 * self.normalizer.normalize("aesthetic", aesthetic_mean, lower_is_better=False)
        )

        metrics = {
            "dynamic_degree": dynamic_degree,
            "motion_diversity": motion_diversity,
            "aesthetic_mean": aesthetic_mean,
            "aesthetic_std": aesthetic_std,
        }

        return self._make_result(score, metrics)  # no per-frame traces needed

    @torch.no_grad()
    def _compute_dynamic_degree(self, tensors: torch.Tensor) -> float:
        """Compute the percentage of pixels with significant optical flow."""
        n = tensors.shape[0]
        if n < 2:
            return 0.0

        moving_ratios = []
        step = max(1, n // 10)

        for i in range(0, n - 1, step):
            flow = self.model_registry.compute_optical_flow(tensors[i], tensors[i + 1])
            flow_np = flow[0].cpu().numpy()  # (2, H, W)
            mag = np.sqrt(flow_np[0] ** 2 + flow_np[1] ** 2)

            # Fraction of pixels with significant motion
            moving = float(np.mean(mag > self.flow_threshold))
            moving_ratios.append(moving)

        return float(np.mean(moving_ratios)) if moving_ratios else 0.0

    @torch.no_grad()
    def _compute_motion_diversity(self, tensors: torch.Tensor) -> float:
        """Compute entropy of optical flow direction histogram."""
        n = tensors.shape[0]
        if n < 2:
            return 0.0

        all_angles = []
        step = max(1, n // 10)

        for i in range(0, n - 1, step):
            flow = self.model_registry.compute_optical_flow(tensors[i], tensors[i + 1])
            flow_np = flow[0].cpu().numpy()
            mag = np.sqrt(flow_np[0] ** 2 + flow_np[1] ** 2)

            # Only consider pixels with significant motion
            mask = mag > self.flow_threshold
            if np.sum(mask) < 100:
                continue

            angles = np.arctan2(flow_np[1][mask], flow_np[0][mask])
            all_angles.extend(angles.flatten().tolist())

        if not all_angles:
            return 0.0

        # Compute entropy of direction histogram
        n_bins = 16
        hist, _ = np.histogram(all_angles, bins=n_bins, range=(-np.pi, np.pi))
        hist = hist.astype(float)
        total = hist.sum()
        if total == 0:
            return 0.0

        probs = hist / total
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))

        # Normalize: max entropy for 16 bins = log2(16) = 4
        return min(1.0, entropy / 4.0)

    @torch.no_grad()
    def _compute_aesthetic(
        self, tensors: torch.Tensor, video: VideoData
    ) -> tuple[float, float]:
        """Compute aesthetic quality using CLIP-based aesthetic predictor.

        Falls back to CLIP-IQA if dedicated aesthetic predictor unavailable.
        """
        try:
            metric = self.model_registry.get_pyiqa_metric("clipiqa+")
        except Exception:
            try:
                metric = self.model_registry.get_pyiqa_metric("clipiqa")
            except Exception:
                # Fallback: use CLIP embeddings as a rough proxy
                return self._aesthetic_from_clip(tensors, video)

        scores = []
        step = max(1, tensors.shape[0] // 10)
        for i in range(0, tensors.shape[0], step):
            frame = tensors[i : i + 1]
            s = metric(frame).item()
            scores.append(s)

        if not scores:
            return 5.0, 0.0

        # Scale to 1-10 range
        arr = np.array(scores)
        mean_score = float(np.mean(arr)) * 10  # pyiqa outputs 0-1
        std_score = float(np.std(arr)) * 10
        return mean_score, std_score

    @torch.no_grad()
    def _aesthetic_from_clip(
        self, tensors: torch.Tensor, video: VideoData
    ) -> tuple[float, float]:
        """Rough aesthetic estimate from CLIP embedding norms."""
        clip_embeds = video.cache_get("clip_embeddings")
        if clip_embeds is None:
            clip_embeds = self.model_registry.compute_clip_image_embeddings(tensors)
            video.cache_set("clip_embeddings", clip_embeds)

        # CLIP embedding norm can be a rough proxy for visual quality
        norms = clip_embeds.norm(dim=-1).cpu().numpy()
        # Scale to approximate 1-10 range
        mean_score = float(np.mean(norms)) * 5 + 3
        std_score = float(np.std(norms)) * 5
        return mean_score, std_score
