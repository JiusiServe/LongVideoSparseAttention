"""Dimension 1: Per-frame image quality (spatial fidelity)."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from vqeval.core.config import EvalConfig, LAPLACIAN_BLUR_THRESHOLD
from vqeval.core.video_loader import VideoData
from vqeval.evaluators.base import BaseEvaluator, EvalResult, register_evaluator
from vqeval.normalization.calibration import MetricNormalizer


@register_evaluator
class SpatialQualityEvaluator(BaseEvaluator):
    """Evaluates per-frame image quality using BRISQUE, NIQE, CLIP-IQA, and blur detection."""

    dimension_name = "spatial_quality"
    requires_gpu = True

    def __init__(self, config: EvalConfig, model_registry=None):
        super().__init__(config, model_registry)
        self.normalizer = MetricNormalizer()
        self.blur_threshold = LAPLACIAN_BLUR_THRESHOLD

    def evaluate(self, video: VideoData) -> EvalResult:
        frames_bgr = video.frames  # (N, H, W, 3) uint8 BGR
        frames_rgb = video.rgb
        n = video.num_frames

        # 1. BRISQUE scores
        brisque_scores = self._compute_brisque(frames_bgr)

        # 2. NIQE scores
        niqe_scores = self._compute_niqe(frames_rgb)

        # 3. CLIP-IQA scores
        clip_iqa_scores = self._compute_clip_iqa(frames_rgb)

        # 4. Sharpness / blur detection
        sharpness_values, blurry_mask = self._compute_sharpness(frames_bgr)
        blur_frame_pct = float(np.sum(blurry_mask)) / n * 100.0

        # Find worst frame (lowest combined quality)
        combined = np.zeros(n)
        if brisque_scores is not None:
            # Invert BRISQUE: lower is better, so high raw = bad
            combined += self.normalizer.normalize(
                "brisque", 0, lower_is_better=True
            ) - np.array([
                self.normalizer.normalize("brisque", b, lower_is_better=True)
                for b in brisque_scores
            ])
        worst_frame_idx = int(np.argmax(combined)) if len(combined) > 0 else 0
        worst_orig_idx = int(video.frame_indices[worst_frame_idx])

        # Flag frames with issues
        flagged = []
        for i in range(n):
            if blurry_mask[i]:
                flagged.append(int(video.frame_indices[i]))

        # Compute dimension score (weighted combination of sub-metrics)
        brisque_mean = float(np.mean(brisque_scores)) if brisque_scores is not None else 30.0
        brisque_p5 = float(np.percentile(brisque_scores, 95)) if brisque_scores is not None else 50.0
        niqe_mean = float(np.mean(niqe_scores)) if niqe_scores is not None else 5.0
        clip_iqa_mean = float(np.mean(clip_iqa_scores)) if clip_iqa_scores is not None else 0.5

        # Normalize each metric to 0-100
        s_brisque = self.normalizer.normalize("brisque", brisque_mean, lower_is_better=True)
        s_niqe = self.normalizer.normalize("niqe", niqe_mean, lower_is_better=True)
        s_clip_iqa = self.normalizer.normalize("clip_iqa", clip_iqa_mean, lower_is_better=False)
        s_blur = max(0, 100 - blur_frame_pct * 2)  # Penalize blur percentage

        # Weighted combination
        score = 0.25 * s_brisque + 0.25 * s_niqe + 0.35 * s_clip_iqa + 0.15 * s_blur

        metrics = {
            "brisque_mean": brisque_mean,
            "brisque_p5": brisque_p5,
            "niqe_mean": niqe_mean,
            "clip_iqa_mean": clip_iqa_mean,
            "blur_frame_pct": round(blur_frame_pct, 1),
            "sharpness_std": float(np.std(sharpness_values)),
            "worst_frame_idx": worst_orig_idx,
            "flagged_frames": flagged[:20],  # Cap at 20 entries
        }

        traces = {
            "brisque_per_frame": brisque_scores,
            "niqe_per_frame": niqe_scores,
            "clip_iqa_per_frame": clip_iqa_scores,
            "sharpness_per_frame": sharpness_values,
            "frame_indices": video.frame_indices,
        }

        return self._make_result(score, metrics, traces)

    def _compute_brisque(self, frames_bgr: np.ndarray) -> np.ndarray | None:
        """Compute BRISQUE score per frame using OpenCV."""
        try:
            scores = []
            for frame in frames_bgr:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                score = cv2.quality.QualityBRISQUE_compute(
                    gray,
                    cv2.quality.QualityBRISQUE_computeFeatures(gray),
                )
                scores.append(max(0, score[0]))
            return np.array(scores)
        except (cv2.error, AttributeError):
            # Fallback: use pyiqa brisque
            try:
                return self._compute_brisque_pyiqa(frames_bgr)
            except Exception:
                return None

    def _compute_brisque_pyiqa(self, frames_bgr: np.ndarray) -> np.ndarray:
        """Fallback BRISQUE using pyiqa."""
        metric = self.model_registry.get_pyiqa_metric("brisque")
        scores = []
        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            t = t.to(self.device)
            with torch.no_grad():
                s = metric(t).item()
            scores.append(s)
        return np.array(scores)

    def _compute_niqe(self, frames_rgb: np.ndarray) -> np.ndarray | None:
        """Compute NIQE score per frame using pyiqa."""
        try:
            metric = self.model_registry.get_pyiqa_metric("niqe")
            scores = []
            for frame in frames_rgb:
                t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                t = t.to(self.device)
                with torch.no_grad():
                    s = metric(t).item()
                scores.append(s)
            return np.array(scores)
        except Exception:
            return None

    def _compute_clip_iqa(self, frames_rgb: np.ndarray) -> np.ndarray | None:
        """Compute CLIP-IQA score per frame using pyiqa."""
        try:
            metric = self.model_registry.get_pyiqa_metric("clipiqa")
            scores = []
            for frame in frames_rgb:
                t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                t = t.to(self.device)
                with torch.no_grad():
                    s = metric(t).item()
                scores.append(s)
            return np.array(scores)
        except Exception:
            return None

    def _compute_sharpness(
        self, frames_bgr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Laplacian variance (sharpness) per frame and detect blur."""
        sharpness = []
        for frame in frames_bgr:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness.append(lap_var)

        sharpness_arr = np.array(sharpness)
        blurry = sharpness_arr < self.blur_threshold
        return sharpness_arr, blurry
