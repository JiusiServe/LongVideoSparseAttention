"""Dimension 2: Temporal coherence evaluation."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from vqeval.core.config import EvalConfig
from vqeval.core.video_loader import VideoData
from vqeval.evaluators.base import BaseEvaluator, EvalResult, register_evaluator
from vqeval.normalization.calibration import MetricNormalizer


@register_evaluator
class TemporalCoherenceEvaluator(BaseEvaluator):
    """Evaluates temporal coherence: flickering, motion smoothness,
    subject/background consistency, and warping detection."""

    dimension_name = "temporal_coherence"
    requires_gpu = True

    def __init__(self, config: EvalConfig, model_registry=None):
        super().__init__(config, model_registry)
        self.normalizer = MetricNormalizer()

    def evaluate(self, video: VideoData) -> EvalResult:
        n = video.num_frames
        if n < 3:
            return self._make_result(50.0, {"note": "too few frames for temporal analysis"})

        tensors = video.get_tensors(self.device)

        # 1. Temporal flickering via CLIP embedding differences
        flickering_score, clip_embeds = self._compute_flickering(tensors, video)

        # 2. Motion smoothness via optical flow
        motion_score, flow_data, flow_indices = self._compute_motion_smoothness(tensors)

        # 3. Subject consistency via DINO embeddings
        subject_consistency = self._compute_subject_consistency(tensors, video)

        # 4. Background consistency via CLIP embeddings
        background_consistency = self._compute_background_consistency(clip_embeds)

        # 5. Warping detection via keypoint tracking
        warping_events, warping_severity, worst_transitions = self._detect_warping(video)

        # Composite score
        warp_penalty = min(warping_events * 5, 30)
        score = (
            0.25 * flickering_score
            + 0.30 * motion_score
            + 0.20 * (subject_consistency * 100)
            + 0.15 * (background_consistency * 100)
            + 0.10 * max(0, 100 - warp_penalty)
        )

        # Clamp contributions from consistency scores
        score = min(100, max(0, score))

        metrics = {
            "flickering_score": round(flickering_score, 1),
            "motion_smoothness": round(motion_score, 1),
            "subject_consistency": round(subject_consistency, 4),
            "background_consistency": round(background_consistency, 4),
            "warping_events": warping_events,
            "warping_severity_mean": round(warping_severity, 4),
            "worst_transition_idx": worst_transitions,
        }

        # Build traces from CLIP consecutive similarities
        traces = {
            "flow_magnitudes": flow_data,
            "flow_pair_indices": flow_indices,
        }
        if clip_embeds is not None and clip_embeds.shape[0] > 1:
            sims = F.cosine_similarity(clip_embeds[:-1], clip_embeds[1:], dim=-1)
            traces["clip_consecutive_sims"] = sims.cpu().numpy()

        return self._make_result(score, metrics, traces)

    @torch.no_grad()
    def _compute_flickering(
        self, tensors: torch.Tensor, video: VideoData
    ) -> tuple[float, torch.Tensor]:
        """Compute flickering score using CLIP frame embedding differences."""
        # Check cache first
        clip_embeds = video.cache_get("clip_embeddings")
        if clip_embeds is None:
            clip_embeds = self.model_registry.compute_clip_image_embeddings(tensors)
            video.cache_set("clip_embeddings", clip_embeds)

        # Cosine similarities between consecutive frames
        sims = F.cosine_similarity(clip_embeds[:-1], clip_embeds[1:], dim=-1)
        sim_diffs = (1.0 - sims).cpu().numpy()

        # Low variance in differences = smooth = good
        mean_diff = float(np.mean(sim_diffs))
        std_diff = float(np.std(sim_diffs))

        # Normalize: small mean_diff and small std = high score
        # Empirically, mean_diff < 0.02 is very smooth, > 0.1 is flickery
        score = max(0, min(100, (1.0 - mean_diff / 0.15) * 100))
        # Penalize high variance (inconsistent transitions)
        score -= min(20, std_diff / 0.05 * 10)

        return max(0, score), clip_embeds

    @torch.no_grad()
    def _compute_motion_smoothness(
        self, tensors: torch.Tensor
    ) -> tuple[float, list]:
        """Compute motion smoothness using optical flow."""
        n = tensors.shape[0]
        flow_magnitudes = []
        flow_directions = []

        # Compute flow for consecutive pairs (subsample if too many frames)
        step = max(1, n // 30)  # Limit to ~30 flow computations
        indices = list(range(0, n - 1, step))
        flow_pair_indices = []

        for i in indices:
            flow = self.model_registry.compute_optical_flow(tensors[i], tensors[i + 1])
            flow_np = flow[0].cpu().numpy()  # (2, H, W)
            mag = np.sqrt(flow_np[0] ** 2 + flow_np[1] ** 2)
            flow_magnitudes.append(float(np.mean(mag)))
            flow_pair_indices.append(i)

            # Direction histogram
            angle = np.arctan2(flow_np[1], flow_np[0])
            flow_directions.append(float(np.mean(np.abs(angle))))

        if len(flow_magnitudes) < 2:
            return 75.0, flow_magnitudes, flow_pair_indices

        mag_arr = np.array(flow_magnitudes)

        # Flow magnitude variance (high = jerky)
        mag_var = float(np.var(mag_arr))

        # Acceleration (second derivative of magnitude)
        if len(mag_arr) > 2:
            accel = np.diff(mag_arr, n=2)
            accel_max = float(np.max(np.abs(accel)))
        else:
            accel_max = 0.0

        # Flow consistency: cosine similarity between consecutive flow magnitudes
        if len(mag_arr) > 1:
            consistency = 1.0 - min(1.0, mag_var / (float(np.mean(mag_arr)) + 1e-6))
        else:
            consistency = 1.0

        # Score: low variance and low acceleration = smooth
        var_score = max(0, 100 - mag_var * 10)
        accel_score = max(0, 100 - accel_max * 5)
        score = 0.5 * var_score + 0.3 * accel_score + 0.2 * (consistency * 100)

        return max(0, min(100, score)), flow_magnitudes, flow_pair_indices

    @torch.no_grad()
    def _compute_subject_consistency(
        self, tensors: torch.Tensor, video: VideoData
    ) -> float:
        """Compute subject consistency using DINO embeddings."""
        dino_embeds = video.cache_get("dino_embeddings")
        if dino_embeds is None:
            dino_embeds = self.model_registry.compute_dino_embeddings(tensors)
            video.cache_set("dino_embeddings", dino_embeds)

        if dino_embeds.shape[0] < 2:
            return 1.0

        # Cosine similarity between all consecutive pairs
        sims = F.cosine_similarity(dino_embeds[:-1], dino_embeds[1:], dim=-1)
        mean_sim = float(sims.mean().item())
        min_sim = float(sims.min().item())

        # Use mean as the subject consistency score (already 0-1)
        return mean_sim

    @torch.no_grad()
    def _compute_background_consistency(self, clip_embeds: torch.Tensor) -> float:
        """Compute background consistency using CLIP embeddings.

        Uses full-frame CLIP embeddings as a proxy (full background segmentation
        would be more accurate but much slower).
        """
        if clip_embeds.shape[0] < 2:
            return 1.0

        sims = F.cosine_similarity(clip_embeds[:-1], clip_embeds[1:], dim=-1)
        return float(sims.mean().item())

    def _detect_warping(
        self, video: VideoData
    ) -> tuple[int, float, list]:
        """Detect warping artifacts using ORB keypoint tracking.

        Tracks structural keypoints across frames and flags sudden
        large displacements in regions that should be static.
        """
        frames_gray = [
            cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in video.frames
        ]
        n = len(frames_gray)
        if n < 2:
            return 0, 0.0, []

        orb = cv2.ORB_create(nfeatures=500)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        warping_events = 0
        severities = []
        worst_transitions = []

        step = max(1, n // 20)
        for i in range(0, n - 1, step):
            kp1, des1 = orb.detectAndCompute(frames_gray[i], None)
            kp2, des2 = orb.detectAndCompute(frames_gray[i + 1], None)

            if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                continue

            matches = bf.match(des1, des2)
            if len(matches) < 10:
                continue

            # Compute displacement statistics
            displacements = []
            for m in matches:
                pt1 = kp1[m.queryIdx].pt
                pt2 = kp2[m.trainIdx].pt
                dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
                displacements.append(dist)

            displacements = np.array(displacements)
            median_disp = np.median(displacements)
            p95_disp = np.percentile(displacements, 95)

            # Warping: when outlier keypoints move much more than the median
            if median_disp > 0:
                ratio = p95_disp / (median_disp + 1e-6)
            else:
                ratio = 0

            # A ratio > 5 suggests some keypoints warped unnaturally
            if ratio > 5.0 and p95_disp > 20:
                warping_events += 1
                severities.append(float(ratio / 10.0))
                orig_i = int(video.frame_indices[i])
                orig_j = int(video.frame_indices[min(i + 1, n - 1)])
                worst_transitions.append([orig_i, orig_j])

        mean_severity = float(np.mean(severities)) if severities else 0.0
        return warping_events, mean_severity, worst_transitions[:5]
