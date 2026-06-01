"""Dimension 3: Repetition / attention-sink loop detection.

Detects when an AI-generated video falls into repetitive cyclic motion —
a common failure mode where the model enters an "attention sink" and the
video degenerates into a GIF-like repeating pattern. Higher score = less
repetition = better quality.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from vqeval.core.config import EvalConfig
from vqeval.core.video_loader import VideoData
from vqeval.evaluators.base import BaseEvaluator, EvalResult, register_evaluator


@register_evaluator
class LoopQualityEvaluator(BaseEvaluator):
    """Detects unwanted repetitive/cyclic motion in AI-generated video.

    A video that loops or repeats itself scores LOW (bad).
    A video with continuous, non-repeating motion scores HIGH (good).
    """

    dimension_name = "loop_quality"
    requires_gpu = True

    def is_applicable(self, video: VideoData) -> bool:
        return True

    def evaluate(self, video: VideoData) -> EvalResult:
        n = video.num_frames
        if n < 10:
            return self._make_result(80.0, {"note": "too few frames for repetition analysis"})

        tensors = video.get_tensors(self.device)

        # 1. Embedding self-similarity matrix analysis
        cycle_score, cycle_period, sim_matrix_stats, sim_matrix = self._detect_cyclic_embeddings(tensors, video)

        # 2. Frame-level duplicate / near-duplicate detection
        duplicate_ratio, longest_repeat = self._detect_frame_repeats(tensors, video)

        # 3. Optical flow periodicity
        flow_periodicity, flow_mag_series, flow_indices = self._detect_flow_periodicity(tensors)

        # 4. Temporal autocorrelation of frame differences
        autocorr_peak, autocorr_strength, autocorr_curve = self._detect_temporal_autocorrelation(tensors, video)

        # Composite: higher = less repetition = better
        # Each sub-metric is 0-1 where 1 = bad (repetitive)
        repetition_severity = (
            0.35 * cycle_score
            + 0.25 * duplicate_ratio
            + 0.20 * flow_periodicity
            + 0.20 * autocorr_strength
        )

        score = max(0, (1.0 - repetition_severity) * 100)

        metrics = {
            "repetition_severity": repetition_severity,
            "cycle_detected": cycle_score > 0.4,
            "cycle_period_frames": cycle_period,
            "duplicate_frame_ratio": duplicate_ratio,
            "longest_repeat_frames": longest_repeat,
            "flow_periodicity": flow_periodicity,
            "autocorrelation_peak_lag": autocorr_peak,
            "autocorrelation_strength": autocorr_strength,
            "self_similarity_offdiag_mean": sim_matrix_stats.get("offdiag_mean", 0),
            "self_similarity_offdiag_std": sim_matrix_stats.get("offdiag_std", 0),
        }

        traces = {
            "self_similarity_matrix": sim_matrix,
            "flow_magnitude_series": np.array(flow_mag_series) if flow_mag_series else np.array([]),
            "flow_pair_indices": flow_indices,
            "autocorrelation_curve": np.array(autocorr_curve) if autocorr_curve else np.array([]),
        }

        return self._make_result(score, metrics, traces)

    @torch.no_grad()
    def _detect_cyclic_embeddings(
        self, tensors: torch.Tensor, video: VideoData
    ) -> tuple[float, int, dict, np.ndarray]:
        """Detect cyclic patterns via CLIP embedding self-similarity matrix.

        Build a NxN cosine similarity matrix of frame embeddings.
        Cyclic/looping video shows diagonal stripe patterns in this matrix
        (high similarity at regular offsets from the main diagonal).
        """
        clip_embeds = video.cache_get("clip_embeddings")
        if clip_embeds is None:
            clip_embeds = self.model_registry.compute_clip_image_embeddings(tensors)
            video.cache_set("clip_embeddings", clip_embeds)

        n = clip_embeds.shape[0]

        # NxN similarity matrix
        sim_matrix = F.cosine_similarity(
            clip_embeds.unsqueeze(0), clip_embeds.unsqueeze(1), dim=-1
        ).cpu().numpy()

        # Analyze off-diagonal bands for periodic peaks
        # For each offset k, compute mean similarity along the k-th diagonal
        max_offset = n // 2
        band_means = []
        for k in range(1, max_offset + 1):
            diag = np.diagonal(sim_matrix, offset=k)
            band_means.append(float(np.mean(diag)))

        if not band_means:
            return 0.0, 0, {}, sim_matrix

        band_means_arr = np.array(band_means)

        # Find peaks in the band means — these indicate cycle periods
        # A repeating video will have high similarity at offset = cycle_length
        overall_mean = float(np.mean(band_means_arr))
        overall_std = float(np.std(band_means_arr))

        # Find the highest off-diagonal band (excluding very short offsets
        # which are naturally high due to temporal smoothness)
        min_cycle = max(3, n // 10)
        if min_cycle < len(band_means_arr):
            search_range = band_means_arr[min_cycle:]
            peak_offset = int(np.argmax(search_range)) + min_cycle
            peak_value = float(search_range.max())
        else:
            peak_offset = 0
            peak_value = overall_mean

        # Cycle score: how much the peak stands out above baseline
        if overall_std > 0:
            cycle_z = (peak_value - overall_mean) / overall_std
            cycle_score = min(1.0, max(0.0, cycle_z / 3.0))
        else:
            # Very uniform similarity = everything looks the same = bad
            cycle_score = min(1.0, max(0.0, (overall_mean - 0.8) / 0.2))

        # Also penalize if overall off-diagonal similarity is very high
        # (video barely changes at all — near-static or fully repeating)
        offdiag_mask = ~np.eye(n, dtype=bool)
        offdiag_mean = float(sim_matrix[offdiag_mask].mean())
        if offdiag_mean > 0.95:
            cycle_score = max(cycle_score, 0.8)
        elif offdiag_mean > 0.90:
            cycle_score = max(cycle_score, 0.5)

        stats = {
            "offdiag_mean": offdiag_mean,
            "offdiag_std": float(sim_matrix[offdiag_mask].std()),
        }

        return cycle_score, peak_offset + 1, stats, sim_matrix

    @torch.no_grad()
    def _detect_frame_repeats(
        self, tensors: torch.Tensor, video: VideoData
    ) -> tuple[float, int]:
        """Detect near-duplicate frames using CLIP embedding similarity.

        Counts frame pairs (beyond immediate neighbors) that are
        near-identical, indicating the video is repeating content.
        """
        clip_embeds = video.cache_get("clip_embeddings")
        if clip_embeds is None:
            clip_embeds = self.model_registry.compute_clip_image_embeddings(tensors)
            video.cache_set("clip_embeddings", clip_embeds)

        n = clip_embeds.shape[0]
        if n < 4:
            return 0.0, 0

        # Similarity matrix
        sim_matrix = F.cosine_similarity(
            clip_embeds.unsqueeze(0), clip_embeds.unsqueeze(1), dim=-1
        ).cpu().numpy()

        # Threshold for "near-duplicate": very high similarity
        # between non-adjacent frames
        dup_threshold = 0.97
        min_gap = max(3, n // 10)

        dup_count = 0
        total_pairs = 0
        longest_repeat = 0

        for i in range(n):
            for j in range(i + min_gap, n):
                total_pairs += 1
                if sim_matrix[i, j] > dup_threshold:
                    dup_count += 1

        # Find longest consecutive stretch where frame[i] ~ frame[i + period]
        for period in range(min_gap, n // 2 + 1):
            streak = 0
            for i in range(n - period):
                if sim_matrix[i, i + period] > dup_threshold:
                    streak += 1
                    longest_repeat = max(longest_repeat, streak)
                else:
                    streak = 0

        dup_ratio = dup_count / max(1, total_pairs)
        return min(1.0, dup_ratio * 3.0), longest_repeat

    @torch.no_grad()
    def _detect_flow_periodicity(self, tensors: torch.Tensor) -> tuple[float, list]:
        """Detect periodic patterns in optical flow magnitude over time.

        If flow magnitudes follow a repeating cycle, the motion itself
        is periodic (e.g., same arm swing repeated).
        """
        n = tensors.shape[0]
        if n < 6:
            return 0.0, [], []

        flow_mags = []
        flow_indices = []
        step = max(1, n // 30)
        for i in range(0, n - 1, step):
            flow = self.model_registry.compute_optical_flow(tensors[i], tensors[i + 1])
            mag = float(torch.sqrt(flow[0, 0] ** 2 + flow[0, 1] ** 2).mean().item())
            flow_mags.append(mag)
            flow_indices.append(i)

        if len(flow_mags) < 4:
            return 0.0, flow_mags, flow_indices

        # Use FFT to detect periodicity in the flow magnitude signal
        signal = np.array(flow_mags)
        signal = signal - signal.mean()

        if np.std(signal) < 1e-6:
            return 0.0, flow_mags, flow_indices

        fft = np.fft.rfft(signal)
        power = np.abs(fft) ** 2

        if len(power) < 3:
            return 0.0, flow_mags, flow_indices

        # Skip DC component
        power_no_dc = power[1:]
        total_power = float(np.sum(power_no_dc))
        if total_power < 1e-6:
            return 0.0, flow_mags, flow_indices

        # Peak frequency dominance: one frequency dominates = periodic motion
        peak_power = float(np.max(power_no_dc))
        dominance = peak_power / total_power

        return min(1.0, max(0.0, (dominance - 0.3) / 0.4)), flow_mags, flow_indices

    @torch.no_grad()
    def _detect_temporal_autocorrelation(
        self, tensors: torch.Tensor, video: VideoData
    ) -> tuple[int, float, list]:
        """Compute autocorrelation of frame-to-frame differences.

        High autocorrelation at non-zero lags indicates the sequence
        of frame changes repeats itself.
        """
        clip_embeds = video.cache_get("clip_embeddings")
        if clip_embeds is None:
            clip_embeds = self.model_registry.compute_clip_image_embeddings(tensors)
            video.cache_set("clip_embeddings", clip_embeds)

        n = clip_embeds.shape[0]
        if n < 6:
            return 0, 0.0, []

        # Frame-to-frame difference norms
        diffs = (clip_embeds[1:] - clip_embeds[:-1]).norm(dim=-1).cpu().numpy()

        if len(diffs) < 4:
            return 0, 0.0, []

        diffs = diffs - diffs.mean()
        norm = float(np.sum(diffs ** 2))
        if norm < 1e-8:
            return 1, 0.8, []

        # Autocorrelation at various lags
        max_lag = len(diffs) // 2
        autocorr = []
        for lag in range(1, max_lag + 1):
            c = float(np.sum(diffs[:len(diffs) - lag] * diffs[lag:])) / norm
            autocorr.append(c)

        if not autocorr:
            return 0, 0.0, []

        autocorr_arr = np.array(autocorr)

        # Find strongest positive peak (skip lag 1-2, naturally high)
        search_start = min(2, len(autocorr_arr) - 1)
        search = autocorr_arr[search_start:]
        if len(search) == 0:
            return 0, 0.0, autocorr

        peak_idx = int(np.argmax(search)) + search_start
        peak_value = float(search.max())

        strength = max(0.0, min(1.0, peak_value))
        return peak_idx + 1, strength, autocorr
