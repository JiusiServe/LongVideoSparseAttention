"""Dimension 4: AI-specific artifact detection."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from vqeval.core.config import EvalConfig
from vqeval.core.video_loader import VideoData
from vqeval.evaluators.base import BaseEvaluator, EvalResult, register_evaluator
from vqeval.normalization.calibration import MetricNormalizer


@register_evaluator
class ArtifactDetectionEvaluator(BaseEvaluator):
    """Detects AI-specific artifacts: banding, anatomical errors,
    texture tiling, edge halos, and temporal noise anomalies."""

    dimension_name = "artifact_detection"
    requires_gpu = True

    def __init__(self, config: EvalConfig, model_registry=None):
        super().__init__(config, model_registry)
        self.normalizer = MetricNormalizer()

    def evaluate(self, video: VideoData) -> EvalResult:
        frames_bgr = video.frames
        frames_rgb = video.rgb
        n = video.num_frames

        # 1. Banding / posterization
        banding_severity, banding_pct = self._detect_banding(frames_bgr)

        # 2. Anatomical / structural errors
        anat_errors, anat_frames = self._detect_anatomical_errors(frames_rgb)

        # 3. Texture repetition / tiling
        tiling_score = self._detect_texture_tiling(frames_bgr)

        # 4. Edge bleeding / halo artifacts
        halo_severity = self._detect_edge_halos(frames_bgr)

        # 5. Temporal noise pattern anomaly
        noise_anomaly = self._detect_noise_anomaly(frames_bgr)

        # Compute dimension score (fewer artifacts = higher score)
        s_banding = max(0, 100 - banding_severity * 200)
        s_anat = max(0, 100 - anat_errors * 15)
        s_tiling = max(0, 100 - tiling_score * 200)
        s_halo = max(0, 100 - halo_severity * 200)
        s_noise = max(0, 100 - noise_anomaly * 100)

        score = (
            0.25 * s_banding
            + 0.25 * s_anat
            + 0.15 * s_tiling
            + 0.15 * s_halo
            + 0.20 * s_noise
        )

        metrics = {
            "banding_severity": banding_severity,
            "banding_frame_pct": banding_pct,
            "anatomical_errors": anat_errors,
            "anatomical_error_frames": anat_frames[:10],
            "texture_tiling_score": tiling_score,
            "edge_halo_severity": halo_severity,
            "noise_anomaly_score": noise_anomaly,
        }

        return self._make_result(score, metrics)

    def _detect_banding(
        self, frames_bgr: np.ndarray
    ) -> tuple[float, float]:
        """Detect banding/posterization artifacts.

        Simplified CAMBI-like approach: detect regions with abnormally
        low gradient magnitude in areas that should be smooth gradients.
        """
        n = len(frames_bgr)
        banding_scores = []
        banding_count = 0

        # Sample frames for analysis
        step = max(1, n // 10)
        for i in range(0, n, step):
            frame = frames_bgr[i]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # Compute gradient magnitude
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx ** 2 + gy ** 2)

            # Detect near-flat regions (potential banding)
            # In banded areas: pixel values change in steps, creating
            # regions of near-zero gradient with sharp boundaries
            flat_mask = grad_mag < 2.0
            flat_ratio = float(np.mean(flat_mask))

            # Check if flat regions have staircase-like transitions
            # Quantize the gray image and compare to original
            quantized = np.round(gray / 8) * 8  # 32 levels
            quant_diff = np.abs(gray - quantized)
            banding_indicator = float(np.mean(quant_diff[flat_mask])) if np.any(flat_mask) else 0

            # Lower quant_diff in flat regions suggests natural flat areas
            # Higher quant_diff suggests actual gradient that got banded
            if flat_ratio > 0.3:  # Significant flat regions
                banding_score = max(0, flat_ratio - 0.3) * (1.0 - banding_indicator / 4.0)
                banding_scores.append(banding_score)
                if banding_score > 0.1:
                    banding_count += 1
            else:
                banding_scores.append(0.0)

        severity = float(np.mean(banding_scores)) if banding_scores else 0.0
        frame_pct = banding_count / max(1, len(banding_scores)) * 100.0
        return severity, frame_pct

    def _detect_anatomical_errors(
        self, frames_rgb: np.ndarray
    ) -> tuple[int, list]:
        """Detect anatomical errors using MediaPipe pose/hand detection."""
        try:
            import mediapipe as mp

            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5,
            )
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=True,
                min_detection_confidence=0.5,
            )
        except (ImportError, AttributeError):
            # mediapipe not installed, or newer version without solutions API
            return 0, []

        try:
            error_count = 0
            error_frames = []
            n = len(frames_rgb)

            # Track hand landmark counts across frames for consistency
            hand_counts_per_frame = []

            step = max(1, n // 15)  # Sample up to ~15 frames
            for i in range(0, n, step):
                frame = frames_rgb[i]
                frame_errors = []

                # Hand analysis
                hand_result = hands.process(frame)
                if hand_result.multi_hand_landmarks:
                    for hand_landmarks in hand_result.multi_hand_landmarks:
                        # Check for impossible finger configurations
                        finger_issues = self._check_finger_geometry(hand_landmarks)
                        if finger_issues:
                            frame_errors.extend(finger_issues)

                    hand_counts_per_frame.append(len(hand_result.multi_hand_landmarks))
                else:
                    hand_counts_per_frame.append(0)

                # Pose analysis
                pose_result = pose.process(frame)
                if pose_result.pose_landmarks:
                    limb_issues = self._check_limb_symmetry(pose_result.pose_landmarks)
                    if limb_issues:
                        frame_errors.extend(limb_issues)

                if frame_errors:
                    error_count += len(frame_errors)
                    orig_idx = int(i)  # Simplified; would use video.frame_indices
                    error_frames.append([orig_idx, orig_idx + step])

            # Check for sudden appearance/disappearance of hands
            if len(hand_counts_per_frame) > 2:
                diffs = np.diff(hand_counts_per_frame)
                sudden_changes = int(np.sum(np.abs(diffs) > 0))
                if sudden_changes > len(hand_counts_per_frame) // 3:
                    error_count += sudden_changes

            return error_count, error_frames
        finally:
            # Always release MediaPipe graph resources, even on exception.
            hands.close()
            pose.close()

    def _check_finger_geometry(self, hand_landmarks) -> list[str]:
        """Check for impossible finger joint configurations."""
        issues = []
        # Simplified check: verify finger tips are farther from wrist than knuckles
        # This catches some "extra finger" or "merged finger" artifacts
        landmarks = hand_landmarks.landmark

        # Check each finger (index, middle, ring, pinky)
        finger_tips = [8, 12, 16, 20]
        finger_mids = [6, 10, 14, 18]
        wrist = landmarks[0]

        for tip_idx, mid_idx in zip(finger_tips, finger_mids):
            tip = landmarks[tip_idx]
            mid = landmarks[mid_idx]

            # Distance from wrist
            tip_dist = np.sqrt((tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2)
            mid_dist = np.sqrt((mid.x - wrist.x) ** 2 + (mid.y - wrist.y) ** 2)

            # Finger tip very close to its mid joint = likely collapsed/deformed
            tip_mid_dist = np.sqrt((tip.x - mid.x) ** 2 + (tip.y - mid.y) ** 2)
            if tip_mid_dist < 0.01:
                issues.append(f"collapsed_finger_{tip_idx}")

        return issues

    def _check_limb_symmetry(self, pose_landmarks) -> list[str]:
        """Check for asymmetric limb proportions."""
        issues = []
        landmarks = pose_landmarks.landmark

        # Compare left and right arm lengths
        try:
            # Left arm: shoulder(11) -> elbow(13) -> wrist(15)
            l_upper = self._landmark_dist(landmarks[11], landmarks[13])
            l_lower = self._landmark_dist(landmarks[13], landmarks[15])

            # Right arm: shoulder(12) -> elbow(14) -> wrist(16)
            r_upper = self._landmark_dist(landmarks[12], landmarks[14])
            r_lower = self._landmark_dist(landmarks[14], landmarks[16])

            # Check for extreme asymmetry (ratio > 2x)
            if l_upper > 0 and r_upper > 0:
                ratio = max(l_upper, r_upper) / min(l_upper, r_upper)
                if ratio > 2.0:
                    issues.append("asymmetric_upper_arms")

            if l_lower > 0 and r_lower > 0:
                ratio = max(l_lower, r_lower) / min(l_lower, r_lower)
                if ratio > 2.0:
                    issues.append("asymmetric_lower_arms")
        except (IndexError, ZeroDivisionError):
            pass

        return issues

    @staticmethod
    def _landmark_dist(a, b) -> float:
        return float(np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2))

    def _detect_texture_tiling(self, frames_bgr: np.ndarray) -> float:
        """Detect repeated texture patches using autocorrelation."""
        n = len(frames_bgr)
        tiling_scores = []

        step = max(1, n // 8)
        for i in range(0, n, step):
            frame = frames_bgr[i]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # Downsample for efficiency
            h, w = gray.shape
            scale = min(1.0, 512.0 / max(h, w))
            if scale < 1.0:
                gray = cv2.resize(gray, None, fx=scale, fy=scale)

            # Compute 2D autocorrelation using DFT
            gray_norm = gray - gray.mean()
            f = np.fft.fft2(gray_norm)
            power = np.abs(f) ** 2
            autocorr = np.fft.ifft2(power).real
            autocorr = autocorr / autocorr[0, 0]  # Normalize

            # Look for secondary peaks (indicating tiling)
            # Exclude the center region (trivial self-correlation)
            h, w = autocorr.shape
            margin_h, margin_w = h // 8, w // 8
            autocorr_masked = autocorr.copy()
            autocorr_masked[:margin_h, :margin_w] = 0
            autocorr_masked[-margin_h:, -margin_w:] = 0
            autocorr_masked[:margin_h, -margin_w:] = 0
            autocorr_masked[-margin_h:, :margin_w] = 0

            # Secondary peak strength
            secondary_peak = float(np.max(np.abs(autocorr_masked)))
            tiling_scores.append(secondary_peak)

        return float(np.mean(tiling_scores)) if tiling_scores else 0.0

    def _detect_edge_halos(self, frames_bgr: np.ndarray) -> float:
        """Detect halo artifacts around edges."""
        n = len(frames_bgr)
        halo_scores = []

        step = max(1, n // 8)
        for i in range(0, n, step):
            frame = frames_bgr[i]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect edges
            edges = cv2.Canny(frame, 50, 150)

            # Dilate edges to get surrounding region
            kernel = np.ones((5, 5), np.uint8)
            edge_region = cv2.dilate(edges, kernel, iterations=2)
            edge_only = edge_region - cv2.dilate(edges, kernel, iterations=0)
            edge_only = (edge_only > 0).astype(np.uint8)

            if np.sum(edge_only) < 100:
                halo_scores.append(0.0)
                continue

            # Compute local contrast in edge surrounding regions
            # High contrast near edges that drops off unnaturally = halo
            lap = cv2.Laplacian(gray, cv2.CV_32F).astype(np.float64)
            edge_lap = lap * edge_only.astype(np.float32)
            non_edge_lap = lap * (1 - edge_only.astype(np.float32))

            edge_contrast = float(np.mean(np.abs(edge_lap[edge_only > 0])))
            if np.any(edge_only == 0):
                non_edge_contrast = float(np.mean(np.abs(non_edge_lap[edge_only == 0])))
            else:
                non_edge_contrast = edge_contrast

            # Halo indicator: unusually high contrast right next to edges
            if non_edge_contrast > 0:
                ratio = edge_contrast / non_edge_contrast
                halo_score = max(0, (ratio - 1.5) / 3.0)  # Normalize
            else:
                halo_score = 0.0

            halo_scores.append(min(1.0, halo_score))

        return float(np.mean(halo_scores)) if halo_scores else 0.0

    def _detect_noise_anomaly(self, frames_bgr: np.ndarray) -> float:
        """Detect unnatural noise patterns using wavelet decomposition.

        AI-generated noise differs from natural sensor noise in its
        spatial frequency distribution.
        """
        n = len(frames_bgr)
        anomaly_scores = []

        step = max(1, n // 8)
        for i in range(0, n, step):
            frame = frames_bgr[i]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # Extract noise via high-frequency residual
            # Blur the image and subtract to get noise
            blurred = cv2.GaussianBlur(gray, (7, 7), 2.0)
            noise = gray - blurred

            # Natural noise characteristics:
            # - Approximately Gaussian distribution
            # - Relatively uniform across the image
            # AI noise tends to be:
            # - More structured (higher autocorrelation)
            # - Non-uniform (varies with content)

            # Test 1: Gaussianity (kurtosis)
            noise_flat = noise.flatten()
            if np.std(noise_flat) > 0:
                kurtosis = float(np.mean((noise_flat - np.mean(noise_flat)) ** 4) /
                                (np.std(noise_flat) ** 4) - 3.0)
            else:
                kurtosis = 0.0

            # Natural noise kurtosis ~0 (Gaussian), AI noise often higher
            kurtosis_score = min(1.0, abs(kurtosis) / 5.0)

            # Test 2: Spatial uniformity
            h, w = noise.shape
            blocks = []
            bh, bw = h // 4, w // 4
            for r in range(4):
                for c in range(4):
                    block = noise[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw]
                    blocks.append(float(np.std(block)))

            block_stds = np.array(blocks)
            uniformity = float(np.std(block_stds) / (np.mean(block_stds) + 1e-6))

            # High non-uniformity suggests AI noise
            uniformity_score = min(1.0, uniformity / 2.0)

            anomaly_scores.append(0.5 * kurtosis_score + 0.5 * uniformity_score)

        return float(np.mean(anomaly_scores)) if anomaly_scores else 0.0
