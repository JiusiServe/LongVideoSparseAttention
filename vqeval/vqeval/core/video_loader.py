"""Video loading, decoding, frame sampling, and caching."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from vqeval.core.config import SAMPLE_ALL_THRESHOLD_SEC, MIN_SAMPLE_FPS


@dataclass
class VideoMeta:
    """Metadata about a loaded video."""

    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    codec: str
    has_audio: bool


@dataclass
class VideoData:
    """Container for decoded video data and cached computations."""

    meta: VideoMeta
    frames: np.ndarray  # (N, H, W, 3) uint8 BGR
    frame_indices: np.ndarray  # Original frame indices that were sampled
    frames_rgb: Optional[np.ndarray] = None
    _cache: dict = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def rgb(self) -> np.ndarray:
        """Lazily convert to RGB."""
        if self.frames_rgb is None:
            self.frames_rgb = self.frames[:, :, :, ::-1].copy()
        return self.frames_rgb

    def get_tensors(self, device: str = "cuda") -> torch.Tensor:
        """Get frames as a float32 tensor (N, 3, H, W) normalized to [0,1]."""
        key = f"tensors_{device}"
        if key not in self._cache:
            t = torch.from_numpy(self.rgb).permute(0, 3, 1, 2).float() / 255.0
            self._cache[key] = t.to(device)
        return self._cache[key]

    def get_tensors_resized(
        self, size: tuple[int, int], device: str = "cuda"
    ) -> torch.Tensor:
        """Get frames resized to (H, W) as tensor."""
        key = f"tensors_{size}_{device}"
        if key not in self._cache:
            t = self.get_tensors(device)
            self._cache[key] = torch.nn.functional.interpolate(
                t, size=size, mode="bilinear", align_corners=False
            )
        return self._cache[key]

    def cache_set(self, key: str, value):
        self._cache[key] = value

    def cache_get(self, key: str, default=None):
        return self._cache.get(key, default)


class VideoLoader:
    """Loads and samples frames from video files."""

    SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}

    def __init__(self, sample_all_threshold: float = SAMPLE_ALL_THRESHOLD_SEC,
                 min_sample_fps: float = MIN_SAMPLE_FPS):
        self.sample_all_threshold = sample_all_threshold
        self.min_sample_fps = min_sample_fps

    def load(self, video_path: str) -> VideoData:
        """Load a video file and return sampled frames with metadata."""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format: {path.suffix}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        try:
            meta = self._extract_metadata(cap, str(path))
            self._validate_duration(meta)
            sample_indices = self._compute_sample_indices(meta)
            frames = self._decode_frames(cap, sample_indices)
        finally:
            cap.release()

        return VideoData(
            meta=meta,
            frames=frames,
            frame_indices=np.array(sample_indices),
        )

    def _extract_metadata(self, cap: cv2.VideoCapture, path: str) -> VideoMeta:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        duration = total_frames / fps if fps > 0 else 0.0

        # Check for audio using a separate probe (simplified)
        has_audio = self._check_audio(path)

        return VideoMeta(
            path=path,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration=duration,
            codec=codec,
            has_audio=has_audio,
        )

    def _check_audio(self, path: str) -> bool:
        """Check if video has an audio stream."""
        try:
            import subprocess
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet", "-select_streams", "a",
                    "-show_entries", "stream=codec_type", "-of",
                    "csv=p=0", path,
                ],
                capture_output=True, text=True, timeout=10,
            )
            return "audio" in result.stdout.lower()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _validate_duration(self, meta: VideoMeta):
        if meta.duration < 0.1:
            raise ValueError(f"Video too short: {meta.duration:.3f}s")

    def _compute_sample_indices(self, meta: VideoMeta) -> list[int]:
        """Determine which frames to sample based on video duration."""
        total = meta.total_frames
        if total <= 0:
            raise ValueError("Video has no frames")

        if meta.duration <= self.sample_all_threshold:
            return list(range(total))

        # Sample at min_sample_fps, plus first/last/middle
        sample_interval = max(1, int(meta.fps / self.min_sample_fps))
        indices = set(range(0, total, sample_interval))
        indices.add(0)
        indices.add(total - 1)
        indices.add(total // 2)
        return sorted(indices)

    def _decode_frames(
        self, cap: cv2.VideoCapture, indices: list[int]
    ) -> np.ndarray:
        """Decode specific frames from the video."""
        frames = []
        current_idx = 0
        indices_set = set(indices)
        sorted_indices = sorted(indices)

        # For sequential reading when sampling most frames
        if len(sorted_indices) > 0.5 * cap.get(cv2.CAP_PROP_FRAME_COUNT):
            idx_ptr = 0
            frame_idx = 0
            while idx_ptr < len(sorted_indices):
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx == sorted_indices[idx_ptr]:
                    frames.append(frame)
                    idx_ptr += 1
                frame_idx += 1
        else:
            # Seek-based reading for sparse sampling
            for idx in sorted_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)

        if not frames:
            raise RuntimeError("No frames could be decoded")

        return np.stack(frames)

    @staticmethod
    def get_video_files(directory: str) -> list[str]:
        """Find all supported video files in a directory."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
        files = []
        for ext in VideoLoader.SUPPORTED_EXTENSIONS:
            files.extend(dir_path.glob(f"*{ext}"))
            files.extend(dir_path.glob(f"*{ext.upper()}"))
        return sorted(str(f) for f in set(files))
