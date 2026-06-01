"""Tests for video loader module."""

import os
import tempfile

import cv2
import numpy as np
import pytest

from vqeval.core.video_loader import VideoLoader, VideoData, VideoMeta


def _create_test_video(path, n_frames=30, fps=30.0, width=320, height=240):
    """Create a synthetic test video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(n_frames):
        # Create a frame with gradual color change
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = int(255 * i / n_frames)  # Blue ramp
        frame[:, :, 1] = 128
        frame[:, :, 2] = int(255 * (1 - i / n_frames))  # Red ramp
        # Add some texture
        cv2.putText(frame, f"Frame {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(frame)
    writer.release()


class TestVideoLoader:
    @pytest.fixture
    def loader(self):
        return VideoLoader()

    @pytest.fixture
    def short_video(self, tmp_path):
        """Create a 1-second video (30 frames at 30fps)."""
        path = str(tmp_path / "short.mp4")
        _create_test_video(path, n_frames=30, fps=30.0)
        return path

    @pytest.fixture
    def long_video(self, tmp_path):
        """Create a 10-second video (300 frames at 30fps)."""
        path = str(tmp_path / "long.mp4")
        _create_test_video(path, n_frames=300, fps=30.0)
        return path

    def test_load_short_video(self, loader, short_video):
        video = loader.load(short_video)
        assert isinstance(video, VideoData)
        assert video.meta.width == 320
        assert video.meta.height == 240
        assert video.meta.fps == pytest.approx(30.0, abs=1.0)
        # Short video (1s < 5s threshold) - all frames sampled
        assert video.num_frames == 30

    def test_load_long_video_samples(self, loader, long_video):
        video = loader.load(long_video)
        # Long video (10s > 5s threshold) - should subsample
        assert video.num_frames < 300
        assert video.num_frames >= 20  # At least 2fps * 10s = 20

    def test_rgb_property(self, loader, short_video):
        video = loader.load(short_video)
        rgb = video.rgb
        assert rgb.shape == video.frames.shape
        assert rgb.dtype == np.uint8

    def test_get_tensors(self, loader, short_video):
        video = loader.load(short_video)
        tensors = video.get_tensors("cpu")
        assert tensors.shape[0] == video.num_frames
        assert tensors.shape[1] == 3  # Channels
        assert tensors.min() >= 0.0
        assert tensors.max() <= 1.0

    def test_cache(self, loader, short_video):
        video = loader.load(short_video)
        video.cache_set("test_key", 42)
        assert video.cache_get("test_key") == 42
        assert video.cache_get("missing") is None
        assert video.cache_get("missing", "default") == "default"

    def test_file_not_found(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent.mp4")

    def test_unsupported_format(self, loader, tmp_path):
        path = str(tmp_path / "test.txt")
        with open(path, "w") as f:
            f.write("not a video")
        with pytest.raises(ValueError, match="Unsupported format"):
            loader.load(path)

    def test_long_video_accepted(self, loader, tmp_path):
        path = str(tmp_path / "long.mp4")
        # 31 seconds at 30fps = 930 frames — should still be accepted
        _create_test_video(path, n_frames=930, fps=30.0)
        video = loader.load(path)
        assert video.meta.duration > 30.0

    def test_frame_indices_include_boundaries(self, loader, long_video):
        video = loader.load(long_video)
        indices = video.frame_indices
        # Should include first and last frame
        assert 0 in indices
        assert video.meta.total_frames - 1 in indices


class TestVideoLoaderStatic:
    def test_get_video_files(self, tmp_path):
        # Create some test files
        (tmp_path / "video1.mp4").touch()
        (tmp_path / "video2.mov").touch()
        (tmp_path / "video3.webm").touch()
        (tmp_path / "not_video.txt").touch()

        files = VideoLoader.get_video_files(str(tmp_path))
        assert len(files) == 3
        assert all(f.endswith((".mp4", ".mov", ".webm")) for f in files)

    def test_get_video_files_empty_dir(self, tmp_path):
        files = VideoLoader.get_video_files(str(tmp_path))
        assert files == []

    def test_get_video_files_not_dir(self, tmp_path):
        path = str(tmp_path / "not_a_dir.txt")
        with open(path, "w") as f:
            f.write("test")
        with pytest.raises(NotADirectoryError):
            VideoLoader.get_video_files(path)
