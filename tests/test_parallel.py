"""Tests for parallel.py — compute_and_validate_seq_len."""

import types

import numpy as np
import pytest
import torch

from lvsa.parallel import compute_and_validate_seq_len


def _make_config(patch_size=(1, 2, 2)):
    """Create a minimal mock transformer config."""
    cfg = types.SimpleNamespace(patch_size=patch_size)
    return cfg


class TestComputeAndValidateSeqLen:
    """Tests for compute_and_validate_seq_len."""

    def test_basic_computation(self):
        """Standard Wan 2.1 parameters: 81 frames, 480x832."""
        # vae_scale_t=4, vae_scale_s=8, patch_size=[1,2,2]
        # T_lat = (81-1)//4 + 1 = 21
        # T_patch = ceil(21/1) = 21
        # H_patch = 480//(8*2) = 30
        # W_patch = 832//(8*2) = 52
        # seq_len = 21*30*52 = 32760
        seq_len, T_lat, num_patches = compute_and_validate_seq_len(
            num_frames=81, height=480, width=832,
            transformer_config=_make_config(),
            vae_scale_t=4, vae_scale_s=8,
            world=1, rank=0,
        )
        assert T_lat == 21
        assert num_patches == 30 * 52  # 1560
        assert seq_len == 21 * 1560

    def test_161_frames_2_gpus(self):
        """161 frames on 2 GPUs."""
        # T_lat = (161-1)//4 + 1 = 41
        # seq_len = 41 * 1560 = 63960
        # 63960 / 2 = 31980 ✓
        seq_len, T_lat, num_patches = compute_and_validate_seq_len(
            num_frames=161, height=480, width=832,
            transformer_config=_make_config(),
            vae_scale_t=4, vae_scale_s=8,
            world=2, rank=0,
        )
        assert T_lat == 41
        assert seq_len == 41 * 1560
        assert seq_len % 2 == 0

    def test_divisibility_assertion(self):
        """Should fail when seq_len is not divisible by world."""
        # 81 frames: seq_len = 32760
        # 32760 / 4 = 8190 ✓ ... let's find a failing case
        # 21 frames: T_lat = (21-1)//4 + 1 = 6
        # seq_len = 6 * 1560 = 9360
        # 9360 / 4 = 2340 ✓
        # We need seq_len % world != 0
        # Try 480x480: H_patch=30, W_patch=30, num_patches=900
        # 81 frames: seq_len = 21*900 = 18900
        # 18900 % 4 = 18900/4 = 4725 ✓
        # Try world=7: 18900 % 7 = 18900 - 2700*7 = 18900 - 18900 = 0 ... hmm
        # Construct: T=5 frames → T_lat=(5-1)//4+1=2, seq=2*900=1800, 1800%7=1800-257*7=1800-1799=1
        with pytest.raises(AssertionError, match="divisibility"):
            compute_and_validate_seq_len(
                num_frames=5, height=480, width=480,
                transformer_config=_make_config(),
                vae_scale_t=4, vae_scale_s=8,
                world=7, rank=0,
            )

    def test_single_gpu_always_valid(self):
        """world=1 should never fail divisibility check."""
        seq_len, T_lat, num_patches = compute_and_validate_seq_len(
            num_frames=81, height=480, width=832,
            transformer_config=_make_config(),
            vae_scale_t=4, vae_scale_s=8,
            world=1, rank=0,
        )
        assert seq_len > 0

    def test_scalar_patch_size(self):
        """Scalar patch_size should use patch_size_t=1, patch_size_s=scalar."""
        cfg = _make_config(patch_size=2)
        seq_len, T_lat, num_patches = compute_and_validate_seq_len(
            num_frames=81, height=480, width=832,
            transformer_config=cfg,
            vae_scale_t=4, vae_scale_s=8,
            world=1, rank=0,
        )
        # Same as [1, 2, 2] for spatial
        assert T_lat == 21
        assert num_patches == 30 * 52

    def test_T_lat_formula(self):
        """T_lat = (num_frames - 1) // vae_scale_t + 1."""
        for nf in [5, 17, 81, 121, 161, 241]:
            _, T_lat, _ = compute_and_validate_seq_len(
                num_frames=nf, height=480, width=832,
                transformer_config=_make_config(),
                vae_scale_t=4, vae_scale_s=8,
                world=1, rank=0,
            )
            assert T_lat == (nf - 1) // 4 + 1

    def test_num_patches_independent_of_frames(self):
        """num_patches is purely spatial, should not change with num_frames."""
        _, _, p1 = compute_and_validate_seq_len(
            num_frames=81, height=480, width=832,
            transformer_config=_make_config(),
            vae_scale_t=4, vae_scale_s=8, world=1, rank=0,
        )
        _, _, p2 = compute_and_validate_seq_len(
            num_frames=161, height=480, width=832,
            transformer_config=_make_config(),
            vae_scale_t=4, vae_scale_s=8, world=1, rank=0,
        )
        assert p1 == p2


