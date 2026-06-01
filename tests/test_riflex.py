"""Unit tests for lvsa.riflex — RoPE frequency rescaling for length extrapolation."""

import math
from types import SimpleNamespace

import pytest
import torch

from lvsa.riflex import (
    apply_riflex_to_wan_pipe,
    identify_intrinsic_k,
    _build_riflex_temporal_table,
)


class TestIdentifyIntrinsicK:
    def test_wan_13b_default(self):
        # attention_head_dim=128 → t_dim=44, Wan 1.3B L=21 → k=3 (period≈22.06)
        assert identify_intrinsic_k(t_dim=44, training_length=21) == 3

    def test_wan_14b_default(self):
        # Wan 14B uses the same t_dim split; L is still 21 latent frames for 1x.
        assert identify_intrinsic_k(t_dim=44, training_length=21) == 3

    def test_picks_closest_period(self):
        # Shorter training → smaller k
        k_short = identify_intrinsic_k(t_dim=44, training_length=7)
        k_long = identify_intrinsic_k(t_dim=44, training_length=50)
        assert k_short < k_long

    def test_returns_valid_index(self):
        k = identify_intrinsic_k(t_dim=44, training_length=21)
        assert 0 <= k < 22

    def test_custom_theta(self):
        # Smaller theta → shorter periods → same L picks larger k
        k_default = identify_intrinsic_k(t_dim=44, training_length=21, theta=10000.0)
        k_small = identify_intrinsic_k(t_dim=44, training_length=21, theta=100.0)
        assert k_small >= k_default


class TestBuildRiflexTemporalTable:
    def test_shape(self):
        cos, sin = _build_riflex_temporal_table(
            t_dim=44, max_seq_len=128, theta=10000.0, k=3, s=2.0,
            training_length=21, dtype=torch.float32, device=torch.device("cpu"),
        )
        assert cos.shape == (128, 44)
        assert sin.shape == (128, 44)

    def test_only_target_columns_change(self):
        cos1, _ = _build_riflex_temporal_table(
            44, 64, 10000.0, k=3, s=1.0, training_length=21,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        cos2, _ = _build_riflex_temporal_table(
            44, 64, 10000.0, k=3, s=2.0, training_length=21,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        diff = (cos1 - cos2).abs().max(dim=0).values
        # repeat_interleave doubles each freq → cols 2k and 2k+1 differ
        changed = (diff > 1e-6).nonzero().flatten().tolist()
        assert changed == [6, 7]

    def test_period_matches_formula(self):
        # At pos = L*s, cos should equal cos(2π) = 1 for the modified freq
        L, s = 21, 2.0
        k = 3
        cos, _ = _build_riflex_temporal_table(
            44, L * int(s) + 1, 10000.0, k=k, s=s, training_length=L,
            dtype=torch.float64, device=torch.device("cpu"),
        )
        # pos = L*s = 42 → cos should equal cos(2π) = 1 at col 2k=6
        assert torch.isclose(cos[L * int(s), 2 * k], torch.tensor(1.0, dtype=torch.float64), atol=1e-10)


class TestApplyRiflexToWanPipe:
    def _fake_pipe(self, t_dim=44, max_seq_len=1024, total_head_dim=128):
        rope = SimpleNamespace(
            t_dim=t_dim,
            h_dim=(total_head_dim - t_dim) // 2,
            w_dim=(total_head_dim - t_dim) // 2,
            max_seq_len=max_seq_len,
            freqs_cos=torch.zeros(max_seq_len, total_head_dim, dtype=torch.float32),
            freqs_sin=torch.zeros(max_seq_len, total_head_dim, dtype=torch.float32),
        )
        # Pre-fill the temporal slice with the original (unmodified) RoPE values
        from diffusers.models.embeddings import get_1d_rotary_pos_embed
        cos_t, sin_t = get_1d_rotary_pos_embed(
            t_dim, max_seq_len, 10000.0, use_real=True, repeat_interleave_real=True,
        )
        rope.freqs_cos[:, :t_dim] = cos_t.float()
        rope.freqs_sin[:, :t_dim] = sin_t.float()
        transformer = SimpleNamespace(
            rope=rope,
            config=SimpleNamespace(sample_frames=81),
        )
        return SimpleNamespace(
            transformer=transformer,
            vae_scale_factor_temporal=4,
            vae_scale_factor_spatial=8,
        )

    def test_s1_is_noop(self):
        pipe = self._fake_pipe()
        cos_before = pipe.transformer.rope.freqs_cos.clone()
        info = apply_riflex_to_wan_pipe(pipe, s=1.0, training_length=21)
        assert info["applied"] is False
        torch.testing.assert_close(pipe.transformer.rope.freqs_cos, cos_before)

    def test_s2_modifies_only_temporal_slice(self):
        pipe = self._fake_pipe()
        t_dim = pipe.transformer.rope.t_dim
        cos_before = pipe.transformer.rope.freqs_cos.clone()
        info = apply_riflex_to_wan_pipe(pipe, s=2.0, training_length=21)
        assert info["applied"] is True
        assert info["k"] == 3
        # Spatial slice (h_dim + w_dim) must be untouched
        torch.testing.assert_close(
            pipe.transformer.rope.freqs_cos[:, t_dim:],
            cos_before[:, t_dim:],
        )
        # Temporal slice must differ
        assert not torch.allclose(
            pipe.transformer.rope.freqs_cos[:, :t_dim],
            cos_before[:, :t_dim],
        )

    def test_auto_detected_training_length(self):
        pipe = self._fake_pipe()  # sample_frames=81 → L=21
        info = apply_riflex_to_wan_pipe(pipe, s=2.0)
        assert info["training_length"] == 21

    def test_explicit_k_override(self):
        pipe = self._fake_pipe()
        info = apply_riflex_to_wan_pipe(pipe, s=2.0, k=5, training_length=21)
        assert info["k"] == 5

    def test_missing_rope_raises(self):
        pipe = SimpleNamespace(transformer=SimpleNamespace())
        with pytest.raises(RuntimeError, match="pipe.transformer.rope"):
            apply_riflex_to_wan_pipe(pipe, s=2.0, training_length=21)
