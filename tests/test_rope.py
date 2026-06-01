"""Tests for rope.py — RoPE application and slicing."""

import torch
import pytest

from lvsa.rope import apply_rotary_emb, slice_rotary_emb


# ── apply_rotary_emb ──────────────────────────────────────────────────────────


class TestApplyRotaryEmb:
    """Tests for apply_rotary_emb."""

    def test_output_shape_matches_input(self):
        B, S, H, D = 2, 16, 4, 64
        hidden = torch.randn(B, S, H, D)
        cos = torch.randn(B, S, 1, D)
        sin = torch.randn(B, S, 1, D)
        out = apply_rotary_emb(hidden, cos, sin)
        assert out.shape == hidden.shape

    def test_dtype_preserved(self):
        """Output dtype should match input dtype."""
        for dtype in [torch.float32, torch.float64]:
            hidden = torch.randn(2, 8, 2, 32, dtype=dtype)
            cos = torch.randn(2, 8, 1, 32, dtype=dtype)
            sin = torch.randn(2, 8, 1, 32, dtype=dtype)
            out = apply_rotary_emb(hidden, cos, sin)
            assert out.dtype == dtype

    def test_zero_rotation_is_identity(self):
        """cos=1, sin=0 should be an identity transform."""
        B, S, H, D = 1, 4, 2, 16
        hidden = torch.randn(B, S, H, D)
        # cos entries at even indices, sin entries at odd indices
        cos = torch.ones(B, S, 1, D)
        sin = torch.zeros(B, S, 1, D)
        out = apply_rotary_emb(hidden, cos, sin)
        torch.testing.assert_close(out, hidden)

    def test_deterministic(self):
        """Same inputs produce same outputs."""
        B, S, H, D = 1, 8, 2, 32
        hidden = torch.randn(B, S, H, D)
        cos = torch.randn(B, S, 1, D)
        sin = torch.randn(B, S, 1, D)
        out1 = apply_rotary_emb(hidden, cos, sin)
        out2 = apply_rotary_emb(hidden, cos, sin)
        torch.testing.assert_close(out1, out2)

    def test_broadcast_heads(self):
        """freqs with head_dim=1 should broadcast across heads."""
        B, S, H, D = 1, 4, 8, 16
        hidden = torch.randn(B, S, H, D)
        cos = torch.randn(B, S, 1, D)  # head_dim=1, broadcasts to H
        sin = torch.randn(B, S, 1, D)
        out = apply_rotary_emb(hidden, cos, sin)
        assert out.shape == (B, S, H, D)

    def test_different_inputs_produce_different_outputs(self):
        """Sanity: different hidden states should give different results."""
        B, S, H, D = 1, 4, 2, 16
        cos = torch.randn(B, S, 1, D)
        sin = torch.randn(B, S, 1, D)
        h1 = torch.randn(B, S, H, D)
        h2 = torch.randn(B, S, H, D)
        out1 = apply_rotary_emb(h1, cos, sin)
        out2 = apply_rotary_emb(h2, cos, sin)
        assert not torch.allclose(out1, out2)


# ── slice_rotary_emb ──────────────────────────────────────────────────────────


class TestSliceRotaryEmb:
    """Tests for slice_rotary_emb."""

    def test_rank0_gets_first_chunk(self):
        full_seq = 40
        local_seq = 10
        world = 4
        cos = torch.randn(1, full_seq, 1, 16)
        sin = torch.randn(1, full_seq, 1, 16)
        sliced = slice_rotary_emb((cos, sin), local_seq, rank=0, world=world)
        torch.testing.assert_close(sliced[0], cos[:, :10])
        torch.testing.assert_close(sliced[1], sin[:, :10])

    def test_last_rank_gets_last_chunk(self):
        full_seq = 40
        local_seq = 10
        world = 4
        cos = torch.randn(1, full_seq, 1, 16)
        sin = torch.randn(1, full_seq, 1, 16)
        sliced = slice_rotary_emb((cos, sin), local_seq, rank=3, world=world)
        torch.testing.assert_close(sliced[0], cos[:, 30:40])
        torch.testing.assert_close(sliced[1], sin[:, 30:40])

    def test_all_ranks_cover_full_sequence(self):
        """Concatenating all rank slices should reconstruct the full tensor."""
        full_seq = 100
        local_seq = 25
        world = 4
        cos = torch.randn(1, full_seq, 1, 32)
        sin = torch.randn(1, full_seq, 1, 32)
        parts_cos = []
        parts_sin = []
        for r in range(world):
            s = slice_rotary_emb((cos, sin), local_seq, r, world)
            parts_cos.append(s[0])
            parts_sin.append(s[1])
        reconstructed_cos = torch.cat(parts_cos, dim=1)
        reconstructed_sin = torch.cat(parts_sin, dim=1)
        torch.testing.assert_close(reconstructed_cos, cos)
        torch.testing.assert_close(reconstructed_sin, sin)

    def test_single_rank_returns_full(self):
        """With world=1, output should equal input."""
        full_seq = 20
        cos = torch.randn(1, full_seq, 1, 16)
        sin = torch.randn(1, full_seq, 1, 16)
        sliced = slice_rotary_emb((cos, sin), full_seq, rank=0, world=1)
        torch.testing.assert_close(sliced[0], cos)
        torch.testing.assert_close(sliced[1], sin)

    def test_no_matching_dim_returns_unchanged(self):
        """If no dimension matches full_seq, tensor is returned as-is."""
        local_seq = 10
        world = 2
        # Tensor with no dim of size 20
        t = torch.randn(4, 8, 16)
        sliced = slice_rotary_emb((t,), local_seq, rank=0, world=world)
        assert sliced[0] is t

    def test_returns_same_container_type(self):
        """Return type should match the input container type (tuple)."""
        cos = torch.randn(1, 20, 1, 8)
        sin = torch.randn(1, 20, 1, 8)
        result = slice_rotary_emb((cos, sin), 10, rank=0, world=2)
        assert isinstance(result, tuple)

    def test_output_is_contiguous(self):
        cos = torch.randn(1, 40, 1, 16)
        sin = torch.randn(1, 40, 1, 16)
        sliced = slice_rotary_emb((cos, sin), 10, rank=1, world=4)
        assert sliced[0].is_contiguous()
        assert sliced[1].is_contiguous()
