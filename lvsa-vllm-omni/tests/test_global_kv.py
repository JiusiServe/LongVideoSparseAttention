"""Tests for global K/V extraction."""
import torch
import pytest
from lvsa_vllm_omni.global_kv import build_global_kv


class TestBuildGlobalKV:
    def test_output_shape(self):
        B, T, P, H, D = 1, 10, 4, 2, 8
        seq = T * P
        k = torch.randn(B, seq, H, D)
        v = torch.randn(B, seq, H, D)
        global_indices = [0, 3, 6, 9]
        k_g, v_g = build_global_kv(k, v, global_indices, P)
        assert k_g.shape == (B, len(global_indices) * P, H, D)
        assert v_g.shape == (B, len(global_indices) * P, H, D)

    def test_correct_values(self):
        B, T, P, H, D = 1, 5, 2, 1, 4
        seq = T * P
        k = torch.arange(seq).float().view(1, seq, 1, 1).expand(B, seq, H, D)
        v = k.clone()
        global_indices = [1, 3]  # frames 1 and 3
        k_g, _ = build_global_kv(k, v, global_indices, P)
        # Frame 1 tokens: indices 2, 3. Frame 3 tokens: indices 6, 7.
        expected_token_ids = [2, 3, 6, 7]
        for i, tid in enumerate(expected_token_ids):
            assert k_g[0, i, 0, 0].item() == tid

    def test_empty_globals(self):
        B, H, D = 1, 2, 8
        k = torch.randn(B, 40, H, D)
        v = torch.randn(B, 40, H, D)
        k_g, v_g = build_global_kv(k, v, [], 4)
        assert k_g.shape == (B, 0, H, D)
        assert v_g.shape == (B, 0, H, D)

    def test_all_frames_global(self):
        B, T, P, H, D = 1, 5, 3, 2, 8
        seq = T * P
        k = torch.randn(B, seq, H, D)
        v = torch.randn(B, seq, H, D)
        global_indices = list(range(T))
        k_g, v_g = build_global_kv(k, v, global_indices, P)
        assert k_g.shape == (B, seq, H, D)
        # When all frames are global, output should equal input
        assert torch.equal(k_g, k)

    def test_batched(self):
        B, T, P, H, D = 3, 8, 4, 2, 8
        seq = T * P
        k = torch.randn(B, seq, H, D)
        v = torch.randn(B, seq, H, D)
        global_indices = [0, 4]
        k_g, v_g = build_global_kv(k, v, global_indices, P)
        assert k_g.shape == (B, 2 * P, H, D)
        # Check each batch element independently
        for b in range(B):
            for gi, gf in enumerate(global_indices):
                expected = k[b, gf * P:(gf + 1) * P]
                actual = k_g[b, gi * P:(gi + 1) * P]
                assert torch.equal(actual, expected)
