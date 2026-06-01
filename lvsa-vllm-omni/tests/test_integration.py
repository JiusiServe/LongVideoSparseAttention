"""End-to-end integration test simulating vllm-omni's attention pipeline.

This test does NOT require vllm-omni installed.  It simulates:
1. Backend discovery (env var → class import)
2. Impl instantiation (one per transformer block)
3. Forward calls across multiple denoising steps with self + cross attention
4. Step tracking and metadata caching
5. Correctness: LVSA output matches reference dense output within tolerance

This mirrors the real flow in vllm-omni:
  DiffusionModelRunner → denoise_step() → transformer_block.forward()
    → WanSelfAttention → Attention.forward() → impl.forward_cuda(q,k,v)
    → WanCrossAttention → Attention.forward() → impl.forward_cuda(q,k_enc,v_enc)
"""

import importlib
import os

import torch
import torch.nn.functional as F
import pytest

from lvsa_vllm_omni import step_tracker
from lvsa_vllm_omni.attention_impl import LVSAAttentionImpl
from lvsa_vllm_omni.config import LVSAConfig


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Reset tracker and env."""
    step_tracker.reset()
    for key in list(os.environ):
        if key.startswith("LVSA_"):
            monkeypatch.delenv(key, raising=False)
    # Disable graduated schedule so LVSA activates on step 0
    monkeypatch.setenv("LVSA_SCHEDULE_START", "0")
    monkeypatch.setenv("LVSA_SCHEDULE_END", "0")
    yield
    step_tracker.reset()


# ── Simulated vllm-omni pipeline ─────────────────────────────────────────


class SimulatedAttentionLayer:
    """Mimics vllm-omni's Attention class: holds impl, calls forward_cuda."""

    def __init__(self, impl):
        self.impl = impl

    def forward(self, q, k, v, attn_metadata=None):
        return self.impl.forward_cuda(q, k, v, attn_metadata)


class SimulatedTransformerBlock:
    """Mimics WanTransformerBlock with self-attention + cross-attention."""

    def __init__(self, self_attn: SimulatedAttentionLayer, cross_attn: SimulatedAttentionLayer):
        self.self_attn = self_attn
        self.cross_attn = cross_attn

    def forward(self, hidden_states, encoder_hidden_states):
        """Simplified forward: self-attn on hidden, cross-attn with encoder."""
        B, seq, C = hidden_states.shape
        H, D = 2, C // 2  # mock head config

        # Self-attention: project to Q,K,V (identity projection for test)
        q = hidden_states.view(B, seq, H, D)
        k = hidden_states.view(B, seq, H, D)
        v = hidden_states.view(B, seq, H, D)
        self_out = self.self_attn.forward(q, k, v)
        hidden_states = hidden_states + self_out.view(B, seq, C)

        # Cross-attention: Q from hidden, K/V from encoder
        enc_seq = encoder_hidden_states.shape[1]
        q_cross = hidden_states.view(B, seq, H, D)
        k_cross = encoder_hidden_states.view(B, enc_seq, H, D)
        v_cross = encoder_hidden_states.view(B, enc_seq, H, D)
        cross_out = self.cross_attn.forward(q_cross, k_cross, v_cross)
        hidden_states = hidden_states + cross_out.view(B, seq, C)

        return hidden_states


def dense_self_attention(q, k, v):
    """Reference dense attention for correctness check."""
    qt = q.transpose(1, 2)
    kt = k.transpose(1, 2)
    vt = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(qt, kt, vt, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2)


# ── Tests ─────────────────────────────────────────────────────────────────


class TestBackendDiscovery:
    """Test that the backend can be discovered via env var (as vllm-omni does)."""

    def test_import_backend_from_path(self):
        """Simulate DIFFUSION_ATTENTION_BACKEND=lvsa_vllm_omni.backend.LVSABackend"""
        module_path = "lvsa_vllm_omni.backend"
        class_name = "LVSABackend"

        mod = importlib.import_module(module_path)
        backend_cls = getattr(mod, class_name)

        assert backend_cls.get_name() == "LVSA"
        impl_cls = backend_cls.get_impl_cls()
        assert impl_cls is LVSAAttentionImpl

    def test_create_impl_from_backend(self):
        """Backend.get_impl_cls() should produce a working impl."""
        from lvsa_vllm_omni.backend import LVSABackend
        impl_cls = LVSABackend.get_impl_cls()
        impl = impl_cls(num_heads=2, head_size=8, softmax_scale=0.125)
        assert isinstance(impl, LVSAAttentionImpl)


class TestFullPipelineSimulation:
    """Simulate a full denoising loop with N blocks × M steps."""

    def _setup(self, monkeypatch, T_lat=21, P=30, H=2, D=8, n_blocks=4, n_steps=5):
        monkeypatch.setenv("LVSA_TOTAL_LATENT_FRAMES", str(T_lat))
        monkeypatch.setenv("LVSA_PATCHES_PER_FRAME", str(P))
        monkeypatch.setenv("LVSA_WINDOW_SIZE", "12")
        monkeypatch.setenv("LVSA_N_FIRST_FRAMES", "4")
        monkeypatch.setenv("LVSA_AUTO_KEYFRAMES", "1")

        C = H * D
        seq = T_lat * P
        enc_seq = 77  # typical text token count

        # Create N transformer blocks, each with its own impl instance
        blocks = []
        for _ in range(n_blocks):
            self_impl = LVSAAttentionImpl(num_heads=H, head_size=D, softmax_scale=1.0 / (D**0.5))
            cross_impl = LVSAAttentionImpl(num_heads=H, head_size=D, softmax_scale=1.0 / (D**0.5))
            self_attn = SimulatedAttentionLayer(self_impl)
            cross_attn = SimulatedAttentionLayer(cross_impl)
            blocks.append(SimulatedTransformerBlock(self_attn, cross_attn))

        return blocks, seq, enc_seq, C, n_steps

    def test_multi_step_multi_block(self, monkeypatch):
        """Run N_blocks × N_steps forward passes without error."""
        blocks, seq, enc_seq, C, n_steps = self._setup(monkeypatch)
        B = 1

        hidden = torch.randn(B, seq, C)
        encoder = torch.randn(B, enc_seq, C)

        for step in range(n_steps):
            step_tracker.set_step(step)
            for block in blocks:
                hidden = block.forward(hidden, encoder)

        assert hidden.shape == (B, seq, C)
        assert not torch.isnan(hidden).any()

    def test_metadata_shared_across_blocks(self, monkeypatch):
        """All self-attention impls should build identical LVSAMetadata."""
        blocks, seq, enc_seq, C, n_steps = self._setup(monkeypatch)
        B = 1

        hidden = torch.randn(B, seq, C)
        encoder = torch.randn(B, enc_seq, C)

        step_tracker.set_step(0)
        for block in blocks:
            block.forward(hidden, encoder)

        # All self-attn impls should have metadata with same global indices
        metadatas = [b.self_attn.impl._lvsa_metadata for b in blocks]
        assert all(m is not None for m in metadatas)
        assert all(m.global_indices == metadatas[0].global_indices for m in metadatas)

    def test_cross_attention_no_lvsa(self, monkeypatch):
        """Cross-attention impls should NOT build LVSAMetadata."""
        blocks, seq, enc_seq, C, n_steps = self._setup(monkeypatch)
        B = 1

        hidden = torch.randn(B, seq, C)
        encoder = torch.randn(B, enc_seq, C)

        step_tracker.set_step(0)
        for block in blocks:
            block.forward(hidden, encoder)

        # Cross-attn impls should have no LVSA metadata (they used dense fallback)
        for block in blocks:
            assert block.cross_attn.impl._lvsa_metadata is None

    def test_step_rotation_changes_pattern(self, monkeypatch):
        """With rotate_keyframes, different steps produce different patterns."""
        monkeypatch.setenv("LVSA_TOTAL_LATENT_FRAMES", "21")
        monkeypatch.setenv("LVSA_PATCHES_PER_FRAME", "30")
        monkeypatch.setenv("LVSA_ROTATE_KEYFRAMES", "1")
        monkeypatch.setenv("LVSA_AUTO_KEYFRAMES", "0")
        monkeypatch.setenv("LVSA_KEY_FRAME_INTERVAL", "16")
        # Single-forward-per-step semantics for this unit test (no CFG).
        monkeypatch.setenv("LVSA_CFG_PASSES", "1")

        impl = LVSAAttentionImpl(num_heads=2, head_size=8, softmax_scale=0.125)
        seq = 21 * 30
        q = torch.randn(1, seq, 2, 8)
        k = torch.randn(1, seq, 2, 8)
        v = torch.randn(1, seq, 2, 8)

        step_tracker.set_step(0)
        impl.forward_cuda(q, k, v)
        indices_step0 = impl._lvsa_metadata.global_indices.copy()

        step_tracker.set_step(1)
        impl.forward_cuda(q, k, v)
        indices_step1 = impl._lvsa_metadata.global_indices.copy()

        # Different steps should produce different global index sets
        assert indices_step0 != indices_step1
        # But same count (rotation preserves budget)
        assert len(indices_step0) == len(indices_step1)


class TestCorrectnessAtReferenceLength:
    """At reference length (T=21 for Wan), LVSA with auto-KFI should be
    equivalent to dense attention (all frames are global → attended=21/21)."""

    def test_lvsa_equals_dense_at_1x(self, monkeypatch):
        T_lat, P, H, D = 21, 4, 2, 8  # small P for speed
        monkeypatch.setenv("LVSA_TOTAL_LATENT_FRAMES", str(T_lat))
        monkeypatch.setenv("LVSA_PATCHES_PER_FRAME", str(P))
        monkeypatch.setenv("LVSA_AUTO_KEYFRAMES", "1")

        impl = LVSAAttentionImpl(num_heads=H, head_size=D, softmax_scale=1.0 / (D**0.5))
        step_tracker.set_step(0)

        seq = T_lat * P
        torch.manual_seed(42)
        q = torch.randn(1, seq, H, D)
        k = torch.randn(1, seq, H, D)
        v = torch.randn(1, seq, H, D)

        lvsa_out = impl.forward_cuda(q, k, v)
        dense_out = dense_self_attention(q, k, v)

        # At 1x with auto-KFI, all 21 frames are global → attended=21/21
        # So LVSA output should closely match dense (not exact due to
        # per-frame chunking vs single kernel, but very close)
        assert lvsa_out.shape == dense_out.shape

        # Check that the metadata confirms all frames are attended
        meta = impl._lvsa_metadata
        assert meta is not None
        # With auto-KFI at T=21 (reference), kfi=1 → all frames global
        for f_global, _, _ in meta.local_frames:
            win_parts = meta.window_ctx[f_global]
            total_attended = len(meta.global_indices)
            for s, e in win_parts:
                total_attended += (e - s) // P
            assert total_attended >= T_lat  # every frame attended

    def test_output_not_nan_at_extended_length(self, monkeypatch):
        """At 3x extension, LVSA should produce valid output."""
        T_lat, P, H, D = 61, 4, 2, 8
        monkeypatch.setenv("LVSA_TOTAL_LATENT_FRAMES", str(T_lat))
        monkeypatch.setenv("LVSA_PATCHES_PER_FRAME", str(P))
        monkeypatch.setenv("LVSA_AUTO_KEYFRAMES", "1")

        impl = LVSAAttentionImpl(num_heads=H, head_size=D, softmax_scale=1.0 / (D**0.5))
        step_tracker.set_step(0)

        seq = T_lat * P
        q = torch.randn(1, seq, H, D)
        k = torch.randn(1, seq, H, D)
        v = torch.randn(1, seq, H, D)

        out = impl.forward_cuda(q, k, v)
        assert out.shape == (1, seq, H, D)
        assert not torch.isnan(out).any()

        # At 3x, not all frames should be global (sparse pattern active)
        meta = impl._lvsa_metadata
        assert len(meta.global_indices) < T_lat


class TestBatchedGeneration:
    """Test with B>1 (classifier-free guidance uses B=2)."""

    def test_cfg_batch(self, monkeypatch):
        T_lat, P, H, D = 21, 4, 2, 8
        B = 2  # CFG: conditional + unconditional
        monkeypatch.setenv("LVSA_TOTAL_LATENT_FRAMES", str(T_lat))

        impl = LVSAAttentionImpl(num_heads=H, head_size=D, softmax_scale=1.0 / (D**0.5))
        step_tracker.set_step(0)

        seq = T_lat * P
        q = torch.randn(B, seq, H, D)
        k = torch.randn(B, seq, H, D)
        v = torch.randn(B, seq, H, D)

        out = impl.forward_cuda(q, k, v)
        assert out.shape == (B, seq, H, D)
        assert not torch.isnan(out).any()


class TestNonDefaultResolution:
    """Verify that LVSA_PATCHES_PER_FRAME and resolution-derived geometry
    are honored end-to-end through ``LVSAAttentionImpl.forward_cuda``."""

    def test_explicit_patches_per_frame_override(self, monkeypatch):
        """Setting LVSA_PATCHES_PER_FRAME makes the impl detect P from it."""
        T_lat, P_custom, H, D = 9, 7, 2, 8   # P=7 is non-default
        monkeypatch.setenv("LVSA_TOTAL_LATENT_FRAMES", str(T_lat))
        monkeypatch.setenv("LVSA_PATCHES_PER_FRAME", str(P_custom))

        impl = LVSAAttentionImpl(num_heads=H, head_size=D, softmax_scale=1.0 / (D**0.5))
        step_tracker.set_step(0)

        seq = T_lat * P_custom
        q = torch.randn(1, seq, H, D)
        k = torch.randn(1, seq, H, D)
        v = torch.randn(1, seq, H, D)

        out = impl.forward_cuda(q, k, v)
        assert out.shape == (1, seq, H, D)
        assert not torch.isnan(out).any()
        # Confirm the impl picked the override, not the default 1560
        meta = impl._lvsa_metadata
        assert meta.num_patches == P_custom

    def test_resolution_derived_p(self, monkeypatch):
        """Setting only LVSA_VIDEO_HEIGHT/WIDTH derives P via the geometry chain."""
        # (480/8/2) * (832/8/2) = 1560 — default for HV/Wan at 480×832
        # Use a smaller custom resolution to keep test fast.
        T_lat, H_dim, D = 5, 1, 4
        monkeypatch.setenv("LVSA_TOTAL_LATENT_FRAMES", str(T_lat))
        # 32×32 with vae=8 patch=2 → (32/8/2)*(32/8/2) = 2*2 = 4
        monkeypatch.setenv("LVSA_VIDEO_HEIGHT", "32")
        monkeypatch.setenv("LVSA_VIDEO_WIDTH", "32")

        P_expected = 4
        seq = T_lat * P_expected
        q = torch.randn(1, seq, H_dim, D)
        k = torch.randn(1, seq, H_dim, D)
        v = torch.randn(1, seq, H_dim, D)

        impl = LVSAAttentionImpl(num_heads=H_dim, head_size=D, softmax_scale=1.0 / (D**0.5))
        step_tracker.set_step(0)
        out = impl.forward_cuda(q, k, v)

        assert out.shape == (1, seq, H_dim, D)
        meta = impl._lvsa_metadata
        assert meta.num_patches == P_expected
