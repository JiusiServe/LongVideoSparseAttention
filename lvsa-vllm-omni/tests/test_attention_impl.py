"""Tests for LVSAAttentionImpl (no vllm-omni dependency)."""
import torch
import pytest
from lvsa_vllm_omni.attention_impl import LVSAAttentionImpl
from lvsa_vllm_omni import step_tracker


@pytest.fixture(autouse=True)
def clean_tracker():
    """Reset step tracker before each test."""
    step_tracker.reset()
    yield
    step_tracker.reset()


@pytest.fixture
def impl(monkeypatch):
    """Create an impl with default config."""
    # Clear env to get defaults
    import os
    for key in list(os.environ):
        if key.startswith("LVSA_"):
            monkeypatch.delenv(key, raising=False)
    # Disable graduated schedule so LVSA activates on step 0
    monkeypatch.setenv("LVSA_SCHEDULE_START", "0")
    monkeypatch.setenv("LVSA_SCHEDULE_END", "0")
    return LVSAAttentionImpl(
        num_heads=2, head_size=8, softmax_scale=0.125,
    )


class TestCrossAttentionDetection:
    def test_different_seq_lengths_is_cross(self, impl):
        """Q and K with different seq lengths → dense fallback."""
        q = torch.randn(1, 20, 2, 8)
        k = torch.randn(1, 10, 2, 8)  # different length
        v = torch.randn(1, 10, 2, 8)
        out = impl.forward_cuda(q, k, v)
        assert out.shape == q.shape

    def test_same_seq_no_geometry_is_dense(self, impl):
        """Same seq lengths but no geometry info → dense fallback."""
        q = torch.randn(1, 20, 2, 8)
        k = torch.randn(1, 20, 2, 8)
        v = torch.randn(1, 20, 2, 8)
        # No total_latent_frames set → falls back to dense
        out = impl.forward_cuda(q, k, v)
        assert out.shape == q.shape


class TestLVSAAttention:
    def test_lvsa_with_tracker(self, impl):
        """Self-attention with geometry from step_tracker."""
        T, P, H, D = 5, 4, 2, 8
        step_tracker.set_total_latent_frames(T)
        step_tracker.set_step(0)

        seq = T * P
        q = torch.randn(1, seq, H, D)
        k = torch.randn(1, seq, H, D)
        v = torch.randn(1, seq, H, D)

        out = impl.forward_cuda(q, k, v)
        assert out.shape == (1, seq, H, D)
        assert not torch.isnan(out).any()

    def test_lvsa_with_env_config(self, monkeypatch):
        """Self-attention with geometry from env var."""
        T, P, H, D = 5, 4, 2, 8
        monkeypatch.setenv("LVSA_TOTAL_LATENT_FRAMES", str(T))
        monkeypatch.setenv("LVSA_WINDOW_SIZE", "4")
        monkeypatch.setenv("LVSA_N_FIRST_FRAMES", "4")
        monkeypatch.setenv("LVSA_KEY_FRAME_INTERVAL", "4")
        monkeypatch.setenv("LVSA_AUTO_KEYFRAMES", "0")

        impl = LVSAAttentionImpl(
            num_heads=H, head_size=D, softmax_scale=1.0 / (D**0.5),
        )

        seq = T * P
        q = torch.randn(1, seq, H, D)
        k = torch.randn(1, seq, H, D)
        v = torch.randn(1, seq, H, D)

        out = impl.forward_cuda(q, k, v)
        assert out.shape == (1, seq, H, D)

    def test_metadata_cached(self, impl):
        """LVSAMetadata should be reused across calls with same geometry."""
        T, P, H, D = 5, 4, 2, 8
        step_tracker.set_total_latent_frames(T)

        seq = T * P
        q = torch.randn(1, seq, H, D)
        k = torch.randn(1, seq, H, D)
        v = torch.randn(1, seq, H, D)

        impl.forward_cuda(q, k, v)
        meta1 = impl._lvsa_metadata

        impl.forward_cuda(q, k, v)
        meta2 = impl._lvsa_metadata

        assert meta1 is meta2  # same object, not rebuilt

    def test_metadata_rebuilt_on_step_change_with_rotation(self, monkeypatch):
        """With rotate_keyframes, metadata rebuilds on step change."""
        T, P, H, D = 10, 4, 2, 8
        monkeypatch.setenv("LVSA_ROTATE_KEYFRAMES", "1")
        monkeypatch.setenv("LVSA_AUTO_KEYFRAMES", "0")
        monkeypatch.setenv("LVSA_TOTAL_LATENT_FRAMES", str(T))
        monkeypatch.setenv("LVSA_PATCHES_PER_FRAME", str(P))
        # Disable graduated schedule so LVSA activates immediately
        monkeypatch.setenv("LVSA_SCHEDULE_START", "0")
        monkeypatch.setenv("LVSA_SCHEDULE_END", "0")
        # Single-forward-per-step semantics for this unit test (no CFG).
        monkeypatch.setenv("LVSA_CFG_PASSES", "1")

        impl = LVSAAttentionImpl(
            num_heads=H, head_size=D, softmax_scale=1.0 / (D**0.5),
        )

        seq = T * P
        q = torch.randn(1, seq, H, D)
        k = torch.randn(1, seq, H, D)
        v = torch.randn(1, seq, H, D)

        step_tracker.set_step(0)
        impl.forward_cuda(q, k, v)
        meta0 = impl._lvsa_metadata

        step_tracker.set_step(1)
        impl.forward_cuda(q, k, v)
        meta1 = impl._lvsa_metadata

        assert meta0 is not meta1  # rebuilt


class TestDenseAttention:
    def test_dense_output_shape(self, impl):
        q = torch.randn(1, 20, 2, 8)
        k = torch.randn(1, 30, 2, 8)
        v = torch.randn(1, 30, 2, 8)
        out = impl._dense_attention(q, k, v)
        assert out.shape == q.shape

    def test_dense_not_nan(self, impl):
        q = torch.randn(1, 20, 2, 8)
        k = torch.randn(1, 30, 2, 8)
        v = torch.randn(1, 30, 2, 8)
        out = impl._dense_attention(q, k, v)
        assert not torch.isnan(out).any()


class TestBackendImport:
    def test_backend_name(self):
        from lvsa_vllm_omni.backend import LVSABackend
        assert LVSABackend.get_name() == "LVSA"

    def test_backend_impl_cls(self):
        from lvsa_vllm_omni.backend import LVSABackend
        assert LVSABackend.get_impl_cls() is LVSAAttentionImpl

    def test_backend_head_sizes(self):
        from lvsa_vllm_omni.backend import LVSABackend
        sizes = LVSABackend.get_supported_head_sizes()
        assert 128 in sizes


class TestStepTracker:
    def test_set_get_step(self):
        step_tracker.set_step(5)
        assert step_tracker.get_step() == 5

    def test_set_get_total_frames(self):
        step_tracker.set_total_latent_frames(61)
        assert step_tracker.get_total_latent_frames() == 61

    def test_default_step(self):
        assert step_tracker.get_step() == 0

    def test_default_total_frames(self):
        assert step_tracker.get_total_latent_frames() is None

    def test_reset(self):
        step_tracker.set_step(10)
        step_tracker.set_total_latent_frames(100)
        step_tracker.reset()
        assert step_tracker.get_step() == 0
        assert step_tracker.get_total_latent_frames() is None
