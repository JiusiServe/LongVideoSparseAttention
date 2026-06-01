"""Tests for LVSA model adapters — registry, geometry methods, and QKV extraction."""

from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import torch
import pytest

from lvsa.adapters import get_adapter, register_adapter, _ADAPTER_REGISTRY
from lvsa.adapters.base import ModelAdapter
from lvsa.adapters.wan import WanAdapter
from lvsa.adapters.hunyuan_video import HunyuanVideoAdapter
from lvsa.adapters.cogvideox import CogVideoXAdapter


# ── Registry ─────────────────────────────────────────────────────────────────


class TestAdapterRegistry:
    """Tests for the adapter registry system."""

    def test_get_wan_adapter(self):
        adapter = get_adapter("wan")
        assert isinstance(adapter, WanAdapter)

    def test_get_hunyuan_adapter(self):
        adapter = get_adapter("hunyuan_video")
        assert isinstance(adapter, HunyuanVideoAdapter)

    def test_get_cogvideox_adapter(self):
        adapter = get_adapter("cogvideox")
        assert isinstance(adapter, CogVideoXAdapter)

    def test_unknown_adapter_raises(self):
        with pytest.raises(KeyError, match="Unknown adapter"):
            get_adapter("nonexistent_model")

    def test_register_custom_adapter(self):
        register_adapter("test_model", "lvsa.adapters.wan", "WanAdapter")
        assert "test_model" in _ADAPTER_REGISTRY
        adapter = get_adapter("test_model")
        assert isinstance(adapter, WanAdapter)
        # Clean up
        del _ADAPTER_REGISTRY["test_model"]

    def test_all_registered_adapters_instantiate(self):
        """Every registered adapter should instantiate without errors."""
        for name in list(_ADAPTER_REGISTRY.keys()):
            adapter = get_adapter(name)
            assert isinstance(adapter, ModelAdapter)


# ── Mock pipeline helpers ────────────────────────────────────────────────────


def _mock_wan_pipe(
    vae_scale_t=4, vae_scale_s=8, patch_size=(1, 2, 2), sample_frames=81,
):
    """Create a mock Wan pipeline with typical config."""
    config = SimpleNamespace(
        patch_size=patch_size,
        sample_frames=sample_frames,
    )
    transformer = SimpleNamespace(config=config)
    pipe = SimpleNamespace(
        transformer=transformer,
        vae_scale_factor_temporal=vae_scale_t,
        vae_scale_factor_spatial=vae_scale_s,
    )
    return pipe


def _mock_hunyuan_pipe(
    vae_scale_t=4, vae_scale_s=16, patch_size=2, sample_n_frames=129,
):
    """Create a mock HunyuanVideo pipeline with typical config."""
    config = SimpleNamespace(
        patch_size=patch_size,
        sample_n_frames=sample_n_frames,
    )
    transformer = SimpleNamespace(config=config)
    pipe = SimpleNamespace(
        transformer=transformer,
        vae_scale_factor_temporal=vae_scale_t,
        vae_scale_factor_spatial=vae_scale_s,
    )
    return pipe


def _mock_cogvideox_pipe(
    vae_scale_t=4, vae_scale_s=8, patch_size=2, patch_size_t=None,
    sample_frames=49,
):
    """Create a mock CogVideoX pipeline with typical config."""
    config = SimpleNamespace(
        patch_size=patch_size,
        patch_size_t=patch_size_t,
        sample_frames=sample_frames,
    )
    transformer = SimpleNamespace(config=config)
    pipe = SimpleNamespace(
        transformer=transformer,
        vae_scale_factor_temporal=vae_scale_t,
        vae_scale_factor_spatial=vae_scale_s,
    )
    return pipe


# ── Geometry: patches_per_frame ──────────────────────────────────────────────


class TestPatchesPerFrame:
    """Tests for patches_per_frame across all adapters."""

    def test_wan_480x832(self):
        adapter = WanAdapter()
        pipe = _mock_wan_pipe()
        # (480 / (8*2)) * (832 / (8*2)) = 30 * 52 = 1560
        assert adapter.patches_per_frame(480, 832, pipe) == 1560

    def test_wan_720x1280(self):
        adapter = WanAdapter()
        pipe = _mock_wan_pipe()
        # (720 / 16) * (1280 / 16) = 45 * 80 = 3600
        assert adapter.patches_per_frame(720, 1280, pipe) == 3600

    def test_hunyuan_480x832(self):
        adapter = HunyuanVideoAdapter()
        pipe = _mock_hunyuan_pipe()
        # (480 / (16*2)) * (832 / (16*2)) = 15 * 26 = 390
        assert adapter.patches_per_frame(480, 832, pipe) == 390

    def test_cogvideox_480x720(self):
        adapter = CogVideoXAdapter()
        pipe = _mock_cogvideox_pipe()
        # (480 / (8*2)) * (720 / (8*2)) = 30 * 45 = 1350
        assert adapter.patches_per_frame(480, 720, pipe) == 1350

    def test_cogvideox_480x720_list_patch(self):
        """CogVideoX with list-form patch_size should use last element."""
        adapter = CogVideoXAdapter()
        pipe = _mock_cogvideox_pipe(patch_size=[1, 2, 2])
        assert adapter.patches_per_frame(480, 720, pipe) == 1350


# ── Geometry: latent_frames ──────────────────────────────────────────────────


class TestLatentFrames:
    """Tests for latent frame count computation."""

    def test_wan_81_frames(self):
        adapter = WanAdapter()
        pipe = _mock_wan_pipe()
        assert adapter.latent_frames(81, pipe) == 21  # (81-1)//4 + 1

    def test_wan_61_frames(self):
        adapter = WanAdapter()
        pipe = _mock_wan_pipe()
        assert adapter.latent_frames(61, pipe) == 16  # (61-1)//4 + 1

    def test_wan_161_frames(self):
        adapter = WanAdapter()
        pipe = _mock_wan_pipe()
        assert adapter.latent_frames(161, pipe) == 41  # (161-1)//4 + 1

    def test_hunyuan_129_frames(self):
        adapter = HunyuanVideoAdapter()
        pipe = _mock_hunyuan_pipe()
        assert adapter.latent_frames(129, pipe) == 33  # (129-1)//4 + 1

    def test_cogvideox_49_frames(self):
        adapter = CogVideoXAdapter()
        pipe = _mock_cogvideox_pipe()
        assert adapter.latent_frames(49, pipe) == 13  # (49-1)//4 + 1

    def test_cogvideox_with_temporal_patch(self):
        """CogVideoX-1.5 with patch_size_t=2 halves latent frames (ceil)."""
        adapter = CogVideoXAdapter()
        pipe = _mock_cogvideox_pipe(patch_size_t=2)
        # (49-1)//4 + 1 = 13 latent, then ceil(13/2) = 7
        assert adapter.latent_frames(49, pipe) == 7

    def test_latent_frames_single_frame(self):
        """Edge case: single frame."""
        adapter = WanAdapter()
        pipe = _mock_wan_pipe()
        assert adapter.latent_frames(1, pipe) == 1


# ── Geometry: reference_latent_frames ────────────────────────────────────────


class TestReferenceLatentFrames:
    """Tests for reference_latent_frames across all adapters."""

    def test_wan_reference(self):
        adapter = WanAdapter()
        pipe = _mock_wan_pipe(sample_frames=81)
        assert adapter.reference_latent_frames(pipe) == 21

    def test_wan_reference_fallback(self):
        """When config has no sample_frames, should fallback to 21."""
        adapter = WanAdapter()
        config = SimpleNamespace()  # no sample_frames
        pipe = SimpleNamespace(
            transformer=SimpleNamespace(config=config),
            vae_scale_factor_temporal=4,
            vae_scale_factor_spatial=8,
        )
        assert adapter.reference_latent_frames(pipe) == 21

    def test_hunyuan_reference(self):
        adapter = HunyuanVideoAdapter()
        pipe = _mock_hunyuan_pipe(sample_n_frames=129)
        assert adapter.reference_latent_frames(pipe) == 33

    def test_hunyuan_reference_fallback(self):
        """When config has no sample_n_frames, should fallback to 33."""
        adapter = HunyuanVideoAdapter()
        config = SimpleNamespace()
        pipe = SimpleNamespace(
            transformer=SimpleNamespace(config=config),
            vae_scale_factor_temporal=4,
            vae_scale_factor_spatial=16,
        )
        assert adapter.reference_latent_frames(pipe) == 33

    def test_cogvideox_reference(self):
        adapter = CogVideoXAdapter()
        pipe = _mock_cogvideox_pipe(sample_frames=49)
        assert adapter.reference_latent_frames(pipe) == 13

    def test_cogvideox_reference_fallback(self):
        """When config has no sample_frames, should fallback to 13."""
        adapter = CogVideoXAdapter()
        config = SimpleNamespace(patch_size=2, patch_size_t=None)
        pipe = SimpleNamespace(
            transformer=SimpleNamespace(config=config),
            vae_scale_factor_temporal=4,
            vae_scale_factor_spatial=8,
        )
        assert adapter.reference_latent_frames(pipe) == 13

    def test_cogvideox_reference_with_temporal_patch(self):
        """CogVideoX-1.5 with patch_size_t should affect reference."""
        adapter = CogVideoXAdapter()
        pipe = _mock_cogvideox_pipe(sample_frames=49, patch_size_t=2)
        # (49-1)//4 + 1 = 13, ceil(13/2) = 7
        assert adapter.reference_latent_frames(pipe) == 7


# ── QKV extraction (CogVideoX shared projections) ───────────────────────────


def _mock_attn_module(inner_dim=1920, num_heads=30):
    """Create a mock attention module with to_q/k/v projections."""
    head_dim = inner_dim // num_heads
    attn = SimpleNamespace(
        heads=num_heads,
        to_q=torch.nn.Linear(inner_dim, inner_dim, bias=False),
        to_k=torch.nn.Linear(inner_dim, inner_dim, bias=False),
        to_v=torch.nn.Linear(inner_dim, inner_dim, bias=False),
        to_out=torch.nn.ModuleList([
            torch.nn.Linear(inner_dim, inner_dim, bias=False),
            torch.nn.Dropout(0.0),
        ]),
        norm_q=None,
        norm_k=None,
    )
    return attn


class TestCogVideoXSharedProjections:
    """Verify CogVideoX uses same to_q/k/v for video and text."""

    def test_extract_qkv_shape(self):
        adapter = CogVideoXAdapter()
        attn = _mock_attn_module(inner_dim=1920, num_heads=30)
        B, seq, C = 1, 100, 1920
        hidden = torch.randn(B, seq, C)
        encoder = torch.randn(B, 20, C)

        q, k, v = adapter.extract_qkv(attn, hidden, encoder)
        assert q.shape == (B, seq, 30, 64)
        assert k.shape == (B, seq, 30, 64)
        assert v.shape == (B, seq, 30, 64)

    def test_extract_encoder_qkv_shape(self):
        adapter = CogVideoXAdapter()
        attn = _mock_attn_module(inner_dim=1920, num_heads=30)
        B, text_seq, C = 1, 226, 1920
        encoder = torch.randn(B, text_seq, C)

        result = adapter.extract_encoder_qkv(attn, encoder)
        assert result is not None
        eq, ek, ev = result
        assert eq.shape == (B, text_seq, 30, 64)
        assert ek.shape == (B, text_seq, 30, 64)
        assert ev.shape == (B, text_seq, 30, 64)

    def test_shared_weights_equivalence(self):
        """Verify linear(cat(text, video)) == cat(linear(text), linear(video))."""
        adapter = CogVideoXAdapter()
        attn = _mock_attn_module(inner_dim=1920, num_heads=30)
        B, vid_seq, text_seq, C = 1, 100, 20, 1920
        video = torch.randn(B, vid_seq, C)
        text = torch.randn(B, text_seq, C)

        # Path 1: project separately (adapter way)
        vq, vk, vv = adapter.extract_qkv(attn, video, text)
        tq, tk, tv = adapter.extract_encoder_qkv(attn, text)

        # Path 2: project concatenated (original CogVideoX way)
        combined = torch.cat([text, video], dim=1)
        cq = attn.to_q(combined).view(B, text_seq + vid_seq, 30, 64)
        ck = attn.to_k(combined).view(B, text_seq + vid_seq, 30, 64)
        cv = attn.to_v(combined).view(B, text_seq + vid_seq, 30, 64)

        # Text portion should match
        torch.testing.assert_close(tq, cq[:, :text_seq])
        torch.testing.assert_close(tk, ck[:, :text_seq])
        torch.testing.assert_close(tv, cv[:, :text_seq])
        # Video portion should match
        torch.testing.assert_close(vq, cq[:, text_seq:])
        torch.testing.assert_close(vk, ck[:, text_seq:])
        torch.testing.assert_close(vv, cv[:, text_seq:])

    def test_encoder_qkv_none_when_no_encoder(self):
        adapter = CogVideoXAdapter()
        attn = _mock_attn_module()
        assert adapter.extract_encoder_qkv(attn, None) is None


# ── QKV extraction (HunyuanVideo dual-stream) ───────────────────────────────


class TestHunyuanVideoQKV:
    """Tests for HunyuanVideo QKV extraction."""

    def test_extract_qkv_shape(self):
        adapter = HunyuanVideoAdapter()
        inner_dim = 2048
        num_heads = 16
        attn = SimpleNamespace(
            heads=num_heads,
            to_q=torch.nn.Linear(inner_dim, inner_dim, bias=False),
            to_k=torch.nn.Linear(inner_dim, inner_dim, bias=False),
            to_v=torch.nn.Linear(inner_dim, inner_dim, bias=False),
            norm_q=None,
            norm_k=None,
        )
        B, seq, C = 1, 100, inner_dim
        hidden = torch.randn(B, seq, C)
        encoder = torch.randn(B, 20, C)

        q, k, v = adapter.extract_qkv(attn, hidden, encoder)
        assert q.shape == (B, seq, num_heads, 128)
        assert k.shape == (B, seq, num_heads, 128)
        assert v.shape == (B, seq, num_heads, 128)

    def test_extract_encoder_qkv_returns_none_without_add_proj(self):
        """Without add_q_proj, should return None (single-stream fallback)."""
        adapter = HunyuanVideoAdapter()
        attn = SimpleNamespace()  # no add_q_proj
        encoder = torch.randn(1, 20, 2048)
        assert adapter.extract_encoder_qkv(attn, encoder) is None


# ── Cross-attention dispatch ─────────────────────────────────────────────────


class TestCrossAttention:
    """Tests for cross_attention and split_encoder methods."""

    def test_cogvideox_no_cross_attention(self):
        adapter = CogVideoXAdapter()
        assert adapter.cross_attention(None, None, None, None) is None

    def test_hunyuan_no_cross_attention(self):
        adapter = HunyuanVideoAdapter()
        assert adapter.cross_attention(None, None, None, None) is None

    def test_cogvideox_split_encoder_no_split(self):
        adapter = CogVideoXAdapter()
        encoder = torch.randn(1, 20, 1920)
        text, image = adapter.split_encoder_for_cross_attn(None, encoder)
        assert text is encoder
        assert image is None

    def test_hunyuan_split_encoder_no_split(self):
        adapter = HunyuanVideoAdapter()
        encoder = torch.randn(1, 20, 2048)
        text, image = adapter.split_encoder_for_cross_attn(None, encoder)
        assert text is encoder
        assert image is None

    def test_cogvideox_no_cross_kv(self):
        adapter = CogVideoXAdapter()
        assert adapter.extract_cross_attn_kv(None, None) is None

    def test_hunyuan_no_cross_kv(self):
        adapter = HunyuanVideoAdapter()
        assert adapter.extract_cross_attn_kv(None, None) is None


# ── Output projection ───────────────────────────────────────────────────────


class TestOutputProjection:
    """Tests for output_projection and format_output."""

    def test_cogvideox_output_projection_shape(self):
        adapter = CogVideoXAdapter()
        inner_dim = 1920
        attn = _mock_attn_module(inner_dim=inner_dim, num_heads=30)
        B, seq, H, D = 1, 100, 30, 64
        hidden = torch.randn(B, seq, H, D)

        out = adapter.output_projection(attn, hidden, torch.float32)
        assert out.shape == (B, seq, inner_dim)

    def test_cogvideox_format_output_tuple(self):
        """With encoder output, should return (video, encoder) tuple."""
        adapter = CogVideoXAdapter()
        attn = _mock_attn_module(inner_dim=1920, num_heads=30)
        B, vid_seq, enc_seq, H, D = 1, 100, 20, 30, 64
        video = torch.randn(B, vid_seq, 1920)
        encoder_out = torch.randn(B, enc_seq, H, D)

        result = adapter.format_output(attn, video, encoder_out, enc_seq, torch.float32)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].shape == (B, vid_seq, 1920)
        assert result[1].shape == (B, enc_seq, 1920)

    def test_cogvideox_format_output_single_when_no_encoder(self):
        """Without encoder output, should return just video tensor."""
        adapter = CogVideoXAdapter()
        attn = _mock_attn_module()
        video = torch.randn(1, 100, 1920)
        result = adapter.format_output(attn, video, None, 0, torch.float32)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 100, 1920)


# ── Adapter abstract interface completeness ──────────────────────────────────


class TestAdapterInterface:
    """Verify all adapters implement the required abstract methods."""

    @pytest.mark.parametrize("adapter_cls", [WanAdapter, HunyuanVideoAdapter, CogVideoXAdapter])
    def test_has_required_methods(self, adapter_cls):
        adapter = adapter_cls()
        required = [
            "patches_per_frame", "latent_frames", "reference_latent_frames",
            "extract_qkv", "extract_cross_attn_kv", "split_encoder_for_cross_attn",
            "apply_rotary", "output_projection", "cross_attention",
            "install_processor", "setup_context_parallel", "patch_rotary_for_cp",
        ]
        for method in required:
            assert hasattr(adapter, method), f"{adapter_cls.__name__} missing {method}"
            assert callable(getattr(adapter, method))

    @pytest.mark.parametrize("adapter_cls", [WanAdapter, HunyuanVideoAdapter, CogVideoXAdapter])
    def test_has_optional_methods(self, adapter_cls):
        adapter = adapter_cls()
        optional = ["extract_encoder_qkv", "format_output"]
        for method in optional:
            assert hasattr(adapter, method)
            assert callable(getattr(adapter, method))
