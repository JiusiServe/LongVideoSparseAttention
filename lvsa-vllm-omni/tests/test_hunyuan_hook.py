"""Tests for HunyuanVideo LVSA hook.

The actual ``install_hunyuan_lvsa_hook`` requires vllm-omni installed, so we
focus on:
  - The module-level helpers (``_mask_log_should_fire``, ``build_global_kv``)
    which are pure and CPU-testable.
  - The ``HunyuanLVSAState`` class (step tracking, metadata caching).
  - Graceful gating via ``maybe_install_hunyuan_hook`` (register.py path).

For the install entry point itself we just verify the module imports cleanly
and the function exists — actually exercising the monkey-patch requires
vllm-omni's HunyuanVideo15Attention class.
"""
from __future__ import annotations

import os
import pytest
import torch

from lvsa_vllm_omni.hunyuan_hook import (
    _mask_log_should_fire,
    HunyuanLVSAState,
)
from lvsa_vllm_omni.global_kv import build_global_kv
from lvsa_vllm_omni.config import LVSAConfig


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("LVSA_"):
            monkeypatch.delenv(key, raising=False)
    yield


# ── _mask_log_should_fire (pure function) ────────────────────────────────────


class TestMaskLogShouldFire:
    def test_empty_spec_never_fires(self):
        assert not _mask_log_should_fire("", 0, -1)
        assert not _mask_log_should_fire("", 5, -1)

    def test_zero_spec_never_fires(self):
        assert not _mask_log_should_fire("0", 0, -1)
        assert not _mask_log_should_fire("0", 10, 3)

    def test_one_fires_every_new_step(self):
        assert _mask_log_should_fire("1", 0, -1)
        assert _mask_log_should_fire("1", 1, 0)

    def test_one_dedups_same_step(self):
        # If last_step == step_idx, should not fire again
        assert not _mask_log_should_fire("1", 5, 5)

    def test_once_fires_only_first_time(self):
        assert _mask_log_should_fire("once", 0, -1)
        # After last_step is set (no longer -1), shouldn't fire
        assert not _mask_log_should_fire("once", 1, 0)

    def test_range_spec(self):
        assert _mask_log_should_fire("3-7", 5, -1)
        assert _mask_log_should_fire("3-7", 3, -1)
        assert _mask_log_should_fire("3-7", 7, -1)
        assert not _mask_log_should_fire("3-7", 2, -1)
        assert not _mask_log_should_fire("3-7", 8, -1)

    def test_list_spec(self):
        assert _mask_log_should_fire("2,5,10", 5, -1)
        assert _mask_log_should_fire("2,5,10", 2, -1)
        assert not _mask_log_should_fire("2,5,10", 3, -1)

    def test_invalid_spec_does_not_crash(self):
        # Non-parseable specs should return False, not raise
        assert _mask_log_should_fire("abc", 0, -1) is False
        assert _mask_log_should_fire("3-abc", 5, -1) is False


# ── build_global_kv (pure tensor function) ──────────────────────────────────


class TestBuildGlobalKV:
    def test_basic_shape(self):
        B, T, P, H, D = 1, 6, 4, 2, 8
        seq = T * P
        k = torch.randn(B, seq, H, D)
        v = torch.randn(B, seq, H, D)
        global_indices = [0, 2, 5]
        k_g, v_g = build_global_kv(k, v, global_indices, P)
        assert k_g.shape == (B, len(global_indices) * P, H, D)
        assert v_g.shape == (B, len(global_indices) * P, H, D)

    def test_values_match_source(self):
        B, T, P, H, D = 1, 4, 2, 1, 4
        seq = T * P
        torch.manual_seed(0)
        k = torch.randn(B, seq, H, D)
        v = torch.randn(B, seq, H, D)
        global_indices = [1, 3]   # tokens 2,3 and 6,7
        k_g, v_g = build_global_kv(k, v, global_indices, P)
        expected_token_ids = [2, 3, 6, 7]
        for i, tid in enumerate(expected_token_ids):
            assert torch.equal(k_g[0, i], k[0, tid])
            assert torch.equal(v_g[0, i], v[0, tid])

    def test_empty_globals(self):
        B, T, P, H, D = 1, 4, 2, 1, 4
        k = torch.randn(B, T * P, H, D)
        v = torch.randn(B, T * P, H, D)
        k_g, v_g = build_global_kv(k, v, [], P)
        assert k_g.shape == (B, 0, H, D)
        assert v_g.shape == (B, 0, H, D)

    def test_batched(self):
        B, T, P, H, D = 3, 4, 2, 2, 8
        k = torch.randn(B, T * P, H, D)
        v = torch.randn(B, T * P, H, D)
        k_g, v_g = build_global_kv(k, v, [0, 2], P)
        assert k_g.shape == (B, 2 * P, H, D)
        # Per-batch slicing preserved
        assert torch.equal(k_g[1], k[1, [0, 1, 4, 5]])


# ── HunyuanLVSAState (step tracking + metadata caching) ──────────────────────


class TestHunyuanLVSAStateTick:
    def test_first_tick_returns_zero(self):
        s = HunyuanLVSAState(LVSAConfig())
        assert s.tick(layer_id=1, seq_len=100) == 0

    def test_seq_len_change_resets_step(self):
        s = HunyuanLVSAState(LVSAConfig())
        # Simulate some calls at seq_len=100
        s.tick(layer_id=1, seq_len=100)
        s.tick(layer_id=2, seq_len=100)
        # New generation with different seq_len → reset
        s.tick(layer_id=1, seq_len=200)
        assert s._generation_seq_len == 200
        assert s._step == 0

    def test_n_blocks_auto_calibration(self):
        s = HunyuanLVSAState(LVSAConfig())
        # Simulate 3 distinct attention layer ids, then repeat — calibration on repeat
        s.tick(layer_id=1, seq_len=100)
        s.tick(layer_id=2, seq_len=100)
        s.tick(layer_id=3, seq_len=100)
        assert s._n_blocks is None  # not yet — no repeat seen
        s.tick(layer_id=1, seq_len=100)   # repeats layer 1
        assert s._n_blocks == 3

    def test_step_advances_after_n_blocks_x_cfg_passes(self, monkeypatch):
        monkeypatch.setenv("LVSA_CFG_PASSES", "1")  # no CFG, simpler arithmetic
        s = HunyuanLVSAState(LVSAConfig())
        # Calibrate n_blocks=3 with 3 unique layers then a repeat
        for lid in (1, 2, 3, 1):
            s.tick(layer_id=lid, seq_len=100)
        assert s._n_blocks == 3
        # After calibration, _seen_ids was cleared and _call_count keeps growing.
        # We need (n_blocks * cfg_passes) = 3 calls per step.
        # We've done 4 calls so far; step at this point = (4-1)//3 = 1.
        assert s._step == 1


class TestHunyuanLVSAStateMetadata:
    def _ready_state(self, **cfg_kwargs):
        return HunyuanLVSAState(LVSAConfig(**cfg_kwargs))

    def test_metadata_cached_when_unchanged(self):
        s = self._ready_state(rotate_keyframes=False)
        m1 = s.get_metadata(33, 1560, 0, torch.device("cpu"))
        m2 = s.get_metadata(33, 1560, 0, torch.device("cpu"))
        assert m1 is m2

    def test_metadata_rebuilt_on_t_lat_change(self):
        s = self._ready_state(rotate_keyframes=False)
        m1 = s.get_metadata(33, 1560, 0, torch.device("cpu"))
        m2 = s.get_metadata(49, 1560, 0, torch.device("cpu"))
        assert m1 is not m2
        # LVSAMetadata exposes total_latent_frames; cache key in HunyuanLVSAState
        # tracks the same value via _cached_total_frames.
        assert s._cached_total_frames == 49

    def test_metadata_rebuilt_on_step_when_rotating(self):
        s = self._ready_state(rotate_keyframes=True)
        m1 = s.get_metadata(49, 1560, 0, torch.device("cpu"))
        m2 = s.get_metadata(49, 1560, 1, torch.device("cpu"))
        # Different keyframe offsets → different metadata objects
        assert m1 is not m2

    def test_metadata_cached_across_steps_when_not_rotating(self):
        s = self._ready_state(rotate_keyframes=False)
        m1 = s.get_metadata(49, 1560, 0, torch.device("cpu"))
        m2 = s.get_metadata(49, 1560, 5, torch.device("cpu"))
        # Step changes but rotation off → cached
        assert m1 is m2


# ── Import / install API surface ─────────────────────────────────────────────


class TestImportSurface:
    def test_module_importable(self):
        """The hunyuan_hook module imports cleanly without vllm-omni."""
        from lvsa_vllm_omni import hunyuan_hook
        assert hasattr(hunyuan_hook, "install_hunyuan_lvsa_hook")
        assert hasattr(hunyuan_hook, "HunyuanLVSAState")

    def test_install_function_lazy_imports_vllm_omni(self):
        """install_hunyuan_lvsa_hook must defer the vllm-omni import until called."""
        # Just importing the module should not require vllm-omni
        from lvsa_vllm_omni.hunyuan_hook import install_hunyuan_lvsa_hook
        # Calling it without vllm-omni should raise an ImportError, not an arbitrary error
        with pytest.raises((ImportError, ModuleNotFoundError)):
            install_hunyuan_lvsa_hook(total_latent_frames=33)
