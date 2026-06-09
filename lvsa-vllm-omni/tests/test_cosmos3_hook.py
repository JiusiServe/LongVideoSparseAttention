"""Tests for the NVIDIA Cosmos 3.0 LVSA hook.

Like ``test_hunyuan_hook``: the actual ``install_cosmos3_lvsa_hook`` monkey-patches
``Cosmos3CrossAttention`` and needs vllm-omni installed, so the closure dispatch
logic (SP→dense fallback, geometry-mismatch→dense fallback, ``LVSA_REPEAT_KV``
toggle) is exercised only by the GPU/integration sweep, not here. On CPU we cover:
  - Import surface: the module imports without vllm-omni; the install function
    exists and defers the vllm-omni import until called.
  - The cosmos GEOMETRY fed to the (reused) ``HunyuanLVSAState.get_metadata`` —
    Cosmos3-Nano's native horizon (48 latent frames, P=920 @720p): the dense
    regime at 1x and genuine sparsity above it. This is the path the dispatch
    closure relies on, validated at cosmos numbers (the hunyuan tests use 33/1560).
"""
from __future__ import annotations

import os

import pytest
import torch

from lvsa_vllm_omni.config import LVSAConfig

# Cosmos3-Nano @720p: 189 frames -> 48 latent frames; P = ceil(720/32)*ceil(1280/32)
COSMOS3_REF_LAT = 48
COSMOS3_P_720P = 920


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("LVSA_"):
            monkeypatch.delenv(key, raising=False)
    yield


# ── Import / install API surface ─────────────────────────────────────────────


class TestImportSurface:
    def test_module_importable(self):
        """cosmos3_hook imports cleanly without vllm-omni."""
        from lvsa_vllm_omni import cosmos3_hook
        assert hasattr(cosmos3_hook, "install_cosmos3_lvsa_hook")

    def test_reuses_hunyuan_state_and_helpers(self):
        """The hook reuses hunyuan_hook's state/logging rather than duplicating."""
        from lvsa_vllm_omni import cosmos3_hook
        from lvsa_vllm_omni.hunyuan_hook import HunyuanLVSAState
        # Imported into the cosmos module namespace (the dispatch builds one).
        assert cosmos3_hook.HunyuanLVSAState is HunyuanLVSAState

    def test_install_function_lazy_imports_vllm_omni(self):
        """install_cosmos3_lvsa_hook must defer the vllm-omni import until called."""
        from lvsa_vllm_omni.cosmos3_hook import install_cosmos3_lvsa_hook
        # Importing the module is fine; calling install without vllm-omni must
        # raise an ImportError, not an arbitrary error.
        with pytest.raises((ImportError, ModuleNotFoundError)):
            install_cosmos3_lvsa_hook(total_latent_frames=COSMOS3_REF_LAT)


# ── Cosmos geometry through the reused state (the dispatch's metadata path) ───


class TestCosmos3Geometry:
    def test_dense_regime_at_native_horizon(self):
        """T_lat == reference (48) → kfi=1, every frame global (dense LVSA path)."""
        from lvsa_vllm_omni.hunyuan_hook import HunyuanLVSAState
        s = HunyuanLVSAState(LVSAConfig(reference_latent_frames=COSMOS3_REF_LAT))
        m = s.get_metadata(COSMOS3_REF_LAT, COSMOS3_P_720P, 0, torch.device("cpu"))
        assert m.num_patches == COSMOS3_P_720P
        assert m.total_latent_frames == COSMOS3_REF_LAT
        assert m.key_frame_interval == 1
        assert len(m.global_indices) == COSMOS3_REF_LAT   # all frames global → dense

    def test_sparse_engages_above_horizon(self):
        """T_lat = 2x reference (96) → fewer than all frames are global anchors."""
        from lvsa_vllm_omni.hunyuan_hook import HunyuanLVSAState
        s = HunyuanLVSAState(LVSAConfig(reference_latent_frames=COSMOS3_REF_LAT))
        T_lat = 2 * COSMOS3_REF_LAT
        m = s.get_metadata(T_lat, COSMOS3_P_720P, 0, torch.device("cpu"))
        assert m.key_frame_interval > 1
        assert 0 < len(m.global_indices) < T_lat          # genuinely sparse

    def test_metadata_cached_when_unchanged(self):
        from lvsa_vllm_omni.hunyuan_hook import HunyuanLVSAState
        s = HunyuanLVSAState(LVSAConfig(reference_latent_frames=COSMOS3_REF_LAT))
        m1 = s.get_metadata(96, COSMOS3_P_720P, 0, torch.device("cpu"))
        m2 = s.get_metadata(96, COSMOS3_P_720P, 0, torch.device("cpu"))
        assert m1 is m2
