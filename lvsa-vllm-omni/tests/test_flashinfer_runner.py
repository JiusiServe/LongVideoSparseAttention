"""Numerical-parity tests for the FlashInfer LSE-merge runner.

The shared ``FlashInferDualStreamRunner`` computes the gen video tokens with a
block-sparse kernel and the und/encoder tokens with a SEPARATE dense
``single_prefill`` term, merged via exact log-sum-exp. This replaced the old
zero-padded-encoder-block approach (which attended phantom zero keys). The merge
math + the runner's NATIVE-GQA compact path (it keeps K/V at ``Hkv``, unlike the
standalone's ``H``) are the parts most worth a guard.

These require a CUDA GPU + FlashInfer, so they are skipped on the CPU suite. Run
in the flashinfer env, e.g.:
    .venv-vllm-main/bin/python -m pytest lvsa-vllm-omni/tests/test_flashinfer_runner.py -v

Cases (all GQA, ``Hkv < H``, the Cosmos/HunyuanVideo layout):
  1. SPARSE pattern (kfi>1): runner output == ``lvsa_sdpa`` output (the two
     backends compute the same block-sparse attention by different kernels).
  2. DENSE regime (T_lat == reference → kfi=1, every gen frame global): both the
     runner and ``lvsa_sdpa`` must match a true dense full-attention reference —
     absolute ground truth, not just fi==sdpa.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from lvsa.sparse_attention import LVSAMetadata, lvsa_sdpa
from lvsa_vllm_omni.global_kv import build_global_kv
from lvsa_vllm_omni.flashinfer_runner import (
    FlashInferDualStreamRunner,
    FLASHINFER_AVAILABLE,
)

# head_dim 128 hits FlashInfer's prebuilt (AOT) bf16 kernels — no nvcc/JIT.
H, HKV, D = 8, 2, 128          # GQA 4:1 (Cosmos-like)
P = 8                          # tokens/frame (small for speed)
S_UND = 16                     # und/encoder tokens


def _flashinfer_kernel_runs() -> bool:
    """True only if FlashInfer can actually EXECUTE a kernel here.

    ``import flashinfer`` succeeds even when the kernel later fails to JIT-compile
    ("Could not find nvcc") — that happens in envs with plain flashinfer but no
    nvcc/AOT cache (e.g. the standalone .venv). The flashinfer-jit-cache env
    (.venv-vllm-main) runs the prebuilt bf16/head_dim-128 kernels. Probe a tiny
    real kernel so this whole file is skipped where flashinfer can't run.
    """
    if not (torch.cuda.is_available() and FLASHINFER_AVAILABLE):
        return False
    try:
        import flashinfer
        t = torch.randn(4, HKV, D, device="cuda", dtype=torch.bfloat16)
        flashinfer.single_prefill_with_kv_cache(t, t, t, causal=False)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _flashinfer_kernel_runs(),
    reason="FlashInfer kernel not runnable here (needs CUDA GPU + AOT cache / nvcc)",
)


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-8)).item()


def _make_inputs(T_lat: int, ref: int, *, kfi, window, n_first, seed=0):
    torch.manual_seed(seed)
    dev, dt = torch.device("cuda"), torch.bfloat16
    S_gen = T_lat * P
    q = torch.randn(1, S_gen, H, D, device=dev, dtype=dt)
    k = torch.randn(1, S_gen, HKV, D, device=dev, dtype=dt)
    v = torch.randn(1, S_gen, HKV, D, device=dev, dtype=dt)
    k_und = torch.randn(1, S_UND, HKV, D, device=dev, dtype=dt)
    v_und = torch.randn(1, S_UND, HKV, D, device=dev, dtype=dt)
    meta = LVSAMetadata.build(
        total_latent_frames=T_lat, num_patches=P, window_size=window,
        n_first_frames=n_first, key_frame_interval=kfi, rank=0, world=1,
        reference_frames=ref,
    )
    # Same construction the hooks use: gen globals + und appended as always-global.
    k_global, v_global = build_global_kv(k, v, meta.global_indices, P)
    k_global = torch.cat([k_global, k_und], dim=1)
    v_global = torch.cat([v_global, v_und], dim=1)
    return q, k, v, k_global, v_global, k_und, v_und, meta


def test_runner_matches_lvsa_sdpa_sparse_gqa():
    # 4x horizon -> genuinely sparse (kfi auto > 1); GQA Hkv<H.
    q, k, v, k_g, v_g, _, _, meta = _make_inputs(
        T_lat=24, ref=6, kfi=None, window=2, n_first=1,
    )
    assert 0 < len(meta.global_indices) < 24, "expected a sparse pattern"

    out_fi = FlashInferDualStreamRunner().run(q, k, v, k_g, v_g, meta)
    out_sdpa = lvsa_sdpa(q, k, v, k_g, v_g, meta)

    assert out_fi.shape == q.shape
    assert torch.isfinite(out_fi.float()).all()
    rel = _rel_l2(out_fi, out_sdpa)
    assert rel < 2e-2, f"FlashInfer vs lvsa_sdpa rel-L2 {rel:.4f} too large (>2%)"


def test_runner_and_sdpa_match_dense_at_kfi1_gqa():
    # T_lat == reference -> kfi=1 -> every gen frame global -> full attention.
    q, k, v, k_g, v_g, k_und, v_und, meta = _make_inputs(
        T_lat=6, ref=6, kfi=1, window=2, n_first=1,
    )
    assert meta.key_frame_interval == 1
    assert len(meta.global_indices) == 6, "kfi=1 should make every frame global"

    # Dense ground truth: gen queries attend to all gen + und K/V (GQA native).
    all_k = torch.cat([k, k_und], dim=1).transpose(1, 2)   # [B,Hkv,S,D]
    all_v = torch.cat([v, v_und], dim=1).transpose(1, 2)
    dense = F.scaled_dot_product_attention(
        q.transpose(1, 2), all_k, all_v, enable_gqa=True,
    ).transpose(1, 2)                                      # [B,S,H,D]

    out_fi = FlashInferDualStreamRunner().run(q, k, v, k_g, v_g, meta)
    out_sdpa = lvsa_sdpa(q, k, v, k_g, v_g, meta)

    rel_fi = _rel_l2(out_fi, dense)
    rel_sdpa = _rel_l2(out_sdpa, dense)
    assert rel_sdpa < 2e-2, f"lvsa_sdpa vs dense rel-L2 {rel_sdpa:.4f} (>2%)"
    assert rel_fi < 2e-2, f"FlashInfer vs dense rel-L2 {rel_fi:.4f} (>2%)"
