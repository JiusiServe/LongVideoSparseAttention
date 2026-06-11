import pytest


def test_cosmos3_geometry_720p_189f():
    from lvsa.cosmos3 import (cosmos3_latent_frames, cosmos3_patches_per_frame,
                              COSMOS3_REFERENCE_LATENT_FRAMES)
    # VAE temporal factor 4: (189-1)//4 + 1 = 48
    assert cosmos3_latent_frames(189) == 48
    assert cosmos3_latent_frames(1) == 1
    # VAE spatial 16, latent patch 2 -> P = ceil(H/32)*ceil(W/32); 720x1280 -> 23*40
    assert cosmos3_patches_per_frame(720, 1280) == 23 * 40 == 920
    assert COSMOS3_REFERENCE_LATENT_FRAMES == 48


import torch
import torch.nn as nn


class _FakeCosmosAttn(nn.Module):
    """Minimal stand-in exposing the attributes both processors read."""
    def __init__(self, hidden=64, heads=4, kv_heads=2, head_dim=16):
        super().__init__()
        self.num_attention_heads = heads
        self.num_key_value_heads = kv_heads
        self.head_dim = head_dim
        qd, kvd = heads * head_dim, kv_heads * head_dim
        self.to_q = nn.Linear(hidden, qd, bias=False)
        self.to_k = nn.Linear(hidden, kvd, bias=False)
        self.to_v = nn.Linear(hidden, kvd, bias=False)
        self.add_q_proj = nn.Linear(hidden, qd, bias=False)
        self.add_k_proj = nn.Linear(hidden, kvd, bias=False)
        self.add_v_proj = nn.Linear(hidden, kvd, bias=False)
        self.norm_q = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_added_q = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_added_k = nn.RMSNorm(head_dim, eps=1e-6)
        self.to_out = nn.Linear(qd, hidden, bias=False)
        self.to_add_out = nn.Linear(qd, hidden, bias=False)


def _rope_tuple(s_und, s_gen, head_dim):
    # identity-ish cos/sin (cos=1, sin=0) so RoPE is a no-op -> exact compare
    cos_u = torch.ones(s_und, head_dim); sin_u = torch.zeros(s_und, head_dim)
    cos_g = torch.ones(s_gen, head_dim); sin_g = torch.zeros(s_gen, head_dim)
    return (cos_u, sin_u, cos_g, sin_g)


def test_lvsa_processor_matches_dense_at_1x():
    # Cosmos3 lives in diffusers main only — skip on release diffusers (e.g. CI).
    pytest.importorskip("diffusers.models.transformers.transformer_cosmos3")
    from diffusers.models.transformers.transformer_cosmos3 import Cosmos3AttnProcessor
    from lvsa.cosmos3 import Cosmos3LVSAAttnProcessor
    torch.manual_seed(0)
    T_lat, P, head_dim = 6, 2, 16          # tiny grid; T_lat == ref below
    attn = _FakeCosmosAttn(head_dim=head_dim).eval()
    und = torch.randn(5, 64)               # 5 und (text) tokens
    gen = torch.randn(T_lat * P, 64)       # gen = clean frame grid
    rot = _rope_tuple(5, T_lat * P, head_dim)
    ref_und, ref_gen = Cosmos3AttnProcessor()(attn, und, gen, rot)
    # ref=T_lat -> 1x horizon -> kfi=1 -> every gen frame global -> dense
    proc = Cosmos3LVSAAttnProcessor(total_latent_frames=T_lat, num_patches=P,
                                    reference_latent_frames=T_lat)
    my_und, my_gen = proc(attn, und, gen, rot)
    assert torch.allclose(my_und, ref_und, atol=1e-5), "und path must be untouched"
    assert torch.allclose(my_gen, ref_gen, atol=1e-5), "gen LVSA must == dense at 1x"


def test_lvsa_processor_engages_sparse_above_ref():
    # The processor's __call__ lazily imports transformer_cosmos3 (main-only).
    pytest.importorskip("diffusers.models.transformers.transformer_cosmos3")
    from lvsa.cosmos3 import Cosmos3LVSAAttnProcessor
    torch.manual_seed(0)
    T_lat, P, ref, head_dim = 24, 2, 6, 16      # 4x horizon -> sparse
    attn = _FakeCosmosAttn(head_dim=head_dim).eval()
    und = torch.randn(4, 64)
    gen = torch.randn(T_lat * P, 64)
    rot = _rope_tuple(4, T_lat * P, head_dim)
    proc = Cosmos3LVSAAttnProcessor(total_latent_frames=T_lat, num_patches=P,
                                    reference_latent_frames=ref)
    my_und, my_gen = proc(attn, und, gen, rot)
    assert my_gen.shape == (T_lat * P, 64)
    assert torch.isfinite(my_gen).all()
    # sparse: fewer than all frames are global anchors
    assert 0 < len(proc.metadata.global_indices) < T_lat


def test_install_swaps_all_layers():
    import torch.nn as nn
    from lvsa.cosmos3 import install_cosmos3_lvsa, Cosmos3LVSAAttnProcessor

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _FakeCosmosAttn()
            self.self_attn.processor = object()   # stand-in original processor
            self.self_attn.set_processor = lambda p: setattr(self.self_attn, "processor", p)

    class _Transformer(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.layers = nn.ModuleList([_Layer() for _ in range(n)])

    tf = _Transformer(5)
    n = install_cosmos3_lvsa(tf, num_frames=189, height=720, width=1280)
    assert n == 5
    for layer in tf.layers:
        assert isinstance(layer.self_attn.processor, Cosmos3LVSAAttnProcessor)


def test_batched_input_raises_not_corrupts():
    """B>1 must raise, never silently corrupt. The processor mirrors the stock
    ``Cosmos3AttnProcessor`` byte-for-byte, and stock flattens batch into
    sequence (``view(-1, H, D)`` + ``unsqueeze(0)``): at B>1 the und causal
    mask would leak across batch elements and the gen LVSA metadata (built for
    S_gen) would mis-cover a B*S_gen query. ``Cosmos3OmniPipeline`` never
    batches (sequential CFG, one sample per call — verified on diffusers main),
    so the guard only trips if a future caller batches. No diffusers import
    needed: the guard fires before ``__call__``'s lazy import, so this test
    runs on release diffusers too.
    """
    from lvsa.cosmos3 import Cosmos3LVSAAttnProcessor, _require_unbatched

    # unit: the guard itself
    ok_2d = torch.randn(6, 64)            # [S, C] (implicit single sample)
    ok_3d = torch.randn(1, 6, 64)         # [1, S, C] (explicit B=1)
    bad = torch.randn(2, 6, 64)           # [B=2, S, C]
    _require_unbatched(ok_2d, ok_2d)      # no raise
    _require_unbatched(ok_3d, ok_3d)      # no raise
    with pytest.raises(NotImplementedError, match="batch"):
        _require_unbatched(bad, ok_3d)
    with pytest.raises(NotImplementedError, match="batch"):
        _require_unbatched(ok_3d, bad)

    # integration: __call__ rejects batched inputs before touching attn
    proc = Cosmos3LVSAAttnProcessor(
        total_latent_frames=4, num_patches=2, reference_latent_frames=2,
    )
    with pytest.raises(NotImplementedError, match="batch"):
        proc(attn=None, und_seq=bad, gen_seq=bad, rotary_emb=None)
