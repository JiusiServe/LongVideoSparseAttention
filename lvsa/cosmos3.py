"""LVSA for NVIDIA Cosmos 3.0 (diffusers standalone).

Cosmos 3's diffusers attention (Cosmos3AttnProcessor) is separate-stream:
und (text/VLM) does causal self-attn; gen (video) does full attn where q_gen
attends to cat([k_und, k_gen]). We swap the processor and LVSA-ize ONLY the gen
pathway (window gen<->gen + keyframes, all und global). und is left untouched.
Geometry: VAE spatial 16 / temporal 4, latent patch 2.
"""
import math

import torch
from lvsa.sparse_attention import (
    LVSAMetadata, lvsa_sdpa, build_global_kv, compute_auto_kfi,
)

# Cosmos3-Nano native horizon: 189 frames -> 48 latent frames.
COSMOS3_REFERENCE_LATENT_FRAMES = 48
_VAE_TEMPORAL = 4
_VAE_SPATIAL = 16
_LATENT_PATCH = 2


def cosmos3_latent_frames(num_frames: int) -> int:
    """Raw video frames -> latent temporal frames (VAE temporal factor 4)."""
    return (num_frames - 1) // _VAE_TEMPORAL + 1


def cosmos3_patches_per_frame(height: int, width: int) -> int:
    """Tokens per latent frame = ceil(H/32) * ceil(W/32)
    (VAE spatial 16, then latent patch 2)."""
    lat_h = math.ceil((height / _VAE_SPATIAL) / _LATENT_PATCH)
    lat_w = math.ceil((width / _VAE_SPATIAL) / _LATENT_PATCH)
    return lat_h * lat_w


def _require_unbatched(und_seq, gen_seq) -> None:
    """Reject batched (B>1) inputs — raise instead of silently corrupting.

    This processor mirrors the stock ``Cosmos3AttnProcessor`` byte-for-byte,
    and stock flattens batch into sequence (``view(-1, H, D)`` then
    ``unsqueeze(0)``). At B>1 that breaks both pathways: the und causal mask
    leaks across batch elements, and the gen LVSA metadata (built for S_gen)
    mis-covers a B*S_gen query. ``Cosmos3OmniPipeline`` never batches (CFG is
    two sequential transformer calls; one sample per call), so this guard only
    trips if a future caller batches. Proper B>1 support belongs upstream
    first — stock shares the limitation.
    """
    for name, seq in (("und_seq", und_seq), ("gen_seq", gen_seq)):
        if seq is not None and seq.ndim == 3 and seq.shape[0] != 1:
            raise NotImplementedError(
                f"Cosmos3LVSAAttnProcessor received batched {name} "
                f"(batch={seq.shape[0]}). batch>1 is unsupported: the stock "
                "Cosmos3AttnProcessor this mirrors flattens batch into "
                "sequence, which cross-contaminates samples. Run samples "
                "sequentially (Cosmos3OmniPipeline already does)."
            )


class Cosmos3LVSAAttnProcessor:
    """Drop-in for diffusers Cosmos3AttnProcessor that LVSA-izes the gen path.

    und (text/VLM) causal self-attn is replicated verbatim; only the gen full
    attention (q_gen over cat([k_und, k_gen])) is replaced by the LVSA windowed
    pattern: window gen<->gen + keyframes/n_first, with ALL k_und kept global.
    Single-GPU (world=1), fixed keyframes (no rotation) for the MVP.
    """

    def __init__(self, total_latent_frames, num_patches, reference_latent_frames,
                 window_size=12, n_first_frames=4, sparsity_scale=1.0,
                 attention_backend=None):
        self.P = num_patches
        self.attention_backend = attention_backend
        # video-frame knobs -> latent-frame knobs (VAE temporal 4)
        w = max(1, window_size // _VAE_TEMPORAL)
        nf = max(1, n_first_frames // _VAE_TEMPORAL)
        kfi = compute_auto_kfi(total_latent_frames, w, nf,
                               reference_frames=reference_latent_frames,
                               sparsity_scale=sparsity_scale)
        self.metadata = LVSAMetadata.build(
            total_latent_frames=total_latent_frames, num_patches=num_patches,
            window_size=w, n_first_frames=nf, key_frame_interval=kfi,
            rank=0, world=1, expand_window=True,
            reference_frames=reference_latent_frames, sparsity_scale=sparsity_scale,
        )

    def __call__(self, attn, und_seq, gen_seq, rotary_emb):
        _require_unbatched(und_seq, gen_seq)
        # NOTE: mirrors diffusers Cosmos3AttnProcessor internals (private _rotate_half + projection attr names); re-verify on diffusers bump.
        from diffusers.models.transformers.transformer_cosmos3 import (
            _rotate_half, dispatch_attention_fn,
        )
        # projections (mirror Cosmos3AttnProcessor)
        H, Hkv, D = attn.num_attention_heads, attn.num_key_value_heads, attn.head_dim
        q_und = attn.to_q(und_seq).view(-1, H, D)
        k_und = attn.to_k(und_seq).view(-1, Hkv, D)
        v_und = attn.to_v(und_seq).view(-1, Hkv, D)
        q_gen = attn.add_q_proj(gen_seq).view(-1, H, D)
        k_gen = attn.add_k_proj(gen_seq).view(-1, Hkv, D)
        v_gen = attn.add_v_proj(gen_seq).view(-1, Hkv, D)
        # QK-norm
        q_und = attn.norm_q(q_und); k_und = attn.norm_k(k_und)
        q_gen = attn.norm_added_q(q_gen); k_gen = attn.norm_added_k(k_gen)
        # RoPE per pathway
        cos_u, sin_u, cos_g, sin_g = rotary_emb
        cos_u, sin_u = cos_u.unsqueeze(1), sin_u.unsqueeze(1)
        q_und = q_und * cos_u + _rotate_half(q_und) * sin_u
        k_und = k_und * cos_u + _rotate_half(k_und) * sin_u
        cos_g, sin_g = cos_g.unsqueeze(1), sin_g.unsqueeze(1)
        q_gen = q_gen * cos_g + _rotate_half(q_gen) * sin_g
        k_gen = k_gen * cos_g + _rotate_half(k_gen) * sin_g

        # und pathway: causal self-attn (UNCHANGED)
        causal_out = dispatch_attention_fn(
            q_und.unsqueeze(0), k_und.unsqueeze(0), v_und.unsqueeze(0),
            is_causal=True, enable_gqa=True, backend=self.attention_backend,
        ).squeeze(0).flatten(-2, -1)
        und_out = attn.to_out(causal_out)

        # gen pathway: LVSA (replaces dense full-attn)
        qg = q_gen.unsqueeze(0)            # [1, S_gen, H, D]
        kg, vg = k_gen.unsqueeze(0), v_gen.unsqueeze(0)
        kglob, vglob = build_global_kv(kg, vg, self.metadata.global_indices, self.P)
        # und is always-global -> append after the gen anchors
        kglob = torch.cat([kglob, k_und.unsqueeze(0)], dim=1)
        vglob = torch.cat([vglob, v_und.unsqueeze(0)], dim=1)
        out = lvsa_sdpa(qg, kg, vg, kglob, vglob, self.metadata,
                        attention_backend=self.attention_backend)
        gen_out = attn.to_add_out(out.squeeze(0).flatten(-2, -1))
        return und_out, gen_out


def install_cosmos3_lvsa(transformer, num_frames, height, width,
                         window_size=12, n_first_frames=4, sparsity_scale=1.0,
                         reference_latent_frames=COSMOS3_REFERENCE_LATENT_FRAMES,
                         attention_backend=None):
    """Swap every layer's self-attn processor with the LVSA gen-pathway version.

    Returns the number of layers patched. Single-GPU only (world=1). All layers
    share one processor (same geometry); the processor is stateless per call.
    """
    T_lat = cosmos3_latent_frames(num_frames)
    P = cosmos3_patches_per_frame(height, width)
    proc = Cosmos3LVSAAttnProcessor(
        total_latent_frames=T_lat, num_patches=P,
        reference_latent_frames=reference_latent_frames,
        window_size=window_size, n_first_frames=n_first_frames,
        sparsity_scale=sparsity_scale, attention_backend=attention_backend,
    )
    n = 0
    for layer in transformer.layers:
        layer.self_attn.set_processor(proc)
        n += 1
    if T_lat <= reference_latent_frames:
        print(f"[LVSA] Cosmos3: T_lat={T_lat} <= ref={reference_latent_frames} "
              f"-> dense regime (no sparsity at this horizon).")
    else:
        print(f"[LVSA] Cosmos3: installed on {n} layers  T_lat={T_lat} P={P}  "
              f"kfi={proc.metadata.key_frame_interval} "
              f"globals={len(proc.metadata.global_indices)}")
    return n
