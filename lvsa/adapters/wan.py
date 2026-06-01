"""
wan.py — ModelAdapter implementation for Wan 2.x video diffusion models.

Extracts all Wan-specific attention layer logic (QKV projections, I2V
cross-attention, RoPE format, output projection, layer naming) so the
LVSA engine in ``DistributedLVSAProcessor`` stays model-agnostic.
"""

from typing import Any, Optional, Tuple

import torch

from ..rope import apply_rotary_emb, slice_rotary_emb
from .base import ModelAdapter


class WanAdapter(ModelAdapter):
    """Adapter bridging Wan 2.x attention layers to the LVSA LVSA engine."""

    # ── Geometry ──────────────────────────────────────────────────────────────

    def patches_per_frame(self, height: int, width: int, pipe: Any) -> int:
        patch_size_cfg = getattr(pipe.transformer.config, "patch_size", [1, 2, 2])
        if isinstance(patch_size_cfg, (list, tuple)):
            patch_size_s = int(patch_size_cfg[1])
        else:
            patch_size_s = int(patch_size_cfg)
        vae_s = pipe.vae_scale_factor_spatial  # typically 8
        return (height // (vae_s * patch_size_s)) * (width // (vae_s * patch_size_s))

    def latent_frames(self, num_frames: int, pipe: Any) -> int:
        vae_t = pipe.vae_scale_factor_temporal  # typically 4
        return (num_frames - 1) // vae_t + 1

    def reference_latent_frames(self, pipe: Any) -> int:
        """Wan 2.x reference: 81 raw → 21 latent frames.

        Tries to compute from config (``sample_frames`` / VAE temporal),
        falls back to 21 latent frames if config is missing.
        """
        sample_frames = getattr(pipe.transformer.config, "sample_frames", None)
        if sample_frames is not None:
            return self.latent_frames(sample_frames, pipe)
        return 21  # (81 - 1) // 4 + 1

    # ── QKV extraction ────────────────────────────────────────────────────────

    def extract_qkv(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from diffusers.models.transformers.transformer_wan import _get_qkv_projections

        query, key, value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states,
        )
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # [B, seq, C] -> [B, seq, H, D]
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        return query, key, value

    def split_encoder_for_cross_attn(
        self,
        attn: Any,
        encoder_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if attn.add_k_proj is not None:
            # Wan I2V: first (seq - 512) tokens are image context
            image_ctx_len = encoder_hidden_states.shape[1] - 512
            encoder_image = encoder_hidden_states[:, :image_ctx_len]
            encoder_text = encoder_hidden_states[:, image_ctx_len:]
            return encoder_text, encoder_image
        return encoder_hidden_states, None

    def extract_cross_attn_kv(
        self,
        attn: Any,
        encoder_hidden_states: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if attn.add_k_proj is None:
            return None

        from diffusers.models.transformers.transformer_wan import (
            _get_added_kv_projections,
        )

        key_img, value_img = _get_added_kv_projections(
            attn, encoder_hidden_states,
        )
        key_img = attn.norm_added_k(key_img).unflatten(2, (attn.heads, -1))
        value_img = value_img.unflatten(2, (attn.heads, -1))
        return key_img, value_img

    # ── RoPE ──────────────────────────────────────────────────────────────────

    def apply_rotary(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        rotary_emb: Any,
        local_seq: int,
        rank: int,
        world: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rotary_emb is None:
            return query, key
        rotary_emb = slice_rotary_emb(rotary_emb, local_seq, rank, world)
        query = apply_rotary_emb(query, *rotary_emb)
        key = apply_rotary_emb(key, *rotary_emb)
        return query, key

    # ── Output projection ─────────────────────────────────────────────────────

    def output_projection(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        query_dtype: torch.dtype,
    ) -> torch.Tensor:
        hidden_states = hidden_states.flatten(2, 3).type_as(
            torch.empty((), dtype=query_dtype, device=hidden_states.device)
        )
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    # ── Cross-attention dispatch ──────────────────────────────────────────────

    def cross_attention(
        self,
        attn: Any,
        query: torch.Tensor,
        encoder_image: torch.Tensor,
        attention_backend: Any,
    ) -> Optional[torch.Tensor]:
        if encoder_image is None:
            return None

        from diffusers.models.attention_dispatch import dispatch_attention_fn

        key_img, value_img = self.extract_cross_attn_kv(attn, encoder_image)
        if key_img is None:
            return None

        out_img = dispatch_attention_fn(
            query,
            key_img,
            value_img,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=attention_backend,
            parallel_config=None,
        )
        return out_img.flatten(2, 3).type_as(query)

    # ── Pipeline integration ──────────────────────────────────────────────────

    def install_processor(self, pipe: Any, processor: Any) -> int:
        for block in pipe.transformer.blocks:
            block.attn1.processor = processor
        return len(pipe.transformer.blocks)

    def setup_context_parallel(self, transformer: Any, world: int) -> None:
        from diffusers import ContextParallelConfig
        from diffusers.models._modeling_parallel import (
            ContextParallelInput,
            ContextParallelOutput,
        )

        transformer._cp_plan = {
            "blocks.0": {
                "hidden_states": ContextParallelInput(
                    split_dim=1,
                    expected_dims=3,
                    split_output=False,
                ),
            },
            "proj_out": ContextParallelOutput(
                gather_dim=1,
                expected_dims=3,
            ),
        }
        transformer.enable_parallelism(
            config=ContextParallelConfig(ulysses_degree=world),
        )

    def patch_rotary_for_cp(self, rank: int, world: int) -> None:
        import diffusers.models.transformers.transformer_wan as _wan_module

        _orig_call = _wan_module.WanAttnProcessor.__call__

        def _patched_call(
            self_proc,
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            rotary_emb=None,
            **kwargs,
        ):
            if rotary_emb is not None and isinstance(rotary_emb, (tuple, list)):
                local_seq = hidden_states.shape[1]
                rotary_emb = slice_rotary_emb(rotary_emb, local_seq, rank, world)
            return _orig_call(
                self_proc,
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                rotary_emb,
                **kwargs,
            )

        _wan_module.WanAttnProcessor.__call__ = _patched_call
