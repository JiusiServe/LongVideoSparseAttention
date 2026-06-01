"""
cogvideox.py — ModelAdapter implementation for CogVideoX.

CogVideoX uses a **single-stream joint attention** design where video and text
tokens are passed separately to the attention processor, which concatenates them
internally for joint QKV projection.  The block's ``forward()`` expects the
attention processor to return a tuple ``(attn_hidden_states, attn_encoder_hidden_states)``.

Architecturally, this maps to the dual-stream adapter pattern: text tokens are
treated as encoder tokens that get full attention against the entire K/V space,
while video tokens get LVSA.

Key differences from HunyuanVideo:
  - **Shared** ``to_q/to_k/to_v`` weights for both text and video
    (HunyuanVideo has separate ``add_q/k/v_proj`` for encoder)
  - **Shared** ``to_out`` projection for both text and video
    (HunyuanVideo has separate ``to_add_out`` for encoder)
  - Layer naming: ``transformer_blocks[i].attn1``
    (HunyuanVideo uses ``transformer_blocks[i].attn``)
  - VAE spatial scale factor is 8 (HunyuanVideo is 16)
  - 3D RoPE applied to video tokens only (same as HunyuanVideo)

Similarities with HunyuanVideo:
  - QK-norm (LayerNorm) applied per-head after unflatten
  - ``image_rotary_emb`` kwarg for RoPE
  - No separate cross-attention path
  - Block returns (hidden_states, encoder_hidden_states) tuple
"""

from typing import Any, Optional, Tuple

import torch

from .base import ModelAdapter


class CogVideoXAdapter(ModelAdapter):
    """Adapter bridging CogVideoX joint attention to LVSA.

    CogVideoX projects text and video tokens through the **same**
    ``to_q/to_k/to_v`` weights.  Since linear projections distribute over
    concatenation (``linear(cat(a, b)) == cat(linear(a), linear(b))``),
    we can project them separately and treat the result identically to a
    dual-stream model.
    """

    # ── Geometry ──────────────────────────────────────────────────────────────

    def patches_per_frame(self, height: int, width: int, pipe: Any) -> int:
        patch_size = getattr(pipe.transformer.config, "patch_size", 2)
        if isinstance(patch_size, (list, tuple)):
            patch_size = int(patch_size[-1])
        else:
            patch_size = int(patch_size)

        # CogVideoX VAE spatial downsampling factor is 8
        vae_s = getattr(pipe, "vae_scale_factor_spatial", 8)
        p = (height // (vae_s * patch_size)) * (width // (vae_s * patch_size))

        # If temporal patch size is set (CogVideoX-1.5), each "token frame"
        # covers patch_size_t latent frames, so P stays spatial-only.
        return p

    def latent_frames(self, num_frames: int, pipe: Any) -> int:
        vae_t = getattr(pipe, "vae_scale_factor_temporal", 4)
        t_lat = (num_frames - 1) // vae_t + 1

        # CogVideoX-1.5 may have temporal patch size that further compresses
        patch_size_t = getattr(pipe.transformer.config, "patch_size_t", None)
        if patch_size_t is not None and patch_size_t > 1:
            t_lat = -(-t_lat // patch_size_t)  # ceil division

        return t_lat

    def reference_latent_frames(self, pipe: Any) -> int:
        """CogVideoX reference: 49 raw → 13 latent frames.

        Tries to compute from config (``sample_frames`` / VAE temporal),
        falls back to 13 latent frames if config is missing.
        """
        sample_frames = getattr(pipe.transformer.config, "sample_frames", None)
        if sample_frames is not None:
            return self.latent_frames(sample_frames, pipe)
        return 13  # (49 - 1) // 4 + 1

    # ── QKV extraction ────────────────────────────────────────────────────────

    def extract_qkv(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project video hidden_states into Q, K, V.

        Uses the **same** ``to_q/to_k/to_v`` weights as the encoder path.
        """
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Reshape to [B, seq, H, D]
        inner_dim = query.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(*query.shape[:-1], attn.heads, head_dim)
        key = key.view(*key.shape[:-1], attn.heads, head_dim)
        value = value.view(*value.shape[:-1], attn.heads, head_dim)

        # Apply QK-norm (LayerNorm on head_dim)
        if hasattr(attn, "norm_q") and attn.norm_q is not None:
            query = attn.norm_q(query)
        if hasattr(attn, "norm_k") and attn.norm_k is not None:
            key = attn.norm_k(key)

        return query, key, value

    # ── Encoder projections (shared weights) ──────────────────────────────────

    def extract_encoder_qkv(
        self,
        attn: Any,
        encoder_hidden_states: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Project text tokens through the **same** to_q/to_k/to_v weights.

        CogVideoX uses shared projections for text and video (unlike
        HunyuanVideo which has separate add_q/k/v_proj).
        """
        if encoder_hidden_states is None:
            return None

        enc_q = attn.to_q(encoder_hidden_states)
        enc_k = attn.to_k(encoder_hidden_states)
        enc_v = attn.to_v(encoder_hidden_states)

        # Reshape to [B, text_seq, H, D]
        inner_dim = enc_q.shape[-1]
        head_dim = inner_dim // attn.heads
        enc_q = enc_q.view(*enc_q.shape[:-1], attn.heads, head_dim)
        enc_k = enc_k.view(*enc_k.shape[:-1], attn.heads, head_dim)
        enc_v = enc_v.view(*enc_v.shape[:-1], attn.heads, head_dim)

        # Apply same QK-norm
        if hasattr(attn, "norm_q") and attn.norm_q is not None:
            enc_q = attn.norm_q(enc_q)
        if hasattr(attn, "norm_k") and attn.norm_k is not None:
            enc_k = attn.norm_k(enc_k)

        return enc_q, enc_k, enc_v

    def split_encoder_for_cross_attn(
        self,
        attn: Any,
        encoder_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Joint attention: encoder handled via extract_encoder_qkv, not split
        return encoder_hidden_states, None

    def extract_cross_attn_kv(
        self,
        attn: Any,
        encoder_hidden_states: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        # Joint attention: no separate cross-attention path
        return None

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
        """Apply 3D RoPE to video Q/K tokens only.

        CogVideoX applies rotary embeddings only to video positions.
        ``image_rotary_emb`` covers the video sequence (not text).
        """
        if rotary_emb is None:
            return query, key

        # For multi-GPU CP, slice RoPE along the sequence dimension.
        if world > 1:
            full_seq = local_seq * world
            start = rank * local_seq
            end = start + local_seq
            if isinstance(rotary_emb, torch.Tensor):
                for dim in range(rotary_emb.dim()):
                    if rotary_emb.shape[dim] == full_seq:
                        idx = [slice(None)] * rotary_emb.dim()
                        idx[dim] = slice(start, end)
                        rotary_emb = rotary_emb[tuple(idx)].contiguous()
                        break
            elif isinstance(rotary_emb, (tuple, list)):
                from ..rope import slice_rotary_emb
                rotary_emb = slice_rotary_emb(rotary_emb, local_seq, rank, world)

        # Apply rotary embeddings.
        # Q/K are [B, seq, H, D], sequence_dim=1.
        try:
            from diffusers.models.embeddings import apply_rotary_emb as _apply_rope
            query = _apply_rope(query, rotary_emb, sequence_dim=1)
            key = _apply_rope(key, rotary_emb, sequence_dim=1)
        except ImportError:
            # Fallback to our own implementation
            if isinstance(rotary_emb, (tuple, list)) and len(rotary_emb) == 2:
                from ..rope import apply_rotary_emb
                query = apply_rotary_emb(query, rotary_emb[0], rotary_emb[1])
                key = apply_rotary_emb(key, rotary_emb[0], rotary_emb[1])

        return query, key

    # ── Output projection ─────────────────────────────────────────────────────

    def output_projection(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        query_dtype: torch.dtype,
    ) -> torch.Tensor:
        # Video output: flatten heads → to_out[0] (linear) + to_out[1] (dropout)
        hidden_states = hidden_states.flatten(2, 3).to(query_dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    def format_output(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_output: Optional[torch.Tensor],
        encoder_seq_len: int,
        query_dtype: torch.dtype,
    ) -> Any:
        """Project encoder output and return (video, encoder) tuple.

        CogVideoX blocks expect ``(attn_hidden_states, attn_encoder_hidden_states)``
        from the attention processor.  Uses the **same** ``to_out`` projection
        for both video and encoder (unlike HunyuanVideo which has ``to_add_out``).
        """
        if encoder_output is not None and encoder_seq_len > 0:
            # Project encoder output through same to_out weights
            encoder_out = encoder_output.flatten(2, 3).to(query_dtype)
            encoder_out = attn.to_out[0](encoder_out)
            encoder_out = attn.to_out[1](encoder_out)

            return hidden_states, encoder_out
        else:
            return hidden_states

    # ── Cross-attention dispatch ──────────────────────────────────────────────

    def cross_attention(
        self,
        attn: Any,
        query: torch.Tensor,
        encoder_image: torch.Tensor,
        attention_backend: Any,
    ) -> Optional[torch.Tensor]:
        # Joint attention: no separate cross-attention
        return None

    # ── Pipeline integration ──────────────────────────────────────────────────

    def install_processor(self, pipe: Any, processor: Any) -> int:
        blocks = pipe.transformer.transformer_blocks
        for block in blocks:
            block.attn1.processor = processor
        return len(blocks)

    def setup_context_parallel(self, transformer: Any, world: int) -> None:
        from diffusers import ContextParallelConfig
        from diffusers.models._modeling_parallel import (
            ContextParallelInput,
            ContextParallelOutput,
        )

        transformer._cp_plan = {
            "transformer_blocks.0": {
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
        try:
            from diffusers.models.transformers.cogvideox_transformer_3d import (
                CogVideoXAttnProcessor2_0,
            )

            _orig_call = CogVideoXAttnProcessor2_0.__call__

            def _patched_call(
                self_proc,
                attn,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                image_rotary_emb=None,
                **kwargs,
            ):
                if image_rotary_emb is not None:
                    local_seq = hidden_states.shape[1]
                    full_seq = local_seq * world
                    start = rank * local_seq

                    if isinstance(image_rotary_emb, (tuple, list)):
                        from ..rope import slice_rotary_emb
                        image_rotary_emb = slice_rotary_emb(
                            image_rotary_emb, local_seq, rank, world,
                        )
                    elif isinstance(image_rotary_emb, torch.Tensor):
                        for dim in range(image_rotary_emb.dim()):
                            if image_rotary_emb.shape[dim] == full_seq:
                                idx = [slice(None)] * image_rotary_emb.dim()
                                idx[dim] = slice(start, start + local_seq)
                                image_rotary_emb = image_rotary_emb[
                                    tuple(idx)
                                ].contiguous()
                                break

                return _orig_call(
                    self_proc,
                    attn,
                    hidden_states,
                    encoder_hidden_states,
                    attention_mask,
                    image_rotary_emb,
                    **kwargs,
                )

            CogVideoXAttnProcessor2_0.__call__ = _patched_call
        except (ImportError, AttributeError):
            pass
