"""
hunyuan_video.py — ModelAdapter implementation for HunyuanVideo-1.5.

**Status: experimental.**

HunyuanVideo-1.5 uses **dual-stream** transformer blocks where video and
encoder (text) tokens have separate Q/K/V projections, are concatenated
for joint attention, then split back.  The block's ``forward()`` expects
the attention processor to return a tuple ``(attn_output, context_attn_output)``.

Key differences from Wan:
  - Dual-stream attention with separate ``add_q/k/v_proj`` for encoder states
  - Separate output projections: ``to_out`` for video, ``to_add_out`` for encoder
  - 3D RoPE (time, height, width) applied only to video tokens, not encoder
  - RoPE passed as ``image_rotary_emb`` kwarg (not ``rotary_emb``)
  - QK-norm (RMSNorm) applied per-head after unflatten
  - No separate cross-attention path (unlike Wan I2V)

TODO:
  - Proper 3D RoPE slicing for context parallelism
  - End-to-end testing with multi-GPU
"""

from typing import Any, Optional, Tuple

import torch

from .base import ModelAdapter


class HunyuanVideoAdapter(ModelAdapter):
    """Adapter bridging HunyuanVideo-1.5 dual-stream attention to LVSA.

    .. warning::

        This adapter is experimental.  The 3D RoPE slicing for multi-GPU
        context parallelism requires further work.
    """

    # ── Geometry ──────────────────────────────────────────────────────────────

    def patches_per_frame(self, height: int, width: int, pipe: Any) -> int:
        patch_size_cfg = getattr(pipe.transformer.config, "patch_size", 2)
        if isinstance(patch_size_cfg, (list, tuple)):
            patch_size_s = int(patch_size_cfg[1])
        else:
            patch_size_s = int(patch_size_cfg)

        vae_s = getattr(pipe, "vae_scale_factor_spatial", 16)
        return (height // (vae_s * patch_size_s)) * (
            width // (vae_s * patch_size_s)
        )

    def latent_frames(self, num_frames: int, pipe: Any) -> int:
        vae_t = getattr(pipe, "vae_scale_factor_temporal", 4)
        return (num_frames - 1) // vae_t + 1

    def reference_latent_frames(self, pipe: Any) -> int:
        """HunyuanVideo-1.5 reference: 129 raw → 33 latent frames.

        Tries to compute from config (``sample_n_frames`` / VAE temporal),
        falls back to 33 latent frames if config is missing.
        """
        sample_frames = getattr(pipe.transformer.config, "sample_n_frames", None)
        if sample_frames is not None:
            return self.latent_frames(sample_frames, pipe)
        return 33  # (129 - 1) // 4 + 1

    # ── QKV extraction ────────────────────────────────────────────────────────

    def extract_qkv(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project video hidden_states into Q, K, V.

        Only projects the video (latent) stream.  Encoder tokens are handled
        separately by :meth:`extract_encoder_qkv` and concatenated by the
        LVSA processor.
        """
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Reshape to [B, seq, H, D] — BEFORE QK-norm,
        # because norm_q/norm_k (RMSNorm) operate on head_dim, not inner_dim.
        inner_dim = query.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(*query.shape[:-1], attn.heads, head_dim)
        key = key.view(*key.shape[:-1], attn.heads, head_dim)
        value = value.view(*value.shape[:-1], attn.heads, head_dim)

        # Apply QK-norm (operates on last dim = head_dim)
        if hasattr(attn, "norm_q") and attn.norm_q is not None:
            query = attn.norm_q(query)
        if hasattr(attn, "norm_k") and attn.norm_k is not None:
            key = attn.norm_k(key)

        return query, key, value

    # ── Encoder projections (dual-stream) ──────────────────────────────────────

    def extract_encoder_qkv(
        self,
        attn: Any,
        encoder_hidden_states: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Project encoder_hidden_states into Q, K, V via add_q/k/v_proj.

        HunyuanVideo-1.5 has dedicated projections for the encoder (text) stream.
        These are concatenated with video Q/K/V for joint attention, then the
        output is split back.
        """
        if not hasattr(attn, "add_q_proj") or attn.add_q_proj is None:
            return None

        enc_q = attn.add_q_proj(encoder_hidden_states)
        enc_k = attn.add_k_proj(encoder_hidden_states)
        enc_v = attn.add_v_proj(encoder_hidden_states)

        # Reshape to [B, text_seq, H, D]
        inner_dim = enc_q.shape[-1]
        head_dim = inner_dim // attn.heads
        enc_q = enc_q.view(*enc_q.shape[:-1], attn.heads, head_dim)
        enc_k = enc_k.view(*enc_k.shape[:-1], attn.heads, head_dim)
        enc_v = enc_v.view(*enc_v.shape[:-1], attn.heads, head_dim)

        # Apply encoder QK-norm if present
        if hasattr(attn, "norm_added_q") and attn.norm_added_q is not None:
            enc_q = attn.norm_added_q(enc_q)
        if hasattr(attn, "norm_added_k") and attn.norm_added_k is not None:
            enc_k = attn.norm_added_k(enc_k)

        return enc_q, enc_k, enc_v

    def split_encoder_for_cross_attn(
        self,
        attn: Any,
        encoder_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Dual-stream: encoder is handled via extract_encoder_qkv, not split
        return encoder_hidden_states, None

    def extract_cross_attn_kv(
        self,
        attn: Any,
        encoder_hidden_states: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        # Dual-stream: no separate cross-attention path
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
        """Apply RoPE to video Q/K tokens.

        In the dual-stream flow, Q/K here are video-only (encoder tokens are
        handled separately).  ``image_rotary_emb`` covers the video sequence.
        """
        if rotary_emb is None:
            return query, key

        # For multi-GPU CP, slice the RoPE along the sequence dimension.
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

        # Apply rotary embeddings using diffusers' apply_rotary_emb which
        # handles HunyuanVideo's (freqs_cos, freqs_sin) tuple natively.
        # Q/K are [B, seq, H, D], so sequence_dim=1.
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
        # Video output projection: to_out[0] (linear) + to_out[1] (dropout)
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

        HunyuanVideo blocks expect ``(attn_output, context_attn_output)`` from
        the attention processor.
        """
        if encoder_output is not None and encoder_seq_len > 0:
            # Project encoder attention output: flatten heads then linear
            encoder_out = encoder_output.flatten(2, 3).to(query_dtype)
            if hasattr(attn, "to_add_out") and attn.to_add_out is not None:
                encoder_out = attn.to_add_out(encoder_out)

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
        # Dual-stream: no separate cross-attention
        return None

    # ── Pipeline integration ──────────────────────────────────────────────────

    def install_processor(self, pipe: Any, processor: Any) -> int:
        blocks = pipe.transformer.transformer_blocks
        for block in blocks:
            block.attn.processor = processor
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
            import diffusers.models.transformers.transformer_hunyuan_video15 as _hv_module

            _orig_call = _hv_module.HunyuanVideo15AttnProcessor2_0.__call__

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

            _hv_module.HunyuanVideo15AttnProcessor2_0.__call__ = _patched_call
        except (ImportError, AttributeError):
            pass
