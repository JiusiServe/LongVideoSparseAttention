"""Monkey-patch WanSelfAttention to use LVSA at the block level.

Wan's transformer uses vllm-omni's ``_sp_plan`` to pre-shard the sequence at
``blocks.0`` input. The hook runs on the (possibly sharded) local sequence.
For Ring SP (ring_world > 1), per-rank LVSA is not applied in this release;
the hook only takes effect when the full sequence is present on the rank.

No dual-stream split is needed — cross-attention is a separate
``WanCrossAttention`` module.

Usage: call ``install_wan_lvsa_hook()`` after model loading in the worker
process. Triggered automatically when ``LVSA_WAN_HOOK=1``.
"""

import os
from typing import Any, Optional

import torch

from lvsa.sparse_attention import lvsa_sdpa

from .config import LVSAConfig
from ._fallback import warn_fallback


# Reuse HunyuanLVSAState — it is model-agnostic (step tracking, metadata cache).
from .hunyuan_hook import HunyuanLVSAState, _mask_log_should_fire, _log_engagement_once
from .global_kv import build_global_kv


def install_wan_lvsa_hook(total_latent_frames: int) -> None:
    """Monkey-patch WanSelfAttention.forward to use LVSA.

    Must be called in the worker process after model loading.
    """
    from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import (
        WanSelfAttention,
    )
    # vllm-omni 0.22 removed the free function ``apply_rotary_emb_wan``; RoPE is
    # now applied through ``RotaryEmbeddingWan``. It is stateless (no params /
    # buffers, cos & sin are passed in), so we build it once and reuse it.
    from vllm_omni.diffusion.layers.rope import RotaryEmbeddingWan

    config = LVSAConfig.from_env()
    state = HunyuanLVSAState(config)
    wan_rope = RotaryEmbeddingWan(is_neox_style=False, half_head_dim=True)

    _orig_forward = WanSelfAttention.forward

    def _lvsa_forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[tuple] = None,
        attn_metadata: Optional[Any] = None,
    ) -> torch.Tensor:
        """LVSA-enhanced forward for Wan self-attention.

        Signature mirrors vllm-omni 0.22 ``WanSelfAttention.forward``: the
        third argument is now an ``AttentionMetadata`` (previously a raw
        ``attn_mask`` tensor in 0.18).
        """

        local_seq = hidden_states.shape[1]
        full_seq = local_seq  # Single-rank assumption (no Ring SP in this release).

        # ── Step tracking ──
        # Ticked on every forward (including fallbacks) so the per-module step
        # cadence is identical to before this early-return guard was added.
        step_idx = state.tick(id(self), local_seq)

        # ── Distributed CP guard ──
        # Under Ulysses / Ring SP, the sequence dimension is sharded across
        # ranks before this forward runs. ``local_seq`` no longer equals the
        # full T_lat × P, so the LVSA geometry detection would silently match
        # the wrong P. Fall back to dense — a per-rank LVSA pattern is not
        # supported in this release.
        try:
            import torch.distributed as _dist
            _is_distributed = _dist.is_initialized() and _dist.get_world_size() > 1
        except Exception:
            _is_distributed = False

        # ── Geometry check BEFORE any projection ──
        # Doing this first avoids wasting QKV / QK-norm / RoPE compute on warmup
        # (small seq) or distributed runs that are guaranteed to fall back.
        # full_seq must equal T_lat * P (no encoder tokens in Wan self-attn).
        geometry_ok = (
            total_latent_frames > 0
            and full_seq % total_latent_frames == 0
            and not _is_distributed
        )

        if not geometry_ok:
            warn_fallback(
                origin="wan_hook",
                reason="distributed_cp" if _is_distributed else "geometry_mismatch",
                seq_len=local_seq,
                extra={"T_lat": total_latent_frames, "full_seq": full_seq,
                       "distributed": _is_distributed},
            )
            # Delegate to the original (dense) forward rather than reimplement
            # the attention call — keeps us correct against vllm-omni's evolving
            # signature (it builds its own AttentionMetadata from the padding).
            return _orig_forward(self, hidden_states, rotary_emb, attn_metadata)

        P = full_seq // total_latent_frames

        # ── QKV projection on the (possibly sharded) local seq ──
        qkv, _ = self.to_qkv(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # QK norm (on flat last dim before unflatten)
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Reshape for multi-head: [B, local_seq, H, D]
        query = query.unflatten(2, (self.num_heads, self.head_dim))
        key = key.unflatten(2, (self.num_kv_heads, self.head_dim))
        value = value.unflatten(2, (self.num_kv_heads, self.head_dim))

        # Apply Wan-specific RoPE
        if rotary_emb is not None:
            freqs_cos, freqs_sin = rotary_emb
            query = wan_rope(query, freqs_cos, freqs_sin)
            key = wan_rope(key, freqs_cos, freqs_sin)

        metadata = state.get_metadata(
            total_latent_frames, P, step_idx, query.device,
        )
        _log_engagement_once(state, "wan", total_latent_frames, P, full_seq, metadata)

        # Opt-in compact attention-mask log (LVSA_MASK_LOG env).
        mask_spec = os.environ.get("LVSA_MASK_LOG", "")
        if _mask_log_should_fire(mask_spec, step_idx, state._mask_log_last_step):
            from lvsa.sparse_attention import print_attention_mask_compact
            state._mask_log_last_step = step_idx
            print(
                f"[LVSA-MASK] step={step_idx}  T_lat={total_latent_frames}  "
                f"W={metadata.window_size}  |G|={len(metadata.global_set)}  "
                f"kfi={metadata.key_frame_interval}",
                flush=True,
            )
            print_attention_mask_compact(
                total_frames=total_latent_frames,
                window_size=metadata.window_size,
                global_set=metadata.global_set,
                expand_window=metadata.expand_window,
            )

        k_global, v_global = build_global_kv(
            key, value, metadata.global_indices, P,
        )
        out_local = lvsa_sdpa(
            query, key, value, k_global, v_global, metadata,
        )

        out_local = out_local.flatten(2, 3).type_as(query)
        out_local = self.to_out(out_local)
        out_local = self.dropout(out_local)
        return out_local

    # Apply the monkey-patch
    WanSelfAttention.forward = _lvsa_forward

    # Let the attention backend know we're handling LVSA at the hook level
    try:
        from .attention_impl import LVSAAttentionImpl
        LVSAAttentionImpl._hook_active = True
    except Exception:
        pass

    print(
        f"[LVSA-hook] Installed LVSA hook on WanSelfAttention "
        f"(T_lat={total_latent_frames}, "
        f"sparsity_scale={config.sparsity_scale})",
        flush=True,
    )
