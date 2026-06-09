"""Monkey-patch Cosmos3CrossAttention to use LVSA at the block level.

NVIDIA Cosmos 3.0 (vllm-omni main) generates video with a Mixture-of-Transformers:
an autoregressive understanding (``und``) tower and a diffusion generation (``gen``)
tower. The video tokens live in the *gen* stream, which uses
``Cosmos3CrossAttention``: the gen queries attend to ``cat([k_und, k_gen])`` —
i.e. the gen video tokens form a clean ``T_lat x patches_per_frame`` frame grid
(unlike Helios, the geometry guard PASSES here) and the understanding/text tokens
are a *separate* pre-computed ``k_und``/``v_und`` (NOT prepended into the gen
sequence). This maps 1:1 onto the dual-stream pattern used by ``hunyuan_hook``:
window the ``gen<->gen`` block, append ``k_und``/``v_und`` to the always-attended
global K/V.

Two structural differences from ``hunyuan_hook``:

1. **Signature.** ``Cosmos3CrossAttention.forward(hidden_states, k_und, v_und,
   freqs_cos, freqs_sin)`` — the understanding K/V arrive pre-computed (already
   QK-normed, RoPE'd and TP-sharded), and RoPE for the gen stream is applied
   inline via ``_apply_rotary_pos_emb`` rather than passed as ``image_rotary_emb``.

2. **GQA.** Cosmos 3 uses grouped-query attention (``num_kv_heads < num_heads``).
   We keep K/V at ``num_kv_heads_local`` and let the backends broadcast heads
   natively: FlashInfer plans ``num_kv_heads=Hkv`` and ``lvsa_sdpa`` passes
   ``enable_gqa`` (its docstring requires callers *not* to repeat-KV) — 4x less
   KV traffic/VRAM, matching the dense path. (Ablation: ``LVSA_REPEAT_KV=1``
   forces the legacy ``repeat_interleave`` path.)

Sequence-parallel (Ulysses) shards the gen grid so the per-rank sequence is no
longer the frame grid — we gate on ``not _is_sp_active()`` and fall back to the
original dense forward under SP, mirroring ``wan_hook``'s CP guard.

Usage: call ``install_cosmos3_lvsa_hook(total_latent_frames=...)`` after model
loading in the worker process. Triggered automatically by
``register_lvsa_backend()`` when ``LVSA_COSMOS3_HOOK=1``.
"""

import os

import torch
import torch.nn.functional as F

from lvsa.sparse_attention import lvsa_sdpa, print_attention_mask_compact

from ._fallback import warn_fallback
from .config import LVSAConfig
from .global_kv import build_global_kv
# The step counter, metadata cache and engagement/mask logging are
# model-agnostic — reuse them rather than duplicating (as wan_hook does).
from .hunyuan_hook import (
    HunyuanLVSAState,
    _log_engagement_once,
    _mask_log_should_fire,
)


def install_cosmos3_lvsa_hook(total_latent_frames: int) -> None:
    """Monkey-patch ``Cosmos3CrossAttention.forward`` to use LVSA.

    Must be called in the worker process after the model class is importable
    (before instantiation). ``total_latent_frames`` is ``(num_frames - 1) //
    vae_temporal_factor + 1`` for the requested horizon.
    """
    from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import (
        Cosmos3CrossAttention,
        _apply_rotary_pos_emb,
        _is_sp_active,
    )

    config = LVSAConfig.from_env()
    state = HunyuanLVSAState(config)

    # Optional per-tile attention-kernel override for lvsa_sdpa's dispatch.
    # Cosmos3 dense uses a fused FlashAttention kernel; the LVSA per-frame loop
    # otherwise defaults to diffusers' NATIVE (generic SDPA) per tile. Forcing
    # e.g. ``_native_flash`` makes each tile use the flash kernel — matching the
    # dense baseline's kernel family. ``None`` keeps diffusers' default.
    dispatch_backend = os.environ.get("LVSA_DISPATCH_BACKEND") or None
    if dispatch_backend:
        print(f"[LVSA-hook] per-tile dispatch backend = {dispatch_backend}", flush=True)

    # Ablation knob: force repeat-KV (legacy) instead of native GQA. Lets the
    # e2e study compare repeat-KV vs native-GQA variants of each backend.
    repeat_kv = os.environ.get("LVSA_REPEAT_KV", "").lower() in ("1", "true", "yes")
    if repeat_kv:
        print("[LVSA-hook] LVSA_REPEAT_KV=1 -> repeat-KV (no native GQA)", flush=True)

    # Optional fused FlashInfer block-sparse backend (LVSA_BACKEND=flashinfer).
    # One fused kernel for the whole gen+und sparse pattern, vs lvsa_sdpa's
    # per-frame loop — removes the per-tile launch/cat overhead.
    fi_runner = None
    if (config.backend or "").lower() == "flashinfer":
        from .flashinfer_runner import (
            FlashInferDualStreamRunner,
            FLASHINFER_AVAILABLE,
        )
        if FLASHINFER_AVAILABLE:
            fi_runner = FlashInferDualStreamRunner()
            print("[LVSA-hook] backend = flashinfer (fused block-sparse)", flush=True)
        else:
            print("[LVSA-hook] LVSA_BACKEND=flashinfer requested but flashinfer "
                  "is unavailable -> using SDPA per-tile loop", flush=True)

    # Save original forward (used for every dense fallback path).
    _orig_forward = Cosmos3CrossAttention.forward

    def _lvsa_forward(
        self,
        hidden_states: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        """LVSA-enhanced forward: sparse on gen video, ``und`` K/V always-global.

        Signature mirrors vllm-omni main ``Cosmos3CrossAttention.forward``:
        ``k_und``/``v_und`` are pre-computed UND keys/values ``[B, S_und,
        H_kv_local, D]``; ``freqs_cos``/``freqs_sin`` are the interleaved 3D
        mRoPE tables for the gen stream.
        """
        B, S_gen, _ = hidden_states.shape

        # ── Step tracking ──
        step_idx = state.tick(id(self), S_gen)

        # ── Sequence-parallel guard ──
        # Under Ulysses SP, ``S_gen`` is the per-rank shard of the gen grid, not
        # the full T_lat x P. Geometry detection would silently corrupt the
        # attention pattern. Delegate to the (SP-aware, joint_*-based) original.
        if _is_sp_active():
            warn_fallback(
                origin="cosmos3_hook",
                reason="sequence_parallel",
                seq_len=S_gen,
                extra={"step": step_idx},
            )
            return _orig_forward(self, hidden_states, k_und, v_und, freqs_cos, freqs_sin)

        # ── Geometry guard ──
        # The gen stream must be a clean T_lat x P frame grid for the sparse
        # pattern to be valid. Fall back to dense otherwise (e.g. warmup).
        if total_latent_frames <= 0 or S_gen % total_latent_frames != 0:
            warn_fallback(
                origin="cosmos3_hook",
                reason="geometry_mismatch",
                seq_len=S_gen,
                extra={"step": step_idx, "T_lat": total_latent_frames},
            )
            return _orig_forward(self, hidden_states, k_und, v_und, freqs_cos, freqs_sin)
        P = S_gen // total_latent_frames

        # ── gen QKV + QK-norm + RoPE (mirror of Cosmos3CrossAttention.forward) ──
        q = self.to_q(hidden_states).view(B, S_gen, self.num_heads_local, self.head_dim)
        k = self.to_k(hidden_states).view(B, S_gen, self.num_kv_heads_local, self.head_dim)
        v = self.to_v(hidden_states).view(B, S_gen, self.num_kv_heads_local, self.head_dim)

        q = F.rms_norm(q, (q.shape[-1],), self.norm_q.weight, self.norm_q.variance_epsilon)
        k = F.rms_norm(k, (k.shape[-1],), self.norm_k.weight, self.norm_k.variance_epsilon)

        q, k = _apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)

        metadata = state.get_metadata(total_latent_frames, P, step_idx, q.device)
        _log_engagement_once(state, "cosmos3", total_latent_frames, P, S_gen, metadata)

        # Opt-in compact attention-mask log (LVSA_MASK_LOG), deduped per step.
        mask_spec = os.environ.get("LVSA_MASK_LOG", "")
        if _mask_log_should_fire(mask_spec, step_idx, state._mask_log_last_step):
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

        # ── GQA (native on both backends) ──
        # FlashInfer plans num_kv_heads=Hkv; lvsa_sdpa passes enable_gqa. So K/V
        # stay at ``num_kv_heads_local`` — 4x less KV traffic + VRAM, matching the
        # dense path's native GQA. No repeat-KV. (No-op if Hkv == num_heads.)
        # Ablation: LVSA_REPEAT_KV=1 forces the legacy repeat-KV path.
        if repeat_kv:
            n_rep = self.num_heads_local // self.num_kv_heads_local
            if n_rep > 1:
                k = k.repeat_interleave(n_rep, dim=2)
                v = v.repeat_interleave(n_rep, dim=2)
                k_und = k_und.repeat_interleave(n_rep, dim=2)
                v_und = v_und.repeat_interleave(n_rep, dim=2)

        # ── Build global K/V from gen keyframes + append und as always-global ──
        k_global, v_global = build_global_kv(k, v, metadata.global_indices, P)
        k_global = torch.cat([k_global, k_und], dim=1)
        v_global = torch.cat([v_global, v_und], dim=1)

        # ── LVSA on gen queries ──
        if fi_runner is not None:
            out = fi_runner.run(q, k, v, k_global, v_global, metadata)  # fused CSR
        else:
            out = lvsa_sdpa(q, k, v, k_global, v_global, metadata,
                            attention_backend=dispatch_backend)  # per-tile loop
        out = out.reshape(B, S_gen, -1)
        return self.to_out(out)

    # Apply the monkey-patch.
    Cosmos3CrossAttention.forward = _lvsa_forward

    # Tell the attention impl to just use dense (the hook handles LVSA).
    from .attention_impl import LVSAAttentionImpl
    LVSAAttentionImpl._hook_active = True

    print(
        f"[LVSA-hook] Installed LVSA hook on Cosmos3CrossAttention "
        f"(T_lat={total_latent_frames})"
    )
