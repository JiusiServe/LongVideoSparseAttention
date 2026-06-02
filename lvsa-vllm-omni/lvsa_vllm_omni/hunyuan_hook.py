"""Monkey-patch HunyuanVideo15Attention to use LVSA at the block level.

Instead of hooking into the attention backend (which receives already-concatenated
video+encoder Q/K/V), we hook into the attention module's forward where video and
encoder streams are still separate. This is equivalent to what the standalone
DistributedLVSAProcessor does.

Usage: call ``install_hunyuan_lvsa_hook()`` after model loading in the worker process.
Triggered automatically by ``register_lvsa_backend()`` when ``LVSA_HUNYUAN_HOOK=1``.
"""

import os
from typing import Any, Optional

import torch
import torch.nn.functional as F

from lvsa.sparse_attention import (
    LVSAMetadata,
    compute_auto_kfi,
    print_attention_mask_compact,
    lvsa_sdpa,
)

from .config import LVSAConfig
from .global_kv import build_global_kv


def _mask_log_should_fire(spec: str, step_idx: int, last_step: int) -> bool:
    """LVSA_MASK_LOG step selector. Supports: '1' (every step), 'once',
    'N' (step N), 'N-M' (range), 'N,M,K' (specific steps). Returns False if
    spec is empty or '0', or if we already printed at this step."""
    if not spec or spec == "0" or step_idx == last_step:
        return False
    if spec == "1":
        return True
    if spec == "once":
        return last_step == -1
    if "-" in spec:
        try:
            lo, hi = spec.split("-", 1)
            return int(lo) <= step_idx <= int(hi)
        except ValueError:
            return False
    try:
        return step_idx in {int(x.strip()) for x in spec.split(",") if x.strip()}
    except ValueError:
        return False


def _log_engagement_once(state, model, total_latent_frames, num_patches, seq_len, metadata):
    """Print a one-time positive confirmation that LVSA actually engaged for
    generation (geometry matched, not a warmup fallback).

    Default-on, deduped per generation via ``state._engaged_logged`` (reset on
    seq_len change in ``tick``). Distinguishes engaged-and-sparse (kfi>1, above
    reference) from engaged-but-dense (kfi==1, at/below reference) — neither of
    which was visible before, since the LVSA path only logged the opt-in
    ``LVSA_MASK_LOG``. Silence here, with a ``[LVSA-FALLBACK]`` only at the
    warmup seq_len, now means "engaged for every generation step"."""
    if state._engaged_logged:
        return
    state._engaged_logged = True
    kfi = metadata.key_frame_interval
    mode = "SPARSE" if kfi > 1 else "DENSE (T_lat <= ref)"
    print(
        f"[LVSA] engaged ({model}): T_lat={total_latent_frames} P={num_patches} "
        f"seq_len={seq_len} kfi={kfi} W={metadata.window_size} "
        f"|G|={len(metadata.global_set)} -> {mode}",
        flush=True,
    )


class HunyuanLVSAState:
    """Shared LVSA state across all hooked attention blocks.

    Concurrency: state is intentionally shared across every block within a
    generation — that's how step counting + metadata caching work. We do not
    wrap it in ``threading.local()`` because every vllm-omni diffusion worker
    is a single process running a single asyncio loop on one OS thread; the
    warmup and every request hit this state sequentially. Thread-local
    storage would also not help if vllm-omni later adds in-worker batching
    (coroutines share the same OS thread) — that case would need a
    request-scoped key.
    """

    def __init__(self, config: LVSAConfig) -> None:
        self.config = config
        self._metadata: Optional[LVSAMetadata] = None
        self._cached_total_frames: Optional[int] = None
        self._cached_patches: Optional[int] = None
        self._cached_step: int = -1
        self._call_count: int = 0
        self._n_blocks: Optional[int] = None
        # Forward passes per denoising step. CFG runs cond + uncond = 2.
        # Without CFG (guidance_scale==1) set LVSA_CFG_PASSES=1.
        self._cfg_passes: int = max(1, int(os.environ.get("LVSA_CFG_PASSES", 2)))
        self._step: int = 0
        self._seen_ids: set = set()
        self._generation_seq_len: Optional[int] = None
        self._last_step_time: Optional[float] = None
        self._mask_log_last_step: int = -1
        # One-time positive engagement log per generation (reset on seq_len change).
        self._engaged_logged: bool = False

    def tick(self, layer_id: int, seq_len: int) -> int:
        """Track denoising step by counting self-attention calls.
        One denoising step = ``n_blocks * cfg_passes`` attention forwards.
        """
        if self._generation_seq_len is not None and seq_len != self._generation_seq_len:
            self._generation_seq_len = seq_len
            self._call_count = 0
            self._step = 0
            self._seen_ids.clear()
            self._engaged_logged = False

        if self._generation_seq_len is None:
            self._generation_seq_len = seq_len

        self._call_count += 1

        # Auto-calibrate n_blocks on first repeated layer_id.
        step_boundary = False
        if self._n_blocks is None:
            if layer_id in self._seen_ids:
                self._n_blocks = len(self._seen_ids)
                self._seen_ids.clear()
                print(
                    f"[LVSA-hook] Step counter calibrated: "
                    f"n_blocks={self._n_blocks} cfg_passes={self._cfg_passes}"
                )
            else:
                self._seen_ids.add(layer_id)

        # Compute step from total call count.
        if self._n_blocks is not None:
            threshold = self._n_blocks * self._cfg_passes
            new_step = (self._call_count - 1) // threshold
            if new_step > self._step:
                self._step = new_step
                step_boundary = True

        # Opt-in per-step memory logging — diagnose cross-step growth.
        # Device-agnostic: works on CUDA and Ascend NPU.
        if step_boundary and os.environ.get("LVSA_MEM_LOG", "0") == "1":
            from lvsa.device import memory_stats
            stats = memory_stats()
            if stats is not None:
                kind, dev, alloc, reserved, peak = stats
                print(
                    f"[LVSA-MEM] step={self._step} {kind}={dev} "
                    f"alloc={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB",
                    flush=True,
                )

        # Opt-in per-step wall-clock timing — diagnose mid-run slowdowns.
        # Logs the time spent on the step that JUST completed (self._step - 1).
        if step_boundary and os.environ.get("LVSA_STEP_TIME_LOG", "0") == "1":
            import time as _time
            now = _time.perf_counter()
            if self._last_step_time is not None:
                dt = now - self._last_step_time
                print(
                    f"[LVSA-TIME] step={self._step - 1} dt={dt:.3f}s",
                    flush=True,
                )
            self._last_step_time = now

        return self._step

    def get_metadata(
        self, total_latent_frames: int, num_patches: int, step_idx: int, device: torch.device,
    ) -> LVSAMetadata:
        """Get or rebuild LVSAMetadata."""
        cfg = self.config
        needs_rebuild = (
            self._metadata is None
            or self._cached_total_frames != total_latent_frames
            or self._cached_patches != num_patches
            or (cfg.rotate_keyframes and self._cached_step != step_idx)
        )

        if needs_rebuild:
            W = cfg.latent_window_size
            n_first = cfg.latent_n_first_frames
            kfi = cfg.latent_key_frame_interval

            if cfg.auto_keyframes:
                kfi = compute_auto_kfi(
                    total_latent_frames, W, n_first,
                    reference_frames=cfg.reference_latent_frames,
                    sparsity_scale=cfg.sparsity_scale,
                )

            offset = 0
            if cfg.rotate_keyframes and kfi > 0:
                offset = step_idx % kfi

            self._metadata = LVSAMetadata.build(
                total_latent_frames=total_latent_frames,
                num_patches=num_patches,
                window_size=W,
                n_first_frames=n_first,
                key_frame_interval=kfi,
                rank=0,
                world=1,
                expand_window=True,
                keyframe_offset=offset,
                reference_frames=cfg.reference_latent_frames,
                sparsity_scale=cfg.sparsity_scale,
            )
            self._metadata.ensure_device(device)
            self._cached_total_frames = total_latent_frames
            self._cached_patches = num_patches
            self._cached_step = step_idx

        return self._metadata


def install_hunyuan_lvsa_hook(total_latent_frames: int) -> None:
    """Monkey-patch HunyuanVideo15Attention.forward to use LVSA.

    Must be called in the worker process after model loading.
    """
    from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import (
        HunyuanVideo15Attention,
    )

    config = LVSAConfig.from_env()
    state = HunyuanLVSAState(config)

    # Save original forward
    _orig_forward = HunyuanVideo15Attention.forward

    def _lvsa_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple | None = None,
        hidden_states_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """LVSA-enhanced forward: sparse attention on video, dense on encoder.

        Signature mirrors vllm-omni 0.22 ``HunyuanVideo15Attention.forward``,
        which added the ``hidden_states_mask`` argument (sequence-parallel
        padding mask) and may return a single tensor when there is no encoder
        stream.
        """

        # ── Video QKV (same as original) ──
        qkv, _ = self.to_qkv(hidden_states)
        q_size = self.to_qkv.num_heads * self.head_dim
        kv_size = self.to_qkv.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = query.unflatten(-1, (self.to_qkv.num_heads, -1))
        key = key.unflatten(-1, (self.to_qkv.num_kv_heads, -1))
        value = value.unflatten(-1, (self.to_qkv.num_kv_heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            cos = cos.to(query.dtype)
            sin = sin.to(query.dtype)
            query = self.rope(query, cos, sin)
            key = self.rope(key, cos, sin)

        # ── Encoder QKV ──
        enc_query = enc_key = enc_value = None
        if encoder_hidden_states is not None:
            encoder_qkv, _ = self.add_kv_proj(encoder_hidden_states)
            add_q_size = self.add_kv_proj.num_heads * self.head_dim
            add_kv_size = self.add_kv_proj.num_kv_heads * self.head_dim
            encoder_query, encoder_key, encoder_value = encoder_qkv.split(
                [add_q_size, add_kv_size, add_kv_size], dim=-1
            )
            encoder_query = encoder_query.unflatten(-1, (self.add_kv_proj.num_heads, -1))
            encoder_key = encoder_key.unflatten(-1, (self.add_kv_proj.num_kv_heads, -1))
            encoder_value = encoder_value.unflatten(-1, (self.add_kv_proj.num_kv_heads, -1))

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

        # ── Step tracking ──
        step_idx = state.tick(id(self), query.shape[1])

        # Warn if encoder is unexpectedly missing
        if encoder_hidden_states is None:
            from ._fallback import warn_fallback
            warn_fallback(
                origin="hunyuan_hook",
                reason="no_encoder",
                seq_len=query.shape[1],
                extra={"step": step_idx},
            )

        # ── Distributed CP guard ──
        # Under Ulysses / Ring SP, ``video_seq`` is the per-rank shard, not the
        # full T_lat × P. Geometry detection would silently corrupt the
        # attention pattern. Fall back to dense in that case.
        try:
            import torch.distributed as _dist
            _is_distributed = _dist.is_initialized() and _dist.get_world_size() > 1
        except Exception:
            _is_distributed = False
        if _is_distributed and encoder_hidden_states is not None:
            from ._fallback import warn_fallback
            warn_fallback(
                origin="hunyuan_hook",
                reason="distributed_cp",
                seq_len=query.shape[1],
                extra={"step": step_idx,
                       "world_size": _dist.get_world_size()},
            )

        if encoder_hidden_states is not None and not _is_distributed:
            # ── LVSA path: sparse on video, dense on encoder ──
            B, video_seq, H, D = query.shape
            P = video_seq // total_latent_frames

            metadata = state.get_metadata(total_latent_frames, P, step_idx, query.device)
            _log_engagement_once(state, "hunyuan", total_latent_frames, P, video_seq, metadata)

            # Opt-in compact attention-mask log (LVSA_MASK_LOG env). Dedups
            # across attention blocks via state._mask_log_last_step so we
            # print once per step boundary, not once per block.
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

            # Build global K/V from video + append encoder K/V
            k_global, v_global = build_global_kv(key, value, metadata.global_indices, P)
            k_global = torch.cat([k_global, encoder_key], dim=1)
            v_global = torch.cat([v_global, encoder_value], dim=1)

            # LVSA on video queries
            video_output = lvsa_sdpa(query, key, value, k_global, v_global, metadata)

            # Dense attention for encoder queries (attend to all video + encoder)
            full_k = torch.cat([key, encoder_key], dim=1)
            full_v = torch.cat([value, encoder_value], dim=1)
            eq = encoder_query.transpose(1, 2)
            ek = full_k.transpose(1, 2)
            ev = full_v.transpose(1, 2)
            encoder_output = F.scaled_dot_product_attention(
                eq, ek, ev, dropout_p=0.0, is_causal=False,
            ).transpose(1, 2)

            hidden_states = video_output
            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.to(query.dtype)
            hidden_states = self.to_out[0](hidden_states)

            encoder_hidden_states = encoder_output
            encoder_hidden_states = encoder_hidden_states.flatten(2, 3)
            encoder_hidden_states = encoder_hidden_states.to(query.dtype)
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            # ── Dense fallback: delegate to the original forward ──
            # Reached on warmup (geometry mismatch), when the encoder stream is
            # absent, or under distributed SP. We re-run _orig_forward rather
            # than reimplement the (now SP-aware) dense path: the QKV / RoPE
            # work above becomes wasted, but this path is rare and delegating
            # keeps us correct against vllm-omni 0.22's signature — including
            # hidden_states_mask handling and the joint-attention metadata.
            return _orig_forward(
                self,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                image_rotary_emb,
                hidden_states_mask,
            )

    # Apply the monkey-patch
    HunyuanVideo15Attention.forward = _lvsa_forward

    # Tell the attention impl to just use dense (hook handles LVSA)
    from .attention_impl import LVSAAttentionImpl
    LVSAAttentionImpl._hook_active = True

    print(f"[LVSA-hook] Installed LVSA hook on HunyuanVideo15Attention "
          f"(T_lat={total_latent_frames})")
