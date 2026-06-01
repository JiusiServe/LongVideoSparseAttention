"""
lvsa_processor.py — Block-sparse Sliding Window Attention processor for context-parallel inference.

Model-agnostic: all model-specific logic (QKV projections, RoPE format,
output projection, cross-attention) is delegated to a ``ModelAdapter``
instance provided at construction time.

The stateless attention math lives in ``sparse_attention.py``.  This module
provides the stateful wrapper that manages communication (all-reduce),
buffer allocation, KV caching, and adapter interaction.
"""

from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist

from .sparse_attention import (
    LVSAMetadata,
    adaptive_window_bounds,
    compute_auto_kfi,
    compute_boundary_guard_frames,
    compute_global_indices,
    expanded_window_bounds,
    get_window_bounds,
    lvsa_flashinfer,
    lvsa_sdpa,
)

try:
    import flashinfer
    _FLASHINFER_AVAILABLE = True
except ImportError:
    _FLASHINFER_AVAILABLE = False

from .adapters.base import ModelAdapter


def _expand_chunk_copies_to_indices(
    copies: list[tuple[int, int]],
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand a list of (src_start, dst_start) chunk-offset tuples into
    flat per-token src/dst index tensors.

    Each tuple describes a contiguous copy of ``chunk_size`` tokens from
    ``src_start`` to ``dst_start``. The result tensors index each token
    individually, suitable for ``buf[..., dst_idx] = src_buf[..., src_idx]``
    vectorized advanced-indexing.

    Returns
    -------
    src_idx, dst_idx : long tensors of shape ``[len(copies) * chunk_size]``
        on ``device``. Empty tensors (shape ``[0]``) if ``copies`` is empty.
    """
    if not copies:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty.clone()
    src_starts = torch.tensor([s for s, _ in copies], dtype=torch.long)
    dst_starts = torch.tensor([d for _, d in copies], dtype=torch.long)
    offsets = torch.arange(chunk_size, dtype=torch.long)
    src = (src_starts[:, None] + offsets[None, :]).reshape(-1).to(device)
    dst = (dst_starts[:, None] + offsets[None, :]).reshape(-1).to(device)
    return src, dst


class DistributedLVSAProcessor:
    """
    Block-sparse attention processor combining sliding window attention (LVSA),
    automatic keyframes, and multi-GPU context parallelism.

    **Model-agnostic**: all model-specific logic is delegated to a
    :class:`~lvsa.adapters.base.ModelAdapter` instance.

    Attention pattern (global frame space)
    ----------------------------------------
    Each query token at global position i belongs to frame (i // P).
    That token attends to:
      1. Global frames   : {0 … n_first_frames-1}  ∪  {0, kfi, 2*kfi, …}
         K/V gathered via all_reduce(SUM) in global_broadcast mode.
      2. Local window    : frames in [f-W, f+W] that have tokens on THIS rank
         and are not already in the global set.

    Why token-based instead of frame-based
    ----------------------------------------
    CP splits the sequence as seq_len // world tokens per rank.
    Typical video models produce latent frame counts that are not divisible
    by common world_size values (e.g. Wan's 4k+1 grid gives 21, 41, 61…).
    Reshaping to [B, local_frames, P, H, D] would therefore be invalid.

    We instead work purely in token space: for each local token i its global
    frame is  (global_token_start + i) // P,  where
    global_token_start = rank * local_seq.  This is correct for any T_lat
    and any world_size with zero constraints on their relationship.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(
        self,
        total_num_latent_frames: int,
        num_patches: int,
        window_size: int,
        n_first_frames: int,
        key_frame_interval: Optional[int],
        rank: int,
        world: int,
        expand_window: bool = True,
        adapter: Optional[ModelAdapter] = None,
        sparsity_scale: float = 1.0,
        reference_frames: int = 21,
    ) -> None:
        self.total_num_frames = total_num_latent_frames
        self.num_patches = num_patches
        self.window_size = window_size
        self.n_first_frames = n_first_frames
        self.rank = rank
        self.world = world
        self._expand_window = expand_window
        self._sparsity_scale = sparsity_scale
        self._reference_frames = reference_frames

        # Model adapter — if None, a WanAdapter is created lazily for
        # backwards compatibility.
        if adapter is None:
            from .adapters.wan import WanAdapter
            adapter = WanAdapter()
        self.adapter: ModelAdapter = adapter

        # ── Boundary guards (fixed across rotations) ──
        self._boundary_guards = compute_boundary_guard_frames(
            total_num_latent_frames,
            total_num_latent_frames * num_patches // world,
            num_patches, world, window_size,
        )
        self._current_offset = 0

        # ── Build all pattern metadata via LVSAMetadata ──
        self._metadata = LVSAMetadata.build(
            total_latent_frames=total_num_latent_frames,
            num_patches=num_patches,
            window_size=window_size,
            n_first_frames=n_first_frames,
            key_frame_interval=key_frame_interval,
            rank=rank,
            world=world,
            expand_window=expand_window,
            keyframe_offset=0,
            boundary_guards=self._boundary_guards,
            reference_frames=reference_frames,
            sparsity_scale=sparsity_scale,
        )
        self._copy_metadata_to_self()

        # ── Cached buffers (reused across calls, avoids repeated allocation) ──
        self._kv_buf: Optional[torch.Tensor] = None
        self._kv_reduce_work: Optional[dist.Work] = None
        self._unified_buf: Optional[torch.Tensor] = None
        self._bsa_compact_buf: Optional[torch.Tensor] = None
        self._bsa_dispatch_cache: Optional[dict] = None
        self._use_flashinfer = False     # enabled only via --flashinfer flag

        # ── FlashInfer block-sparse state ──
        self._fi_wrapper = None      # flashinfer.BlockSparseAttentionWrapper
        self._fi_planned = False     # True after plan() has been called
        self._fi_compact_k: Optional[torch.Tensor] = None  # cached compact K buffer
        self._fi_compact_v: Optional[torch.Tensor] = None  # cached compact V buffer
        self._fi_q_pad: Optional[torch.Tensor] = None      # cached padded Q buffer

        # Device-cached tensor refs (lazily populated)
        self._global_frame_mask_device: Optional[torch.Tensor] = None
        self._window_bounds_device: Optional[torch.Tensor] = None
        self._attended_indices_device: Optional[torch.Tensor] = None

        P = num_patches
        gts = self.global_token_start
        first_frame = gts // P
        last_frame = (gts + self.local_seq - 1) // P

        if rank == 0:
            kfi_str = (
                str(self.key_frame_interval) if self.key_frame_interval else "none"
            )
            # Compute attended count for a sample frame (middle of rank 0)
            mid_f = (first_frame + last_frame) // 2
            mid_lo, mid_hi = self._get_window_bounds(
                mid_f, window_size, total_num_latent_frames
            )
            mid_win_set = set(range(mid_lo, mid_hi + 1))
            mid_attended = len(self._global_set | mid_win_set)
            print(
                f"[LVSA] total_lat_frames={total_num_latent_frames}  "
                f"local_seq={self.local_seq}  "
                f"rank0_frames~[{first_frame},{last_frame}]  "
                f"window={window_size}  n_first={n_first_frames}  "
                f"kfi={kfi_str}  "
                f"global_count={len(self._global_indices)}  "
                f"attended_per_frame={mid_attended}/{total_num_latent_frames}"
            )

    def _copy_metadata_to_self(self) -> None:
        """Copy LVSAMetadata fields to self for backward compatibility.

        Existing code and tests access attributes like ``self._global_indices``,
        ``self._fi_indptr``, etc. directly.  This method bridges the new
        ``LVSAMetadata`` dataclass with the old attribute-based interface.
        """
        m = self._metadata
        self.local_seq = m.local_seq
        self.global_token_start = m.global_token_start
        self.key_frame_interval = m.key_frame_interval
        self._global_indices = m.global_indices
        self._global_set = m.global_set
        self._local_frames = m.local_frames
        self._window_ctx = m.window_ctx
        self._global_frame_mask = m.global_frame_mask
        self._window_bounds = m.window_bounds
        self._attended_indices = m.attended_indices
        self._attended_C = m.attended_C
        self._global_src_idx = m.global_src_idx
        self._global_dst_idx = m.global_dst_idx
        self._local_src_idx = m.local_src_idx
        self._local_dst_idx = m.local_dst_idx
        self._fi_indptr = m.fi_indptr
        self._fi_indices = m.fi_indices
        self._fi_M = m.fi_M
        self._fi_N = m.fi_N
        self._fi_MB = m.fi_MB
        self._fi_compact_n = m.fi_compact_n
        self._fi_global_copies = m.fi_global_copies
        self._fi_local_copies = m.fi_local_copies

    def print_attention_mask(self) -> None:
        """Print a visual T×T attention matrix showing which frames each query
        frame attends to.  Legend: G=global, W=window, .=not attended."""
        T = self.total_num_frames
        W = self.window_size

        # Header: frame indices
        # Use 2-char columns for readability
        hdr = "Q\\K " + "".join(f"{k:>3}" for k in range(T))
        print(hdr)
        print("    " + "---" * T)

        counts = []
        for f in range(T):
            win_lo, win_hi = self._get_window_bounds(f, W, T)
            win_set = set(range(win_lo, win_hi + 1))
            attended = self._global_set | win_set

            row_chars = []
            for k in range(T):
                if k in self._global_set and k in win_set:
                    row_chars.append("  X")  # both global and window
                elif k in self._global_set:
                    row_chars.append("  G")
                elif k in win_set:
                    row_chars.append("  W")
                else:
                    row_chars.append("  .")
            counts.append(len(attended))
            print(f"{f:>3} |" + "".join(row_chars) + f"  | {len(attended)}/{T}")

        print("    " + "---" * T)
        min_c, max_c = min(counts), max(counts)
        avg_c = sum(counts) / len(counts)
        print(
            f"Legend: G=global, W=window, X=both  |  "
            f"attended min={min_c} max={max_c} avg={avg_c:.1f}"
        )

    def print_attention_mask_compact(self) -> None:
        """Compact 1-char-per-column attention mask for narrow terminals.
        Delegates to the free function in ``lvsa.sparse_attention``.
        """
        from .sparse_attention import print_attention_mask_compact as _print
        _print(
            total_frames=self.total_num_frames,
            window_size=self.window_size,
            global_set=self._global_set,
            expand_window=self._expand_window,
        )

    # ── Step-wise keyframe rotation ────────────────────────────────────────────

    def set_window_size(self, new_window_size: int) -> None:
        """Dynamically switch window size (e.g. between windowed and globals-only).

        When *new_window_size* is 0 every frame sees the same set of global
        anchors — equivalent to full attention at *reference_frames* budget.
        When restored to the original value, normal LVSA resumes.

        Recomputes kfi for the new budget split and rebuilds all dependent
        data structures.
        """
        if new_window_size == self.window_size:
            return

        self.window_size = new_window_size

        # Recompute kfi for the new window/global budget split
        self.key_frame_interval = compute_auto_kfi(
            self.total_num_frames,
            self.window_size,
            self.n_first_frames,
            reference_frames=self._reference_frames,
            sparsity_scale=self._sparsity_scale,
        )

        # Force full rebuild (reset offset so _rebuild recalculates)
        old_offset = self._current_offset
        self._current_offset = -1  # sentinel to force rebuild
        self._rebuild_for_current_params(old_offset)

    def set_step(self, step_idx: int) -> None:
        """Rotate periodic keyframes for this denoising step.

        The periodic keyframes shift by ``step_idx % key_frame_interval``
        positions each step, so over multiple steps every frame gets a turn
        as a global anchor.  n_first_frames and boundary guards stay fixed.
        """
        if not self.key_frame_interval:
            return  # no periodic keyframes to rotate

        offset = step_idx % self.key_frame_interval
        if offset == self._current_offset:
            return  # same pattern — skip recomputation

        self._rebuild_for_current_params(offset)

    def set_sparsity_scale(self, new_sparsity_scale: float) -> None:
        """Dynamically adjust the sparsity scale.

        *sparsity_scale* < 1.0 → more sparse (fewer attended frames).
        *sparsity_scale* > 1.0 → less sparse (more attended frames).
        Default 1.0 preserves the original behaviour.

        Recomputes kfi for the new budget and rebuilds all dependent
        data structures.
        """
        if new_sparsity_scale == self._sparsity_scale:
            return

        self._sparsity_scale = new_sparsity_scale

        # Recompute kfi for the new sparsity budget
        self.key_frame_interval = compute_auto_kfi(
            self.total_num_frames,
            self.window_size,
            self.n_first_frames,
            reference_frames=self._reference_frames,
            sparsity_scale=self._sparsity_scale,
        )

        # Force full rebuild (reset offset so _rebuild recalculates)
        old_offset = self._current_offset
        self._current_offset = -1  # sentinel to force rebuild
        self._rebuild_for_current_params(old_offset)

    def _rebuild_for_current_params(self, offset: int) -> None:
        """Recompute all derived data structures for the current window_size,
        key_frame_interval, and the given keyframe offset.

        Called by both ``set_step()`` (rotation) and ``set_window_size()``.
        Delegates to ``LVSAMetadata.build()``.
        """
        self._current_offset = offset

        # Wait for any in-flight async all-reduce before clearing buffers
        if self._kv_reduce_work is not None:
            self._kv_reduce_work.wait()
            self._kv_reduce_work = None

        # Rebuild all pattern metadata
        self._metadata = LVSAMetadata.build(
            total_latent_frames=self.total_num_frames,
            num_patches=self.num_patches,
            window_size=self.window_size,
            n_first_frames=self.n_first_frames,
            key_frame_interval=self.key_frame_interval,
            rank=self.rank,
            world=self.world,
            expand_window=self._expand_window,
            keyframe_offset=offset,
            boundary_guards=self._boundary_guards,
            reference_frames=self._reference_frames,
            sparsity_scale=self._sparsity_scale,
        )
        self._copy_metadata_to_self()

        # Force re-upload of device tensors
        self._global_frame_mask_device = None
        self._window_bounds_device = None
        self._attended_indices_device = None

        # Reset cached KV buffers (global count may change)
        self._kv_buf = None

        # Reset FlashInfer state
        self._fi_planned = False
        self._fi_compact_k = None
        self._fi_compact_v = None
        self._fi_q_pad = None

    # ── Static helpers (delegators to sparse_attention module functions) ────────

    def _get_window_bounds(self, f: int, W: int, T: int) -> Tuple[int, int]:
        """Dispatch to expanded or adaptive window bounds based on config."""
        return get_window_bounds(
            f, W, T, self._expand_window, self._global_set, len(self._global_indices),
        )

    @staticmethod
    def _adaptive_window_bounds(f: int, W: int, T: int) -> Tuple[int, int]:
        return adaptive_window_bounds(f, W, T)

    def _expanded_window_bounds(self, f: int, W: int, T: int) -> Tuple[int, int]:
        return expanded_window_bounds(
            f, W, T, self._global_set, len(self._global_indices),
        )

    @staticmethod
    def _compute_boundary_guard_frames(
        total_frames: int, local_seq: int, num_patches: int,
        world: int, window_size: int,
    ) -> List[int]:
        return compute_boundary_guard_frames(
            total_frames, local_seq, num_patches, world, window_size,
        )

    @staticmethod
    def _compute_auto_kfi(
        total_frames: int, window_size: int, n_first_frames: int,
        reference_frames: int = 21, sparsity_scale: float = 1.0,
    ) -> int:
        return compute_auto_kfi(
            total_frames, window_size, n_first_frames, reference_frames,
            sparsity_scale,
        )

    @staticmethod
    def _compute_global_indices(
        total_frames: int, n_first_frames: int,
        key_frame_interval: Optional[int], offset: int = 0,
    ) -> List[int]:
        return compute_global_indices(
            total_frames, n_first_frames, key_frame_interval, offset,
        )

    # ── Main forward ──────────────────────────────────────────────────────────

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Any] = None,
        image_rotary_emb: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """
        Distributed block-sparse self-attention.

        hidden_states        : [B, local_seq, C]   — already sharded by CP hook
        encoder_hidden_states: [B, text_len, C]    — full text, replicated
        rotary_emb           : Wan-style RoPE (cos, sin) tuple
        image_rotary_emb     : HunyuanVideo-style RoPE tensor

        All model-specific operations (QKV extraction, RoPE, output projection,
        cross-attention) are delegated to ``self.adapter``.

        Returns a single tensor for single-stream models (Wan) or a tuple
        ``(hidden_states, encoder_hidden_states)`` for dual-stream (HunyuanVideo).
        """
        adapter = self.adapter

        # ── Unify rotary_emb from either parameter ────────────────────────────
        # Wan passes rotary_emb directly; HunyuanVideo passes image_rotary_emb.
        if rotary_emb is None and image_rotary_emb is not None:
            rotary_emb = image_rotary_emb

        # ── Split encoder states for cross-attention (if applicable) ──────────
        encoder_hidden_states_img = None
        if encoder_hidden_states is not None:
            encoder_hidden_states, encoder_hidden_states_img = (
                adapter.split_encoder_for_cross_attn(attn, encoder_hidden_states)
            )

        # ── Q / K / V projections for video tokens (model-specific) ──────────
        query, key, value = adapter.extract_qkv(
            attn, hidden_states, encoder_hidden_states,
        )
        query_dtype = query.dtype

        # ── Dual-stream encoder projections (if applicable) ──────────────────
        # For models like HunyuanVideo with separate add_q/k/v_proj for
        # encoder tokens, project them separately.  Encoder K/V becomes
        # extra global context for LVSA; encoder Q gets full attention.
        enc_q = enc_k = enc_v = None
        encoder_seq_len = 0
        if encoder_hidden_states is not None:
            enc_qkv = adapter.extract_encoder_qkv(attn, encoder_hidden_states)
            if enc_qkv is not None:
                enc_q, enc_k, enc_v = enc_qkv
                encoder_seq_len = enc_q.shape[1]

        # ── Rotary embeddings (model-specific) ────────────────────────────────
        # RoPE applies only to video tokens — adapter.apply_rotary handles this.
        local_seq = hidden_states.shape[1]
        query, key = adapter.apply_rotary(
            query, key, rotary_emb, local_seq, self.rank, self.world,
        )

        # ── Attention computation ────────────────────────────────────────────
        # Start global KV build — the all-reduce runs asynchronously so we
        # can overlap communication with the encoder KV concatenation below.
        k_global, v_global = self._build_global_kv(key, value)

        # ── While the all-reduce is in flight, prepare encoder KV ────────────
        # This work is independent of the global KV buffer contents, so it
        # can execute concurrently with the NCCL collective.
        _enc_k_ready = enc_k
        _enc_v_ready = enc_v

        # ── Wait for the async all-reduce to complete ────────────────────────
        if self._kv_reduce_work is not None:
            self._kv_reduce_work.wait()
            self._kv_reduce_work = None

        # Dual-stream: append encoder K/V to global context so every
        # video frame attends to all text tokens.
        if _enc_k_ready is not None:
            k_global = torch.cat([k_global, _enc_k_ready], dim=1)
            v_global = torch.cat([v_global, _enc_v_ready], dim=1)

        hidden_states = self._compute_lvsa(
            query, key, value, k_global, v_global, encoder_hidden_states,
        )

        # ── Dual-stream: encoder query full attention ────────────────────────
        # Encoder query tokens need full attention against all K/V (video + encoder).
        enc_output = None
        if enc_q is not None:
            # Full attention: encoder queries attend to all video K/V + encoder K/V
            full_k = torch.cat([key, enc_k], dim=1)
            full_v = torch.cat([value, enc_v], dim=1)
            enc_output = self._compute_full_attention(enc_q, full_k, full_v)

        # ── Cross-attention (model-specific, e.g. I2V image context) ─────────
        cross_out = adapter.cross_attention(
            attn, query, encoder_hidden_states_img, self._attention_backend,
        )
        if cross_out is not None:
            hidden_states = hidden_states + cross_out

        # ── Output projection (model-specific) ────────────────────────────────
        hidden_states = adapter.output_projection(attn, hidden_states, query_dtype)

        # ── Format output for the model's block forward() ────────────────────
        if enc_output is not None:
            # Dual-stream: project encoder output and return tuple
            return adapter.format_output(
                attn, hidden_states, enc_output, encoder_seq_len, query_dtype,
            )
        return hidden_states

    # ── Global K/V construction ───────────────────────────────────────────────

    def _build_global_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build K/V tensors for all global (anchor) frames.

        Token-based overlap logic
        -------------------------
        Global frame gf occupies global tokens [gf*P, (gf+1)*P).
        Our local range is [global_token_start, global_token_start+local_seq).
        The intersection tells us which local tokens to write into the buffer.
        A frame may be split across two ranks (partial ownership at boundaries);
        both ranks contribute their portion and all_reduce(SUM) assembles them.

        K and V are stacked into a single [2, B, num_global*P, H, D] buffer so
        that one all_reduce(SUM) collective covers both, halving round trips vs.
        two separate all_reduce calls.  The buffer is cached on the processor
        instance and reused across calls to avoid repeated GPU allocations.
        """
        B, local_seq, H, D = key.shape
        P = self.num_patches
        gts = self.global_token_start
        num_global = len(self._global_indices)

        # Reuse cached buffer when possible; reallocate only on shape change.
        buf_shape = (2, B, num_global * P, H, D)
        if self._kv_buf is None or self._kv_buf.shape != buf_shape:
            self._kv_buf = key.new_zeros(*buf_shape)
        else:
            self._kv_buf.zero_()

        for gi, gf in enumerate(self._global_indices):
            gf_start = gf * P
            gf_end = (gf + 1) * P
            ovl_start = max(gf_start, gts)
            ovl_end = min(gf_end, gts + local_seq)
            if ovl_start >= ovl_end:
                continue  # this global frame is entirely on another rank
            l_start = ovl_start - gts
            l_end = ovl_end - gts
            g_start = ovl_start - gf_start
            g_end = ovl_end - gf_start
            buf_sl = slice(gi * P + g_start, gi * P + g_end)
            self._kv_buf[0, :, buf_sl] = key[:, l_start:l_end]
            self._kv_buf[1, :, buf_sl] = value[:, l_start:l_end]

        # Single collective for both K and V — SUM is correct because every
        # buffer position is written by exactly the rank(s) that own those tokens.
        # Launch as async_op so the caller can overlap other work (encoder KV
        # preparation, output buffer allocation) while the all-reduce is in flight.
        if self.world > 1:
            self._kv_reduce_work = dist.all_reduce(
                self._kv_buf, op=dist.ReduceOp.SUM, async_op=True,
            )
        else:
            # Single-GPU: no communication needed.  We still copy global frame
            # tokens into _kv_buf rather than indexing key/value directly because
            # downstream consumers (Triton unified kernel, FlashInfer) require a
            # contiguous, compactly-packed global KV buffer.
            self._kv_reduce_work = None

        return self._kv_buf[0], self._kv_buf[1]

    # ── Full attention (for encoder queries in dual-stream) ────────────────────

    def _compute_full_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Full (dense) attention for encoder query tokens in dual-stream models.

        Parameters
        ----------
        query : [B, enc_seq, H, D] — encoder query tokens
        key   : [B, full_seq, H, D] — all K (video + encoder)
        value : [B, full_seq, H, D] — all V (video + encoder)

        Returns
        -------
        [B, enc_seq, H, D] — attention output for encoder queries
        """
        import torch.nn.functional as F

        # Transpose to [B, H, seq, D] for scaled_dot_product_attention
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return out.transpose(1, 2)  # back to [B, seq, H, D]

    # ── LVSA loop ──────────────────────────────────────────────────────────────

    def _compute_lvsa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_global: torch.Tensor,
        v_global: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self._use_flashinfer:
            return self._compute_lvsa_flashinfer(query, key, value, k_global, v_global)
        return self._compute_lvsa_sdpa(query, key, value, k_global, v_global)

    def _compute_lvsa_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_global: torch.Tensor,
        v_global: torch.Tensor,
    ) -> torch.Tensor:
        """Per-frame block-sparse LVSA — delegates to ``lvsa_sdpa()``."""
        return lvsa_sdpa(
            query, key, value, k_global, v_global,
            self._metadata, self._attention_backend,
        )

    # ── FlashInfer block-sparse path ─────────────────────────────────────────

    def _ensure_flashinfer_planned(
        self, query: torch.Tensor, enc_tokens: int = 0,
    ) -> None:
        """Initialize FlashInfer wrapper and call plan().

        The plan is reused across all diffusion steps as long as ``enc_tokens``
        stays the same.  On the first call (or if encoder length changes) the
        CSR is extended with extra block columns so every Q block also attends
        to the encoder tokens appended to k_global by dual-stream models.

        Parameters
        ----------
        enc_tokens : number of encoder tokens appended to k_global (0 for
                     single-stream models like Wan).
        """
        # Re-plan only when enc_tokens changes (first call, or 0 → N).
        if self._fi_planned and getattr(self, '_fi_enc_tokens', 0) == enc_tokens:
            return

        device = query.device
        H = query.shape[2]
        D = query.shape[3]
        P = self.num_patches

        # ── Extend CSR for encoder tokens (if any) ──────────────────────
        # Encoder tokens are appended after the compact video frames in the
        # KV buffer.  We add ceil(enc_tokens / P) extra block columns that
        # every Q block row attends to.
        base_indptr = self._fi_indptr.tolist()
        base_indices = self._fi_indices.tolist()
        compact_n = self._fi_compact_n

        enc_blocks = -(-enc_tokens // P) if enc_tokens > 0 else 0  # ceil div
        if enc_blocks > 0:
            MB = len(base_indptr) - 1
            new_indices = []
            new_indptr = [0]
            enc_col_ids = list(range(compact_n, compact_n + enc_blocks))
            for qi in range(MB):
                row_start = base_indptr[qi]
                row_end = base_indptr[qi + 1]
                row = base_indices[row_start:row_end] + enc_col_ids
                new_indices.extend(row)
                new_indptr.append(len(new_indices))
            indptr = torch.tensor(new_indptr, dtype=torch.int32, device=device)
            indices = torch.tensor(new_indices, dtype=torch.int32, device=device)
            N = (compact_n + enc_blocks) * P
        else:
            indptr = self._fi_indptr.to(device)
            indices = self._fi_indices.to(device)
            N = self._fi_N

        # Allocate workspace (128 MB)
        if not hasattr(self, '_fi_workspace') or self._fi_workspace is None:
            self._fi_workspace = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device=device,
            )

        self._fi_wrapper = flashinfer.BlockSparseAttentionWrapper(self._fi_workspace)

        dtype_map = {
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.float32: "float32",
        }
        q_dtype_str = dtype_map.get(query.dtype, "bfloat16")

        self._fi_wrapper.plan(
            indptr=indptr,
            indices=indices,
            M=self._fi_M,
            N=N,
            R=P,
            C=P,
            num_qo_heads=H,
            num_kv_heads=H,
            head_dim=D,
            q_data_type=q_dtype_str,
        )
        self._fi_planned = True
        self._fi_enc_tokens = enc_tokens
        self._fi_N_actual = N  # total KV length including encoder blocks

    def _compute_lvsa_flashinfer(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        k_global: torch.Tensor,
        v_global: torch.Tensor,
    ) -> torch.Tensor:
        """FlashInfer block-sparse LVSA — buffer fill + delegates to ``lvsa_flashinfer()``."""
        B, local_seq, H, D = query.shape
        P = self.num_patches
        M = self._fi_M

        # ── Detect encoder tokens appended to k_global ──
        num_global_video_tokens = len(self._global_indices) * P
        total_global_tokens = k_global.shape[1]
        enc_tokens = total_global_tokens - num_global_video_tokens

        self._ensure_flashinfer_planned(query, enc_tokens)
        compact_N = self._fi_N_actual

        # ── Build compact KV buffer ──
        compact_shape = (B, compact_N, H, D)
        if self._fi_compact_k is None or self._fi_compact_k.shape != compact_shape:
            self._fi_compact_k = query.new_zeros(*compact_shape)
            self._fi_compact_v = query.new_zeros(*compact_shape)
        ck, cv = self._fi_compact_k, self._fi_compact_v

        # Video globals → compact positions (from pre-computed copy list)
        for src_s, dst_s in self._fi_global_copies:
            ck[:, dst_s:dst_s + P] = k_global[:, src_s:src_s + P]
            cv[:, dst_s:dst_s + P] = v_global[:, src_s:src_s + P]

        # Local (non-global) video frames → compact positions
        for src_s, dst_s in self._fi_local_copies:
            ck[:, dst_s:dst_s + P] = key[:, src_s:src_s + P]
            cv[:, dst_s:dst_s + P] = value[:, src_s:src_s + P]

        # Encoder tokens → appended after compact video frames
        if enc_tokens > 0:
            enc_start_dst = self._fi_compact_n * P
            enc_start_src = num_global_video_tokens
            ck[:, enc_start_dst:enc_start_dst + enc_tokens] = (
                k_global[:, enc_start_src:enc_start_src + enc_tokens]
            )
            cv[:, enc_start_dst:enc_start_dst + enc_tokens] = (
                v_global[:, enc_start_src:enc_start_src + enc_tokens]
            )
            remainder = enc_tokens % P
            if remainder != 0:
                pad_start = enc_start_dst + enc_tokens
                pad_end = enc_start_dst + (-(-enc_tokens // P)) * P
                ck[:, pad_start:pad_end] = 0
                cv[:, pad_start:pad_end] = 0

        # ── Pad Q to M = MB * P if needed (cached, tail stays zero) ──
        if local_seq < M:
            if self._fi_q_pad is None or self._fi_q_pad.shape != (B, M, H, D):
                self._fi_q_pad = query.new_zeros(B, M, H, D)
            self._fi_q_pad[:, :local_seq] = query
            q_padded = self._fi_q_pad
        else:
            q_padded = query

        return lvsa_flashinfer(q_padded, ck, cv, self._fi_wrapper, local_seq, M)


# Backwards-compatible alias for existing code that references the old name.
WanDistributedLVSAProcessor = DistributedLVSAProcessor
