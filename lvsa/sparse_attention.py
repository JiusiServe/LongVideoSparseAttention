"""
sparse_attention.py — Stateless sparse windowed attention primitives.

This module contains:
- Pure helper functions for window/keyframe computation
- ``LVSAMetadata`` dataclass holding all pre-computed pattern indices
- Standalone backend functions (SDPA, FlashInfer)
- ``sparse_windowed_attention()`` top-level dispatcher

All functions are stateless and have no dependency on ``lvsa_processor.py``,
``ModelAdapter``, or ``torch.distributed``.  This is the foundation for
both the existing ``DistributedLVSAProcessor`` wrapper and the future
``lvsa-vllm-omni`` plugin.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F


# ── Lazy optional imports ────────────────────────────────────────────────────

def _get_dispatch_attention_fn():
    """Get diffusers' optimized attention dispatch, or None."""
    try:
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        return dispatch_attention_fn
    except ImportError:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions (extracted from DistributedLVSAProcessor static methods)
# ═══════════════════════════════════════════════════════════════════════════════


def adaptive_window_bounds(f: int, W: int, T: int) -> Tuple[int, int]:
    """Compute adaptive sliding window bounds that maintain constant width
    at sequence boundaries.  Instead of clipping to [max(0,f-W), min(T-1,f+W)]
    which loses frames at the edges, shift the window so every frame attends
    to min(2W+1, T) frames."""
    win_start = max(0, min(f - W, T - 1 - 2 * W))
    win_end = min(T - 1, max(f + W, 2 * W))
    return win_start, win_end


def expanded_window_bounds(
    f: int, W: int, T: int,
    global_set: Set[int],
    global_count: int,
) -> Tuple[int, int]:
    """Expand the adaptive window so that it contains 2W+1 *non-global* frames.

    The base adaptive window may overlap with global frames, wasting slots.
    This method expands outward (alternating left/right) until the window
    covers 2W+1 non-global frames or hits the sequence boundaries.

    Total unique attended frames = num_global + min(2W+1, T - num_global).
    """
    win_lo, win_hi = adaptive_window_bounds(f, W, T)
    target = min(2 * W + 1, T - global_count)

    non_global = sum(
        1 for wf in range(win_lo, win_hi + 1) if wf not in global_set
    )

    while non_global < target:
        expanded = False
        if win_lo > 0:
            win_lo -= 1
            if win_lo not in global_set:
                non_global += 1
            expanded = True
        if non_global >= target:
            break
        if win_hi < T - 1:
            win_hi += 1
            if win_hi not in global_set:
                non_global += 1
            expanded = True
        if not expanded:
            break
    return win_lo, win_hi


def get_window_bounds(
    f: int, W: int, T: int,
    expand: bool,
    global_set: Set[int],
    global_count: int,
) -> Tuple[int, int]:
    """Dispatch to expanded or adaptive window bounds.

    When W=0 (globals-only mode), returns an empty range so no window
    tokens are included — all attention goes through globals.
    """
    if W == 0:
        return (f, f - 1)  # empty range: start > end
    if expand:
        return expanded_window_bounds(f, W, T, global_set, global_count)
    return adaptive_window_bounds(f, W, T)


def compute_boundary_guard_frames(
    total_frames: int,
    local_seq: int,
    num_patches: int,
    world: int,
    window_size: int,
) -> List[int]:
    """Pre-compute boundary guard frames for context parallelism.

    For each rank boundary at token b = r * local_seq (r = 1..world-1):
    boundary_frame = b // num_patches
    guard range    = [boundary_frame - window_size, boundary_frame + window_size]
    These frames are made global so global_broadcast assembles their complete
    K/V on every rank, closing the attention gap at CP token splits.
    """
    guards: Set[int] = set()
    for r in range(1, world):
        bf = (r * local_seq) // num_patches
        for f in range(
            max(0, bf - window_size), min(total_frames, bf + window_size + 1)
        ):
            guards.add(f)
    return sorted(guards)


def compute_auto_kfi(
    total_frames: int,
    window_size: int,
    n_first_frames: int,
    reference_frames: int = 21,
    sparsity_scale: float = 1.0,
) -> int:
    """Compute key_frame_interval so that total attended frames per query
    approximates *reference_frames * sparsity_scale*.

    *sparsity_scale* < 1.0 → more sparse (fewer attended frames).
    *sparsity_scale* > 1.0 → less sparse (more attended frames).

    With expanded window bounds, total_attended = num_globals + min(2W+1, T - num_globals).
    Integer keyframe spacing means we cannot always hit the target exactly;
    the realized budget may differ from the target by up to a few frames.
    Two heuristics are combined:

      1. *Strict principled* — largest kfi whose periodic+initial anchor set
         has size >= target_globals. Exact when achievable; may collapse to
         kfi=1 (dense) at extrapolation horizons just above the reference
         where no kfi >= 2 reaches the target.
      2. *Closed-form fallback* — keeps the pattern sparse at those edge cases
         by sizing kfi from a (T - mandatory) / (scaled_ref - mandatory) ratio.
         May under-allocate vs target by up to a few frames.

    Returning ``max(strict, fallback)`` produces the sparser pattern when the
    two disagree, preserving the under-budget-but-still-sparse behavior at
    HunyuanVideo extrapolation horizons (T just above ref=33) and matching
    the legacy pre-consolidation behavior across the report's experimental
    grid (verified by ``scripts/verify_auto_kfi.py``).
    """
    scaled_ref = max(n_first_frames + 1, int(reference_frames * sparsity_scale))
    # At or below the reference budget, every frame is a global anchor.
    # This keeps the globals-only bands of the denoising schedule at 100%
    # coverage; without it, warmup/cooldown steps would collapse to only
    # the n_first anchors and introduce sparsity even when the budget
    # should cover the whole sequence.
    if total_frames <= scaled_ref:
        return 1

    target_attended = min(scaled_ref, total_frames)
    target_globals = max(
        n_first_frames,
        target_attended - min(2 * window_size + 1, total_frames),
    )

    # Strict principled kfi: largest kfi with |G_init ∪ G_per| >= target_globals.
    strict_kfi = 1
    for kfi in range(total_frames, 0, -1):
        indices = set(range(min(n_first_frames, total_frames)))
        indices.update(range(0, total_frames, kfi))
        if len(indices) >= target_globals:
            strict_kfi = kfi
            break

    # Closed-form fallback: keeps the pattern sparse when the strict criterion
    # cannot be met with kfi >= 2 (HV-style extrapolation just above ref).
    mandatory = 2 * window_size + n_first_frames
    max_frames_total = max(total_frames - mandatory, scaled_ref - mandatory)
    max_frames_ref = max(scaled_ref - mandatory, 1)
    fallback_kfi = math.ceil(max_frames_total / max_frames_ref)

    return max(strict_kfi, fallback_kfi)


def compute_global_indices(
    total_frames: int,
    n_first_frames: int,
    key_frame_interval: Optional[int],
    offset: int = 0,
) -> List[int]:
    """Sorted list of global (anchor) latent frame indices.

    *offset* shifts the periodic keyframe grid.  Uses modular wrapping
    so that every offset produces exactly ``ceil(T / kfi)`` periodic
    keyframes, keeping the total attended frame budget constant across
    rotation steps.
    """
    global_set = set(range(min(n_first_frames, total_frames)))
    if key_frame_interval is not None and key_frame_interval > 0:
        n_keyframes = math.ceil(total_frames / key_frame_interval)
        for i in range(n_keyframes):
            global_set.add((offset + i * key_frame_interval) % total_frames)
    return sorted(global_set)


def print_attention_mask_compact(
    total_frames: int,
    window_size: int,
    global_set: Set[int],
    expand_window: bool,
    file=None,
) -> None:
    """Compact 1-char-per-column attention mask suitable for narrow terminals.

    Free function form: callable from any code path that has the four primitives
    (T, W, global_set, expand_window) — including the vllm-omni plugin where
    there is no DistributedLVSAProcessor instance.

    Legend: ``G`` = global anchor, ``W`` = local-window, ``X`` = both,
    ``.`` = not attended. Each row ends with the per-frame attended count.

    Pass ``file=sys.stderr`` (or any text stream) to redirect output.
    """
    T = total_frames
    W = window_size
    pad = len(str(T - 1)) if T > 1 else 1

    # Header with tick marks every 5 frames
    hdr = " " * (pad + 2)
    for k in range(T):
        hdr += str(k % 10) if k % 5 == 0 else " "
    print(hdr, file=file)

    counts = []
    for f in range(T):
        win_lo, win_hi = get_window_bounds(
            f, W, T, expand=expand_window,
            global_set=global_set, global_count=len(global_set),
        )
        win_set = set(range(win_lo, win_hi + 1)) if win_lo <= win_hi else set()
        attended = global_set | win_set

        row_chars = []
        for k in range(T):
            is_g = k in global_set
            is_w = k in win_set
            if is_g and is_w:
                row_chars.append("X")
            elif is_g:
                row_chars.append("G")
            elif is_w:
                row_chars.append("W")
            else:
                row_chars.append(".")
        counts.append(len(attended))
        print(f"{f:>{pad}} |{''.join(row_chars)}| {len(attended)}", file=file)

    if counts:
        min_c, max_c = min(counts), max(counts)
        avg_c = sum(counts) / len(counts)
        print(
            f"G=global W=window X=both .=skip  "
            f"min={min_c} max={max_c} avg={avg_c:.1f}/{T}",
            file=file,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LVSAMetadata — all pre-computed pattern state
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LVSAMetadata:
    """Pre-computed sparse attention pattern metadata.

    Contains all index structures needed by the three backends.
    Constructed via ``LVSAMetadata.build()`` or manually.

    Not frozen — tensor fields are moved to device via ``ensure_device()``.
    Config fields are immutable by convention.
    """

    # ── Pattern config ──
    total_latent_frames: int
    num_patches: int
    window_size: int
    n_first_frames: int
    key_frame_interval: Optional[int]
    expand_window: bool
    rank: int
    world: int

    # ── Derived geometry ──
    local_seq: int
    global_token_start: int
    global_indices: List[int]
    global_set: Set[int]
    local_frames: List[Tuple[int, int, int]]
    window_ctx: Dict[int, List[Tuple[int, int]]]

    # ── Triton data (CPU tensors, moved to device via ensure_device) ──
    global_frame_mask: torch.Tensor
    window_bounds: torch.Tensor
    attended_indices: torch.Tensor
    attended_C: int
    global_src_idx: torch.Tensor
    global_dst_idx: torch.Tensor
    local_src_idx: torch.Tensor
    local_dst_idx: torch.Tensor

    # ── FlashInfer CSR data ──
    fi_indptr: torch.Tensor
    fi_indices: torch.Tensor
    fi_M: int
    fi_N: int
    fi_MB: int
    fi_compact_n: int
    fi_global_copies: List[Tuple[int, int]]
    fi_local_copies: List[Tuple[int, int]]

    @classmethod
    @torch.compiler.disable  # data-dependent CSR build: opaque to dynamo so it
    # does not graph-break / recompile per step under torch.compile (the eager
    # standalone is unaffected; this only matters under vllm-omni's compile).
    def build(
        cls,
        total_latent_frames: int,
        num_patches: int,
        window_size: int,
        n_first_frames: int,
        key_frame_interval: Optional[int],
        rank: int,
        world: int,
        expand_window: bool = True,
        keyframe_offset: int = 0,
        boundary_guards: Optional[List[int]] = None,
        reference_frames: int = 21,
        sparsity_scale: float = 1.0,
    ) -> "LVSAMetadata":
        """Factory: compute all derived index structures from config params."""
        T = total_latent_frames
        P = num_patches

        # ── Sequence geometry ──
        total_seq = T * P
        assert total_seq % world == 0, (
            f"total_seq={total_seq} (T={T} x P={P}) must be divisible by world={world}."
        )
        local_seq = total_seq // world
        gts = rank * local_seq

        # ── Global indices ──
        # kfi comes from the caller verbatim; callers that want autoscaling
        # (parallel.py for --auto-keyframes / --rotate-keyframes, the plugin
        # hooks for vllm-omni) call compute_auto_kfi themselves before passing
        # the result here.
        kfi = key_frame_interval
        user_globals = compute_global_indices(T, n_first_frames, kfi, keyframe_offset)
        if boundary_guards is None:
            boundary_guards = []
        global_indices_list = sorted(set(user_globals) | set(boundary_guards))
        global_set = set(global_indices_list)

        # ── Per-frame token ranges ──
        first_frame = gts // P
        last_frame = (gts + local_seq - 1) // P

        local_frames: List[Tuple[int, int, int]] = []
        for f in range(first_frame, last_frame + 1):
            q_l_start = max(f * P, gts) - gts
            q_l_end = min((f + 1) * P, gts + local_seq) - gts
            local_frames.append((f, q_l_start, q_l_end))

        # ── Window context per frame ──
        window_ctx: Dict[int, List[Tuple[int, int]]] = {}
        for f_global, _, _ in local_frames:
            win_start, win_end = get_window_bounds(
                f_global, window_size, T, expand_window, global_set, len(global_indices_list),
            )
            parts: List[Tuple[int, int]] = []
            for wf in range(win_start, win_end + 1):
                if wf in global_set:
                    continue
                ovl_s = max(wf * P, gts)
                ovl_e = min((wf + 1) * P, gts + local_seq)
                if ovl_s < ovl_e:
                    parts.append((ovl_s - gts, ovl_e - gts))
            window_ctx[f_global] = parts

        # ── Global frame mask [T] for Triton ──
        global_frame_mask = torch.zeros(T, dtype=torch.int8)
        for gf in global_indices_list:
            global_frame_mask[gf] = 1

        # ── Window bounds [T, 2] for Triton ──
        window_bounds_t = torch.zeros(T, 2, dtype=torch.int32)
        for f in range(T):
            lo, hi = get_window_bounds(
                f, window_size, T, expand_window, global_set, len(global_indices_list),
            )
            window_bounds_t[f, 0] = lo
            window_bounds_t[f, 1] = hi

        # ── Global/local src/dst index tensors for Triton unified buffer fill ──
        global_src, global_dst = [], []
        for gi, gf in enumerate(global_indices_list):
            for t in range(P):
                global_src.append(gi * P + t)
                global_dst.append(gf * P + t)
        global_src_idx = torch.tensor(global_src, dtype=torch.long)
        global_dst_idx = torch.tensor(global_dst, dtype=torch.long)

        local_src, local_dst = [], []
        for f in range(first_frame, last_frame + 1):
            if f in global_set:
                continue
            ovl_s = max(f * P, gts)
            ovl_e = min((f + 1) * P, gts + local_seq)
            if ovl_s >= ovl_e:
                continue
            for t in range(ovl_e - ovl_s):
                local_src.append((ovl_s - gts) + t)
                local_dst.append(ovl_s + t)
        local_src_idx = torch.tensor(local_src, dtype=torch.long)
        local_dst_idx = torch.tensor(local_dst, dtype=torch.long)

        # ── Attended indices [T_local, C] for Triton indexed kernel ──
        attended_indices, attended_C = _build_attended_indices(
            T, P, gts, local_seq, window_size, global_indices_list,
            expand_window, global_set,
        )

        # ── FlashInfer CSR ──
        fi_result = _build_flashinfer_csr(
            T, P, gts, local_seq, window_size, global_indices_list,
            global_set, expand_window,
        )

        return cls(
            total_latent_frames=T,
            num_patches=P,
            window_size=window_size,
            n_first_frames=n_first_frames,
            key_frame_interval=kfi,
            expand_window=expand_window,
            rank=rank,
            world=world,
            local_seq=local_seq,
            global_token_start=gts,
            global_indices=global_indices_list,
            global_set=global_set,
            local_frames=local_frames,
            window_ctx=window_ctx,
            global_frame_mask=global_frame_mask,
            window_bounds=window_bounds_t,
            attended_indices=attended_indices,
            attended_C=attended_C,
            global_src_idx=global_src_idx,
            global_dst_idx=global_dst_idx,
            local_src_idx=local_src_idx,
            local_dst_idx=local_dst_idx,
            fi_indptr=fi_result["indptr"],
            fi_indices=fi_result["indices"],
            fi_M=fi_result["M"],
            fi_N=fi_result["N"],
            fi_MB=fi_result["MB"],
            fi_compact_n=fi_result["compact_n"],
            fi_global_copies=fi_result["global_copies"],
            fi_local_copies=fi_result["local_copies"],
        )

    def ensure_device(self, device: torch.device) -> None:
        """Move tensor fields to *device* (idempotent — no-op if already there).

        Note: ``fi_indptr`` and ``fi_indices`` are intentionally NOT moved
        here. They're block-CSR data consumed by host-side mask builders
        (``build_bsa_mask_compact``) and by FlashInfer's planning pass
        (which makes its own NPU copies). Keeping them on CPU avoids
        unnecessary host↔device round-trips. Mask builders defensively
        coerce to CPU before iterating, so this is robust even if a
        future caller hands us already-device tensors.
        """
        tensor_fields = [
            "global_frame_mask", "window_bounds", "attended_indices",
            "global_src_idx", "global_dst_idx", "local_src_idx", "local_dst_idx",
        ]
        for fname in tensor_fields:
            t = getattr(self, fname)
            if t.device != device:
                object.__setattr__(self, fname, t.to(device))


# ── Internal builders (called by LVSAMetadata.build) ──────────────────────────


def _build_attended_indices(
    T: int, P: int, gts: int, local_seq: int,
    window_size: int, global_indices: List[int],
    expand_window: bool, global_set: Set[int],
) -> Tuple[torch.Tensor, int]:
    """Build [T_local, C] attended frame indices for the Triton indexed kernel.

    Returns (attended_indices, C).
    """
    W = window_size
    first_frame = gts // P
    last_frame = (gts + local_seq - 1) // P
    T_local = last_frame - first_frame + 1

    per_frame: list = []
    max_c = 0
    for qi in range(T_local):
        f_global = first_frame + qi
        attended = set(global_indices)
        win_lo, win_hi = get_window_bounds(
            f_global, W, T, expand_window, global_set, len(global_indices),
        )
        for wf in range(win_lo, win_hi + 1):
            attended.add(wf)
        attended_sorted = sorted(attended)
        per_frame.append(attended_sorted)
        if len(attended_sorted) > max_c:
            max_c = len(attended_sorted)

    C = max_c
    indices = torch.empty(T_local, C, dtype=torch.int32)
    for qi, att in enumerate(per_frame):
        n = len(att)
        indices[qi, :n] = torch.tensor(att, dtype=torch.int32)
        if n < C:
            indices[qi, n:] = att[-1]

    return indices, C


def _build_flashinfer_csr(
    T: int, P: int, gts: int, local_seq: int,
    window_size: int, global_indices: List[int],
    global_set: Set[int], expand_window: bool,
) -> dict:
    """Build CSR (indptr, indices) for FlashInfer BlockSparseAttentionWrapper.

    Returns a dict with keys: indptr, indices, M, N, MB, compact_n,
    global_copies, local_copies.
    """
    W = window_size
    first_frame = gts // P
    MB = -(-local_seq // P)  # ceil(local_seq / P)

    # ── Pass 1: collect attended frames per Q block ──
    all_attended: Set[int] = set()
    per_q_attended: List[Set[int]] = []

    for qi in range(MB):
        f_global = first_frame + qi
        attended: Set[int] = set(global_indices)
        win_lo, win_hi = get_window_bounds(
            f_global, W, T, expand_window, global_set, len(global_indices),
        )
        for wf in range(win_lo, win_hi + 1):
            if wf in global_set:
                continue
            ovl_s = max(wf * P, gts)
            ovl_e = min((wf + 1) * P, gts + local_seq)
            if ovl_s < ovl_e:
                attended.add(wf)
        per_q_attended.append(attended)
        all_attended |= attended

    # ── Compact layout ──
    compact_frames = sorted(all_attended)
    compact_n = len(compact_frames)
    frame_to_compact = {f: i for i, f in enumerate(compact_frames)}

    # ── CSR with compact indices ──
    indptr_list = [0]
    indices_list: List[int] = []
    for qi in range(MB):
        ci_sorted = sorted(frame_to_compact[f] for f in per_q_attended[qi])
        indices_list.extend(ci_sorted)
        indptr_list.append(len(indices_list))

    # ── Copy instructions ──
    global_idx_map = {gf: gi for gi, gf in enumerate(global_indices)}
    global_copies: List[Tuple[int, int]] = []
    local_copies: List[Tuple[int, int]] = []

    for ci, gf in enumerate(compact_frames):
        dst = ci * P
        if gf in global_set:
            global_copies.append((global_idx_map[gf] * P, dst))
        else:
            local_copies.append((gf * P - gts, dst))

    # ── Validation ──
    indptr = torch.tensor(indptr_list, dtype=torch.int32)
    indices = torch.tensor(indices_list, dtype=torch.int32)

    assert indptr[0] == 0, "CSR indptr must start at 0"
    assert len(indptr) == MB + 1, f"CSR indptr length {len(indptr)} != MB+1={MB + 1}"
    total_nnz = indptr[-1].item()
    assert len(indices) == total_nnz, f"CSR indices length {len(indices)} != nnz={total_nnz}"
    if compact_n > 0:
        assert indices.max().item() < compact_n, (
            f"CSR index {indices.max().item()} >= compact_n={compact_n}"
        )
    assert len(global_copies) + len(local_copies) == compact_n, (
        f"Copy instructions ({len(global_copies)} global + {len(local_copies)} local) "
        f"!= compact_n={compact_n}"
    )

    return {
        "indptr": indptr,
        "indices": indices,
        "M": MB * P,
        "N": compact_n * P,
        "MB": MB,
        "compact_n": compact_n,
        "global_copies": global_copies,
        "local_copies": local_copies,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone backend functions
# ═══════════════════════════════════════════════════════════════════════════════


def lvsa_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_global: torch.Tensor,
    v_global: torch.Tensor,
    metadata: LVSAMetadata,
    attention_backend: Any = None,
) -> torch.Tensor:
    """Per-frame block-sparse LVSA using SDPA/Flash Attention dispatch.

    For each local frame, builds K/V context = global anchors + local window,
    then calls the attention function.

    Parameters
    ----------
    query : [B, local_seq, H, D]
    key, value, k_global, v_global : [B, *, H_kv, D]
        ``H_kv`` may be < ``H`` (grouped-query attention). When so, the SDPA
        kernel handles head broadcasting natively (``enable_gqa=True``) — the
        caller must NOT repeat-KV up to ``H`` first (doing so wastes ~4x KV
        bandwidth + VRAM). When ``H_kv == H`` this is a no-op.
    metadata : LVSAMetadata with local_frames and window_ctx
    attention_backend : optional backend hint for dispatch_attention_fn
    """
    B, local_seq, H, D = query.shape
    enable_gqa = key.shape[2] != H
    output = query.new_empty(B, local_seq, H, D)

    _dispatch_fn = _get_dispatch_attention_fn()

    for f_global, q_l_start, q_l_end in metadata.local_frames:
        q_chunk = query[:, q_l_start:q_l_end]

        win_parts = metadata.window_ctx[f_global]
        if win_parts:
            k_ctx = torch.cat(
                [k_global] + [key[:, s:e] for s, e in win_parts], dim=1
            )
            v_ctx = torch.cat(
                [v_global] + [value[:, s:e] for s, e in win_parts], dim=1
            )
        else:
            k_ctx = k_global
            v_ctx = v_global

        if _dispatch_fn is not None:
            output[:, q_l_start:q_l_end] = _dispatch_fn(
                q_chunk, k_ctx, v_ctx,
                attn_mask=None, dropout_p=0.0, is_causal=False,
                enable_gqa=enable_gqa,
                backend=attention_backend, parallel_config=None,
            )
        else:
            q_t = q_chunk.transpose(1, 2)
            k_t = k_ctx.transpose(1, 2)
            v_t = v_ctx.transpose(1, 2)
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, attn_mask=None,
                dropout_p=0.0, is_causal=False, enable_gqa=enable_gqa,
            )
            output[:, q_l_start:q_l_end] = out.transpose(1, 2)

    return output


def lvsa_flashinfer(
    query_padded: torch.Tensor,
    k_compact: torch.Tensor,
    v_compact: torch.Tensor,
    fi_wrapper: Any,
    local_seq: int,
    fi_M: int,
) -> torch.Tensor:
    """Single-call block-sparse LVSA via FlashInfer.

    The caller must pre-fill k_compact/v_compact and call plan() on
    the wrapper.  This function only runs the kernel and slices output.

    Parameters
    ----------
    query_padded : [B, M, H, D] — padded to M = MB * P
    k_compact, v_compact : [B, compact_N, H, D]
    fi_wrapper : flashinfer.BlockSparseAttentionWrapper (pre-planned)
    local_seq : actual query length (for output slicing)
    fi_M : M = MB * P
    """
    B = query_padded.shape[0]
    H = query_padded.shape[2]
    D = query_padded.shape[3]

    output = query_padded.new_empty(B, fi_M, H, D)
    for b in range(B):
        output[b] = fi_wrapper.run(
            query_padded[b],
            k_compact[b],
            v_compact[b],
        )

    return output[:, :local_seq]


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level dispatcher
# ═══════════════════════════════════════════════════════════════════════════════


def sparse_windowed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_global: torch.Tensor,
    v_global: torch.Tensor,
    metadata: LVSAMetadata,
    backend: str = "sdpa",
    *,
    # Backend-specific optional args:
    k_compact: Optional[torch.Tensor] = None,
    v_compact: Optional[torch.Tensor] = None,
    fi_wrapper: Optional[Any] = None,
    attention_backend: Any = None,
) -> torch.Tensor:
    """Stateless sparse windowed attention — top-level dispatcher.

    Dispatches to the requested backend.  For FlashInfer the caller
    must provide pre-filled compact buffers.

    Parameters
    ----------
    query, key, value : [B, local_seq, H, D]
    k_global, v_global : [B, num_global*P, H, D] (pre-gathered)
    metadata : LVSAMetadata
    backend : "sdpa" | "flashinfer"
    """
    if backend == "flashinfer":
        assert k_compact is not None and v_compact is not None and fi_wrapper is not None, (
            "FlashInfer backend requires k_compact, v_compact, and fi_wrapper"
        )
        return lvsa_flashinfer(
            query, k_compact, v_compact, fi_wrapper,
            metadata.local_seq, metadata.fi_M,
        )
    else:
        return lvsa_sdpa(
            query, key, value, k_global, v_global, metadata, attention_backend,
        )
