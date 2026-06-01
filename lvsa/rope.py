"""
rope.py — Rotary Position Embedding helpers for Wan context-parallel inference.
"""

from typing import Tuple

import torch


def apply_rotary_emb(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embeddings to hidden_states.

    hidden_states : [B, seq, heads, head_dim]
    freqs_cos/sin : [B?, seq, 1?, head_dim/2] (broadcast-compatible with hidden_states)
    """
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


def slice_rotary_emb(
    rotary_emb: Tuple[torch.Tensor, ...],
    local_seq: int,
    rank: int,
    world: int,
) -> Tuple[torch.Tensor, ...]:
    """
    Slice a (cos, sin) rotary-embedding tuple from full_seq to the local rank's
    chunk [rank * local_seq : (rank+1) * local_seq].

    Scans each tensor for a dimension of size full_seq and slices it.  If no
    such dimension is found the tensor is returned unchanged (e.g., head-dim
    only tensors).
    """
    full_seq = local_seq * world
    sliced = []
    for r in rotary_emb:
        cut = False
        for dim in range(r.dim()):
            if r.shape[dim] == full_seq:
                idx = [slice(None)] * r.dim()
                idx[dim] = slice(rank * local_seq, (rank + 1) * local_seq)
                sliced.append(r[tuple(idx)].contiguous())
                cut = True
                break
        if not cut:
            sliced.append(r)
    return type(rotary_emb)(sliced)
