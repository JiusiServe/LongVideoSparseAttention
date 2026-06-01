"""RIFLEx: training-free length extrapolation via RoPE frequency modification.

Reference: "RIFLEx: A Free Lunch for Length Extrapolation in Video Diffusion
Transformers" (arXiv 2502.15894).

Modifies a single temporal RoPE frequency such that its period matches the
extrapolated sequence length, eliminating periodic-repetition failures when
generating videos longer than the training horizon. Fully orthogonal to
LVSA's sparse attention: operates on the RoPE buffer at pipeline load time,
whereas LVSA patches the attention computation. Can be composed with LVSA
by applying RIFLEx first, then installing the LVSA processor.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch


def identify_intrinsic_k(t_dim: int, training_length: int,
                         theta: float = 10000.0) -> int:
    """Pick the temporal frequency index whose period is closest to
    ``training_length``.

    RoPE period for index j is ``2π · theta^(2j/t_dim)``.

    Args:
        t_dim: temporal component of the RoPE head-dim split.
        training_length: number of (latent) frames seen at training.
        theta: RoPE base frequency (default 10000.0).
    """
    half = t_dim // 2
    periods = [2.0 * math.pi * (theta ** (2.0 * j / t_dim)) for j in range(half)]
    diffs = [abs(p - training_length) for p in periods]
    return int(min(range(half), key=diffs.__getitem__))


def _build_riflex_temporal_table(t_dim: int, max_seq_len: int, theta: float,
                                 k: int, s: float, training_length: int,
                                 dtype: torch.dtype, device: torch.device):
    """Rebuild the temporal (cos, sin) RoPE sub-table with freq[k] overridden.

    Mirrors ``diffusers.models.embeddings.get_1d_rotary_pos_embed`` with
    ``use_real=True, repeat_interleave_real=True`` so the returned tensors
    slot directly into the first ``t_dim`` columns of ``WanRotaryPosEmbed``'s
    concatenated ``freqs_cos`` / ``freqs_sin`` buffers.
    """
    freqs_dtype = torch.float64
    pos = torch.arange(max_seq_len, dtype=freqs_dtype, device=device)
    base = torch.arange(0, t_dim, 2, dtype=freqs_dtype, device=device)
    freqs = 1.0 / (theta ** (base / t_dim))
    freqs[k] = 2.0 * math.pi / (float(training_length) * float(s))
    table = torch.outer(pos, freqs)
    cos = table.cos().repeat_interleave(2, dim=1).to(device=device, dtype=dtype)
    sin = table.sin().repeat_interleave(2, dim=1).to(device=device, dtype=dtype)
    return cos, sin


def apply_riflex_to_wan_pipe(pipe: Any, s: float,
                             k: Optional[int] = None,
                             training_length: Optional[int] = None,
                             theta: float = 10000.0) -> dict:
    """Patch the pipeline's RoPE buffers with RIFLEx-modified temporal
    frequencies. Safe to call after pipeline load and before generation.

    Returns a dict describing the applied configuration, for logging.
    """
    rope = getattr(pipe.transformer, "rope", None)
    if rope is None:
        raise RuntimeError(
            "pipe.transformer.rope not found; RIFLEx expects a "
            "WanRotaryPosEmbed-compatible module."
        )

    t_dim = int(rope.t_dim)
    max_seq_len = int(rope.max_seq_len)

    if training_length is None:
        from .adapters.wan import WanAdapter
        training_length = int(WanAdapter().reference_latent_frames(pipe))

    if k is None:
        k = identify_intrinsic_k(t_dim, training_length, theta)

    info = {
        "k": k,
        "s": float(s),
        "t_dim": t_dim,
        "training_length": training_length,
        "applied": False,
    }

    # s=1 is a no-op by convention: the original pipeline already works at
    # training length. Skipping the buffer rewrite also avoids the tiny
    # numerical drift that would come from overriding freq[k] with its
    # nearby-but-not-equal value 2π/L.
    if float(s) == 1.0:
        return info

    dev = rope.freqs_cos.device
    dt = rope.freqs_cos.dtype
    new_cos, new_sin = _build_riflex_temporal_table(
        t_dim=t_dim, max_seq_len=max_seq_len, theta=theta,
        k=k, s=s, training_length=training_length,
        dtype=dt, device=dev,
    )
    with torch.no_grad():
        rope.freqs_cos[:, :t_dim].copy_(new_cos)
        rope.freqs_sin[:, :t_dim].copy_(new_sin)

    info["applied"] = True
    return info
