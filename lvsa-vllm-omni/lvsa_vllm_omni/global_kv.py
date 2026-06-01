"""Extract global-frame K/V from full sequence tensors.

On single GPU, this is pure indexing (no communication).
On multi-GPU, this would use all-reduce — deferred to Phase 4.
"""

from typing import List, Tuple

import torch


def build_global_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    global_indices: List[int],
    num_patches: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract K/V tokens for global (anchor) frames.

    Parameters
    ----------
    key, value : [B, seq_len, H, D]
    global_indices : sorted list of global frame indices
    num_patches : tokens per frame (P)

    Returns
    -------
    k_global, v_global : [B, num_global * P, H, D]
    """
    if not global_indices:
        B, _, H, D = key.shape
        return key.new_empty(B, 0, H, D), value.new_empty(B, 0, H, D)

    P = num_patches
    device = key.device
    # Vectorized: gi*P, gi*P+1, ..., gi*P+P-1 for each gi, all on-device.
    g = torch.as_tensor(global_indices, dtype=torch.long, device=device)
    offsets = torch.arange(P, dtype=torch.long, device=device)
    idx = (g.unsqueeze(1) * P + offsets.unsqueeze(0)).flatten()
    k_global = key[:, idx]   # [B, num_global*P, H, D]
    v_global = value[:, idx]  # [B, num_global*P, H, D]
    return k_global, v_global
