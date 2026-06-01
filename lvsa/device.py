"""Device-agnostic helpers for example scripts.

LVSA's example launchers used to hard-code CUDA. This module centralises
device selection so the same scripts work on CUDA, Ascend NPU, or CPU.

Preference order: CUDA > Ascend NPU > CPU.
"""

from __future__ import annotations

import torch


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return getattr(torch, "npu", None) is not None and torch.npu.is_available()


def get_device(rank: int = 0) -> torch.device:
    """Pick the best available device, set it as the current device, return it."""
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        return torch.device("cuda", rank)
    if _npu_available():
        torch.npu.set_device(rank)
        return torch.device("npu", rank)
    return torch.device("cpu")


def get_distributed_backend() -> str:
    """Collective backend matching the active hardware."""
    if torch.cuda.is_available():
        return "nccl"
    if _npu_available():
        return "hccl"
    return "gloo"


def enable_fast_matmul() -> None:
    """Enable TF32 on CUDA (no-op elsewhere)."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True


def max_memory_allocated(device: int | None = None) -> int:
    """Peak allocated bytes on the active accelerator. ``0`` on CPU-only hosts."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device)
    if _npu_available():
        return torch.npu.max_memory_allocated(device)
    return 0


def device_count() -> int:
    """Number of accelerator devices. ``0`` on CPU-only hosts."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    if _npu_available():
        return torch.npu.device_count()
    return 0


def mem_get_info(device: int | torch.device | None = None) -> tuple[int, int]:
    """Return ``(free_bytes, total_bytes)`` for the active accelerator.
    ``(0, 0)`` on CPU-only hosts.
    """
    if torch.cuda.is_available():
        return torch.cuda.mem_get_info(device)
    if _npu_available():
        return torch.npu.mem_get_info(device)
    return 0, 0


def memory_stats() -> tuple[str, int, float, float, float] | None:
    """Return ``(kind, device_idx, alloc_GB, reserved_GB, peak_GB)`` for
    the active accelerator, or ``None`` if neither CUDA nor NPU is available.

    Same numbers, same units across CUDA and Ascend NPU. Use this anywhere
    you'd otherwise call ``torch.cuda.memory_*`` directly — those raise on
    Ascend.
    """
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        return (
            "cuda", dev,
            torch.cuda.memory_allocated(dev) / 1e9,
            torch.cuda.memory_reserved(dev) / 1e9,
            torch.cuda.max_memory_allocated(dev) / 1e9,
        )
    if _npu_available():
        dev = torch.npu.current_device()
        return (
            "npu", dev,
            torch.npu.memory_allocated(dev) / 1e9,
            torch.npu.memory_reserved(dev) / 1e9,
            torch.npu.max_memory_allocated(dev) / 1e9,
        )
    return None
