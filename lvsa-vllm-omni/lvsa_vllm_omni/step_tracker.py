"""Thread-local state for tracking denoising step and frame geometry.

Since vllm-omni's ``AttentionImpl.forward_cuda()`` doesn't receive step index
or frame count directly, we use thread-local storage that can be set by:

1. A hook in ``DiffusionModelRunner.execute_stepwise()`` (1-line upstream PR)
2. Manual ``set_step()`` calls from the user's serving script
3. Auto-incrementing via block counting (see ``BlockCountStepTracker``)

For frame geometry, use ``set_total_latent_frames()`` or ``LVSA_TOTAL_LATENT_FRAMES`` env var.
"""

import threading
from typing import Optional

_state = threading.local()
_global_tracker = None  # Shared BlockCountStepTracker, lazily initialized


# ── Direct setters/getters (for hooks or manual use) ─────────────────────


def set_step(step_idx: int) -> None:
    """Set the current denoising step index."""
    _state.step = step_idx


def get_step() -> int:
    """Get the current denoising step index (default 0)."""
    return getattr(_state, "step", 0)


def set_total_latent_frames(n: int) -> None:
    """Set the total number of latent frames for the current generation."""
    _state.total_latent_frames = n


def get_total_latent_frames() -> Optional[int]:
    """Get total latent frames, or None if not set."""
    return getattr(_state, "total_latent_frames", None)


def set_num_patches(p: int) -> None:
    """Set the number of patches per frame (seq_len / T_lat)."""
    _state.num_patches = p


def get_num_patches() -> Optional[int]:
    """Get patches per frame, or None if not set."""
    return getattr(_state, "num_patches", None)


def reset() -> None:
    """Clear all tracked state."""
    global _global_tracker
    for attr in ("step", "total_latent_frames", "num_patches"):
        if hasattr(_state, attr):
            delattr(_state, attr)
    _global_tracker = None
    # Also drop attention_impl's module-level step counter so its
    # auto-calibrated block count / seen-ids don't leak between generations
    # (previously caused test_step_rotation_changes_pattern to observe
    # step indices carried over from other tests in the same process).
    try:
        from . import attention_impl
        attention_impl._reset_step_counter()
    except Exception:
        pass


def get_global_tracker() -> "BlockCountStepTracker":
    """Get or create the shared BlockCountStepTracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = BlockCountStepTracker()
    return _global_tracker


def tick_self_attention() -> int:
    """Record one self-attention forward call and return current step.

    Call this once per self-attention forward_cuda() to auto-track steps.
    """
    tracker = get_global_tracker()
    return tracker.tick()


# ── Block-counting step tracker ──────────────────────────────────────────


class BlockCountStepTracker:
    """Infer denoising step index by counting forward_cuda() calls.

    In vllm-omni, each denoising step calls ``forward_cuda()`` exactly
    ``n_blocks`` times (once per transformer block for self-attention).
    After ``n_blocks`` self-attention calls, we know one step has completed.

    Usage::

        tracker = BlockCountStepTracker()

        # In forward_cuda(), for each self-attention call:
        step = tracker.tick()  # auto-increments step after n_blocks calls
    """

    def __init__(self) -> None:
        self._call_count: int = 0
        self._n_blocks: Optional[int] = None
        self._step: int = 0
        self._calibrated: bool = False

    def set_n_blocks(self, n: int) -> None:
        """Set number of transformer blocks (if known ahead of time)."""
        self._n_blocks = n
        self._calibrated = True

    def tick(self) -> int:
        """Record one self-attention forward call, return current step index.

        On the first step, counts calls to determine n_blocks automatically.
        The step increments when the count rolls over.
        """
        step = self._step
        self._call_count += 1

        if self._calibrated and self._n_blocks is not None:
            if self._call_count >= self._n_blocks:
                self._call_count = 0
                self._step += 1

        # Also update the thread-local for consistency
        _state.step = step
        return step

    def mark_step_boundary(self) -> None:
        """Called at the end of the first step to calibrate n_blocks."""
        if not self._calibrated:
            self._n_blocks = self._call_count
            self._calibrated = True
            self._call_count = 0
            self._step = 1

    @property
    def current_step(self) -> int:
        return self._step

    @property
    def n_blocks(self) -> Optional[int]:
        return self._n_blocks

    def reset(self) -> None:
        self._call_count = 0
        self._step = 0
        self._calibrated = False
        self._n_blocks = None
