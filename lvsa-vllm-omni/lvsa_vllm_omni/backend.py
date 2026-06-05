"""LVSA AttentionBackend for vllm-omni.

``register_lvsa_backend()`` adds ``LVSA`` to vllm-omni's
``DiffusionAttentionBackendEnum``. Select it per attention role via the 0.22
AttentionConfig, e.g. on the CLI::

    --diffusion-attention-config '{"per_role": {"self": {"backend": "LVSA"}}}'

or through the Python API::

    Omni(model=..., diffusion_attention_config={"per_role": {"self": {"backend": "LVSA"}}})

DiT self-attention then uses sparse windowed attention while cross-attention
falls back to dense SDPA (different q/kv sequence lengths are detected and
routed to the dense path automatically).
"""

from typing import List, Type

from .attention_impl import LVSAAttentionImpl


class LVSABackend:
    """Sparse windowed attention backend for video DiTs.

    Conforms to vllm-omni's ``AttentionBackend`` interface.  The actual ABC
    import is deferred to avoid hard dependency on vllm-omni at import time —
    this module works standalone for testing.
    """

    @staticmethod
    def get_name() -> str:
        return "LVSA"

    @staticmethod
    def get_impl_cls() -> Type:
        return LVSAAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type:
        # Try to import vllm-omni's base metadata; fall back to a stub
        try:
            from vllm_omni.diffusion.attention.backends.abstract import (
                AttentionMetadata,
            )
            return AttentionMetadata
        except ImportError:
            # vllm-omni not installed — return a placeholder for testing
            return type("AttentionMetadata", (), {})

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [64, 96, 128, 192, 256]
