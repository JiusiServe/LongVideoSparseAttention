"""Extract global-frame K/V from full sequence tensors.

The single-GPU primitive (pure indexing, no communication) now lives in the LVSA
core as ``lvsa.sparse_attention.build_global_kv``. This module re-exports it so the
existing ``from lvsa_vllm_omni.global_kv import build_global_kv`` imports (and the
plugin tests) keep working without a duplicate implementation.

Multi-GPU global gather (all-reduce, ``global_broadcast`` mode) is a separate path:
``DistributedLVSAProcessor._build_global_kv`` in the core.
"""

from lvsa.sparse_attention import build_global_kv

__all__ = ["build_global_kv"]
