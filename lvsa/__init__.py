"""LVSA — Long Video Sparse Attention for video diffusion models."""

__version__ = "1.0.0"

# Lazy imports: DistributedLVSAProcessor pulls in diffusers at module level,
# so we defer it to avoid ImportError when diffusers isn't installed (e.g. in
# test environments that only test pure-Python helpers like rope or window math).


def __getattr__(name):
    if name in ("DistributedLVSAProcessor", "WanDistributedLVSAProcessor"):
        from .lvsa_processor import DistributedLVSAProcessor, WanDistributedLVSAProcessor
        globals()["DistributedLVSAProcessor"] = DistributedLVSAProcessor
        globals()["WanDistributedLVSAProcessor"] = WanDistributedLVSAProcessor
        return globals()[name]
    if name in ("sparse_windowed_attention", "LVSAMetadata"):
        from .sparse_attention import sparse_windowed_attention, LVSAMetadata
        globals()["sparse_windowed_attention"] = sparse_windowed_attention
        globals()["LVSAMetadata"] = LVSAMetadata
        return globals()[name]
    if name == "ModelAdapter":
        from .adapters.base import ModelAdapter
        globals()["ModelAdapter"] = ModelAdapter
        return ModelAdapter
    if name in ("get_adapter", "register_adapter"):
        from .adapters import get_adapter, register_adapter
        globals()["get_adapter"] = get_adapter
        globals()["register_adapter"] = register_adapter
        return globals()[name]
    raise AttributeError(f"module 'lvsa' has no attribute {name!r}")


__all__ = [
    "DistributedLVSAProcessor",
    "WanDistributedLVSAProcessor",
    "sparse_windowed_attention",
    "LVSAMetadata",
    "ModelAdapter",
    "get_adapter",
    "register_adapter",
]
