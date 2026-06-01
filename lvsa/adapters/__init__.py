"""
adapters — Model-specific bridges for the LVSA LVSA engine.

Each adapter implements :class:`ModelAdapter` to translate a diffusion
model's attention internals into the generic interface expected by
:class:`DistributedLVSAProcessor`.
"""

from importlib import import_module
from typing import Dict, Type

from .base import ModelAdapter

# Lazy registry: module_path, class_name
_ADAPTER_REGISTRY: Dict[str, tuple] = {
    "wan": ("lvsa.adapters.wan", "WanAdapter"),
    "hunyuan_video": ("lvsa.adapters.hunyuan_video", "HunyuanVideoAdapter"),
    "cogvideox": ("lvsa.adapters.cogvideox", "CogVideoXAdapter"),
}


def get_adapter(name: str) -> ModelAdapter:
    """Instantiate an adapter by name.

    Parameters
    ----------
    name : str
        One of the registered adapter names (e.g. ``"wan"``, ``"hunyuan_video"``).

    Returns
    -------
    ModelAdapter
        An instance of the requested adapter.

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    """
    if name not in _ADAPTER_REGISTRY:
        available = ", ".join(sorted(_ADAPTER_REGISTRY))
        raise KeyError(
            f"Unknown adapter {name!r}. Available adapters: {available}"
        )
    module_path, class_name = _ADAPTER_REGISTRY[name]
    module = import_module(module_path)
    cls: Type[ModelAdapter] = getattr(module, class_name)
    return cls()


def register_adapter(name: str, module_path: str, class_name: str) -> None:
    """Register a custom adapter for use with :func:`get_adapter`.

    Parameters
    ----------
    name : str
        Short name for the adapter (e.g. ``"my_model"``).
    module_path : str
        Fully qualified module path (e.g. ``"my_package.adapters.my_model"``).
    class_name : str
        Class name within the module (e.g. ``"MyModelAdapter"``).
    """
    _ADAPTER_REGISTRY[name] = (module_path, class_name)


__all__ = ["ModelAdapter", "get_adapter", "register_adapter"]
