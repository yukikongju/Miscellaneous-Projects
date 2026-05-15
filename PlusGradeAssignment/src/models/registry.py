"""
Lightweight decorator-based model registry.

Usage:
    from models.registry import MODEL_REGISTRY, register

    @register(MODEL_REGISTRY, "my_model")
    class MyModel: ...

    cls = MODEL_REGISTRY["my_model"]

Caveats:
    - Registration happens at import time; all model modules must be imported
      before calling MODEL_REGISTRY lookups (models.py handles this).
"""

from typing import Any

MODEL_REGISTRY: dict[str, type] = {}


def register(registry: dict, name: str):
    """Decorator that registers a class under a given name."""

    def decorator(cls):
        registry[name] = cls
        return cls

    return decorator
