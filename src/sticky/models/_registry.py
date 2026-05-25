"""Model-init registry.

Each model family registers its init function under ``cfg.model.name``. The
training entry point (``init_state`` in ``src/sticky/training/state.py``) reads
from the registry instead of dispatching on the name with if/elif.

Adding a new family is a matter of creating a module under
``sticky/models/factories/`` and decorating its init function with
``@register_init("<name>")``. Importing ``sticky.models.factories`` triggers all
registrations.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

from omegaconf import DictConfig


# Init function signature: (model, cfg, rng) -> variables_dict
InitFn = Callable[[Any, DictConfig, Any], dict]

_INIT_REGISTRY: Dict[str, InitFn] = {}


def register_init(name: str) -> Callable[[InitFn], InitFn]:
    """Decorator: register the init function for a model family."""
    def deco(fn: InitFn) -> InitFn:
        if name in _INIT_REGISTRY:
            raise ValueError(f"Init for {name!r} already registered.")
        _INIT_REGISTRY[name] = fn
        return fn
    return deco


def get_init(name: str) -> InitFn:
    """Lookup the init function for ``name``; raises if missing."""
    if name not in _INIT_REGISTRY:
        raise ValueError(
            f"No init registered for model.name={name!r}. "
            f"Known: {sorted(_INIT_REGISTRY)}"
        )
    return _INIT_REGISTRY[name]
