# src/sticky/models/factory.py
"""Model factory dispatcher.

Per-family build functions live in ``sticky.models.factories.<family>``.
The dispatch table is composed in ``sticky.models.factories.__init__``.
"""
from __future__ import annotations

from typing import Any, Callable

from omegaconf import DictConfig

from sticky.models.factories import (
    MODEL_BUILDERS,
    bitdiff,
    cadd,
    candi,
    d3pm,
    ddpm,
    md4,
    mdlm,
    mdm,
    sjd,
)

__all__ = ["MODEL_BUILDERS", "build_model"]


def build_model(
    cfg: DictConfig, *, data_shape: tuple[int, ...], vocab_size: int
) -> Any:
    name = str(cfg.model.name)
    builder = MODEL_BUILDERS.get(name)
    if builder is None:
        raise ValueError(
            f"Unknown model.name={name!r}. Known: {sorted(MODEL_BUILDERS)}"
        )
    return builder(cfg, data_shape=data_shape, vocab_size=vocab_size)
