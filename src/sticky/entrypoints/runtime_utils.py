# src/sticky/entrypoints/runtime_utils.py
from __future__ import annotations

import os
from typing import Optional

from omegaconf import DictConfig, OmegaConf


def configure_runtime(cfg: DictConfig) -> None:
    """
    Set environment flags BEFORE importing JAX or any module that imports JAX.
    """
    dev = str(cfg.runtime.device).lower()
    if dev in ("cpu", "gpu"):
        os.environ.setdefault("JAX_PLATFORMS", dev)

    if cfg.runtime.xla_preallocate is False:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    if cfg.runtime.xla_mem_fraction is not None:
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(cfg.runtime.xla_mem_fraction))


def maybe_init_wandb(cfg: DictConfig):
    """Initialize wandb if enabled. Returns the run or None."""
    if not bool(cfg.wandb.enabled):
        return None

    import wandb
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        tags=list(cfg.wandb.get("tags", [])),
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    return run


def save_flax_params(params, out_dir: str, step: int, prefix: str = "sticky_step") -> str:
    """
    Save Flax params to {out_dir}/checkpoints/{prefix}{step}.params.
    Lazily imports flax to avoid JAX import before configure_runtime.
    """
    from flax.serialization import to_bytes

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{prefix}{step}.params")
    with open(path, "wb") as f:
        f.write(to_bytes(params))
    return path
