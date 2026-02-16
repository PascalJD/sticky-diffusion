from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import jax
from omegaconf import DictConfig, OmegaConf

try:
    import numpy as np
except Exception:
    np = None


def numpy_available() -> bool:
    return np is not None


def to_numpy(x: Any):
    """Best-effort conversion of JAX arrays to NumPy."""
    if np is None:
        raise RuntimeError("NumPy not available but required for W&B image logging.")
    try:
        x = jax.device_get(x)
    except Exception:
        pass
    return np.asarray(x)


def make_image_grid_np(images: "np.ndarray", max_images: int = 64) -> "np.ndarray":
    """Build a single HxWxC uint8 grid from a batch (N,H,W,C)."""
    assert images.ndim == 4, f"Expected (N,H,W,C), got {images.shape}"
    imgs = images[:max_images]
    n = imgs.shape[0]
    g = int(np.floor(np.sqrt(n)))
    g = max(1, g)
    imgs = imgs[: g * g]

    imgs = imgs.reshape(g, g, imgs.shape[1], imgs.shape[2], imgs.shape[3])
    rows = [np.concatenate(list(imgs[r]), axis=1) for r in range(g)]
    grid = np.concatenate(rows, axis=0)
    return grid.astype(np.uint8)


def is_scalar_array(x: Any) -> bool:
    if hasattr(x, "shape"):
        try:
            return (x.shape == ()) or (getattr(x, "size", 0) == 1)
        except Exception:
            return False
    return False


def to_py_scalar(x: Any) -> Optional[float]:
    try:
        x = jax.device_get(x)
    except Exception:
        pass

    if isinstance(x, (int, float)):
        return float(x)

    if is_scalar_array(x):
        try:
            return float(x)
        except Exception:
            pass
        try:
            if np is not None:
                return float(np.asarray(x).reshape(()))
        except Exception:
            return None

    return None


def sanitize_metrics(metrics: Mapping[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        key = f"{prefix}{k}" if prefix == "" else f"{prefix}/{k}"
        if isinstance(v, dict):
            out.update(sanitize_metrics(v, prefix=key))
        else:
            s = to_py_scalar(v)
            if s is not None:
                out[key] = s
    return out


def maybe_init_wandb(full_cfg: DictConfig):
    wb = full_cfg.get("wandb", None)
    if wb is None or not bool(wb.get("enabled", False)):
        return None

    if jax.process_index() != 0:
        return None

    mode = str(wb.get("mode", "online"))
    if mode == "disabled":
        return None

    try:
        import wandb
    except ImportError as e:
        raise ImportError(
            "Weights & Biases is enabled in config, but wandb isn't installed. "
            "Install with `pip install wandb` (or add it to environment.yml)."
        ) from e

    entity = wb.get("entity", None)
    if entity in (None, "null"):
        entity = None

    run_name = wb.get("run_name", None)
    if run_name in (None, "", "null"):
        run_name = f"{full_cfg.experiment.task.name}_{full_cfg.experiment.model.name}"

    tags = wb.get("tags", None)
    tags = list(tags) if tags is not None else None

    resolved = OmegaConf.to_container(full_cfg, resolve=True)
    wandb.init(
        project=str(wb.project),
        entity=entity,
        name=str(run_name),
        tags=tags,
        mode=mode,
        config=resolved,
    )

    wandb.config.update(
        {
            "jax_device_count": jax.device_count(),
            "jax_platform": jax.default_backend(),
        },
        allow_val_change=True,
    )
    return wandb


def log_images_to_wandb(
    *,
    wandb_mod,
    step_i: int,
    gt_images: Any,
    max_images: int,
    samples: Any = None,
):
    if wandb_mod is None:
        return

    gt_np = np.clip(to_numpy(gt_images), 0, 255).astype(np.uint8)
    gt_grid = make_image_grid_np(gt_np, max_images=max_images)
    log_payload = {
        "images/ground_truth": wandb_mod.Image(gt_grid, caption="GT"),
    }

    if samples is not None:
        samp_np = np.clip(to_numpy(samples), 0, 255).astype(np.uint8)
        samp_grid = make_image_grid_np(samp_np, max_images=max_images)
        log_payload["images/samples"] = wandb_mod.Image(samp_grid, caption="Samples")

    wandb_mod.log(log_payload, step=step_i)
