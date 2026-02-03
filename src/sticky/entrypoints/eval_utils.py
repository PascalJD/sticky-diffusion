# src/sticky/entrypoints/eval_utils.py
from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from sticky.entrypoints.viz_utils import make_grid, to_uint8


def log_mnist_eval_images(
    *,
    step: int,
    key,
    state,
    anchors,
    beta,
    hazard,
    jump,
    sampler_cfg,
    num_samples: int,
    grid_cols: int,
    batch_np: np.ndarray,
    wb_run: Any,
    shape_hw: Tuple[int, int] = (28, 28),
) -> None:
    """
    Samples from the reverse sampler and logs:
      - samples grid
      - GT grid
      - quantized GT grid
      - sampler metrics
    """
    if wb_run is None:
        return

    import wandb
    from sticky.core.sampler import reverse_sample

    H, W = shape_hw

    res = reverse_sample(
        key,
        params=state.params,
        apply_model=state.apply_fn,
        anchors=anchors,
        beta=beta,
        hazard=hazard,
        jump=jump,
        shape_hw=(H, W),
        B=int(num_samples),
        cfg=sampler_cfg,
    )

    # Use k_filled for visualization (safe even if force_classify_at_end=False)
    k_final = np.asarray(res.k_filled)
    samples_01 = (k_final.astype(np.float32) + 0.5) / float(anchors.L)
    samples_grid = make_grid(samples_01, cols=int(grid_cols), pad=2)

    b = min(int(batch_np.shape[0]), int(num_samples))
    gt = batch_np[:b].reshape(b, H, W)
    gt_grid = make_grid(np.asarray(gt), cols=int(grid_cols), pad=2)

    # Quantize GT to L bins (bin centers)
    k_gt = np.clip(np.floor(gt * float(anchors.L)).astype(np.int32), 0, anchors.L - 1)
    gtq_01 = (k_gt.astype(np.float32) + 0.5) / float(anchors.L)
    gtq_grid = make_grid(gtq_01, cols=int(grid_cols), pad=2)

    sm = {k: float(np.asarray(v)) for k, v in res.metrics.items()}

    wandb.log(
        {
            "samples/grid": wandb.Image(to_uint8(samples_grid), caption="SJD samples"),
            "examples/gt_grid": wandb.Image(to_uint8(gt_grid), caption="MNIST ground truth"),
            "examples/gt_quantized_grid": wandb.Image(
                to_uint8(gtq_grid), caption=f"Quantized (L={anchors.L})"
            ),
            **{f"samples/{k}": v for k, v in sm.items()},
        },
        step=int(step),
    )
