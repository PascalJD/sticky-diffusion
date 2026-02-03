# src/sticky/entrypoints/viz_utils.py
from __future__ import annotations

from typing import Tuple

import numpy as np


def make_grid(
    imgs_hw: np.ndarray, cols: int = 8, pad: int = 2, pad_value: float = 1.0
) -> np.ndarray:
    """
    imgs_hw: (B,H,W) float in [0,1]
    returns: (H_grid, W_grid) float32 in [0,1]
    """
    imgs = np.asarray(imgs_hw)
    if imgs.ndim != 3:
        raise ValueError(f"imgs_hw must be (B,H,W); got {imgs.shape}")

    b, h, w = imgs.shape
    cols = max(1, int(cols))
    rows = int(np.ceil(b / cols))
    grid_h = rows * h + (rows - 1) * pad
    grid_w = cols * w + (cols - 1) * pad

    grid = np.full((grid_h, grid_w), pad_value, dtype=np.float32)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= b:
                break
            y0 = r * (h + pad)
            x0 = c * (w + pad)
            grid[y0:y0 + h, x0:x0 + w] = imgs[idx]
            idx += 1
    return grid


def to_uint8(img_01: np.ndarray) -> np.ndarray:
    """Convert float image in [0,1] to uint8 [0,255]."""
    img = np.asarray(img_01, dtype=np.float32)
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
