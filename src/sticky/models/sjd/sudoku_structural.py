"""Sudoku-specific structural embeddings for the multi-axis input projection.

Provides a Flax module that materializes learned (row/col/box) and
(within-type group-index) structural embeddings as a `structural_offsets`
dict consumable by `MultiAxisInputProj`. The dict has shape
`{axis_name: (site_count, feature_dim)}` and is added per-axis before
concatenation.

  * cells       : row + col + box embedding indexed by the cell's coordinates.
  * row_slacks  : group_idx_emb (which row this slack tracks; 0..8).
  * col_slacks  : group_idx_emb (which column).
  * box_slacks  : group_idx_emb (which 3x3 box).

The 3-way distinction between row/col/box slack axes is already provided
by `MultiAxisInputProj`'s per-axis site-type embedding (4 rows for the
SUDOKU_SLACK_LAYOUT). This adapter therefore reuses a single 9-row
`group_idx_emb` table across the three slack axes — equivalent in
expressivity to keeping three separate tables and one less parameter
matrix.

All structural embeddings are zero-initialized so the forward output at
initialization equals the forward output of `MultiAxisInputProj` alone (no
structural offsets). They diverge only once they receive gradient.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from .joint_input import (
    CELL_BOX_IDX,
    CELL_COL_IDX,
    CELL_ROW_IDX,
)


class SudokuStructuralAdapter(nn.Module):
    feature_dim: int

    @nn.compact
    def __call__(self) -> dict[str, jnp.ndarray]:
        feat = int(self.feature_dim)
        zero = nn.initializers.zeros
        row_emb = self.param("row_emb", zero, (9, feat))
        col_emb = self.param("col_emb", zero, (9, feat))
        box_emb = self.param("box_emb", zero, (9, feat))
        group_idx_emb = self.param("group_idx_emb", zero, (9, feat))

        cell_struct = (
            row_emb[CELL_ROW_IDX]
            + col_emb[CELL_COL_IDX]
            + box_emb[CELL_BOX_IDX]
        )  # (81, feat)
        return {
            "cells": cell_struct,
            "row_slacks": group_idx_emb,
            "col_slacks": group_idx_emb,
            "box_slacks": group_idx_emb,
        }
