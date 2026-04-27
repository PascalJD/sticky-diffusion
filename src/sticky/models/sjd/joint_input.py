"""Joint input projection for the slack-augmented Sudoku SJD task.

Projects (cell_x_t, slack_x_t) into a single (B, 108, feature_dim) sequence
that the existing ContinuousClassifier can consume in sequence mode.

In addition to the per-axis Dense projection (cell_proj, slack_proj) and the
2-row site-type embedding (cell vs slack), this module adds geometry-aware
structural embeddings that bake the Sudoku adjacency into the pre-attention
representation:

  * Cell tokens get learned `row_emb[9, d]`, `col_emb[9, d]`, `box_emb[9, d]`
    indexed by the cell's (row, col, box) coordinates.
  * Slack tokens get learned `group_type_emb[3, d]` (0=row, 1=col, 2=box) and
    `group_idx_emb[9, d]` indexed by the within-type group index.

All structural embeddings are zero-initialized so the forward output is
bit-equivalent to the previous (no-structural-embedding) joint input projection
at initialization. They diverge only once they receive gradient, which lets
us do same-seed regression checks against the pre-augmentation behavior.

The position embedding for the joint sequence is provided by the GPT-2-like
backbone's own learned `pos_embed` (length 108).
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


# Pre-computed structural indices for Sudoku cells, in row-major order.
# row_idx[c]  in 0..8 = c // 9
# col_idx[c]  in 0..8 = c % 9
# box_idx[c]  in 0..8 = 3 * (row // 3) + (col // 3)
def _cell_row_indices() -> jnp.ndarray:
    return jnp.arange(81, dtype=jnp.int32) // 9


def _cell_col_indices() -> jnp.ndarray:
    return jnp.arange(81, dtype=jnp.int32) % 9


def _cell_box_indices() -> jnp.ndarray:
    rows = jnp.arange(81, dtype=jnp.int32) // 9
    cols = jnp.arange(81, dtype=jnp.int32) % 9
    return 3 * (rows // 3) + (cols // 3)


# Slack ordering: 0..8 rows, 9..17 cols, 18..26 boxes.
def _slack_group_type_indices() -> jnp.ndarray:
    arr = jnp.zeros(27, dtype=jnp.int32)
    arr = arr.at[9:18].set(1)
    arr = arr.at[18:27].set(2)
    return arr


def _slack_group_idx_indices() -> jnp.ndarray:
    return jnp.tile(jnp.arange(9, dtype=jnp.int32), 3)


CELL_ROW_IDX = _cell_row_indices()
CELL_COL_IDX = _cell_col_indices()
CELL_BOX_IDX = _cell_box_indices()
SLACK_GROUP_TYPE_IDX = _slack_group_type_indices()
SLACK_GROUP_IDX = _slack_group_idx_indices()


class SudokuJointInputProj(nn.Module):
    feature_dim: int

    @nn.compact
    def __call__(self, cell_x_t: jnp.ndarray, slack_x_t: jnp.ndarray) -> jnp.ndarray:
        if cell_x_t.ndim != 3 or slack_x_t.ndim != 3:
            raise ValueError(
                "SudokuJointInputProj expects rank-3 inputs (B, S, d); got "
                f"cell_x_t.ndim={cell_x_t.ndim}, slack_x_t.ndim={slack_x_t.ndim}."
            )
        if cell_x_t.shape[0] != slack_x_t.shape[0]:
            raise ValueError(
                "cell_x_t and slack_x_t must share batch dim, got "
                f"{cell_x_t.shape[0]} vs {slack_x_t.shape[0]}."
            )
        if cell_x_t.shape[1] != 81:
            raise ValueError(
                f"cell_x_t must have 81 sites; got {cell_x_t.shape[1]}."
            )
        if slack_x_t.shape[1] != 27:
            raise ValueError(
                f"slack_x_t must have 27 sites; got {slack_x_t.shape[1]}."
            )

        feat = int(self.feature_dim)
        cell_proj = nn.Dense(feat, name="cell_proj")(cell_x_t)
        slack_proj = nn.Dense(feat, name="slack_proj")(slack_x_t)

        site_type_emb = self.param(
            "site_type_emb",
            nn.initializers.normal(stddev=0.02),
            (2, feat),
        )
        cell_proj = cell_proj + site_type_emb[0][None, None, :]
        slack_proj = slack_proj + site_type_emb[1][None, None, :]

        # Structural embeddings (init-zero so init-time forward is identical).
        zero = nn.initializers.zeros
        row_emb = self.param("row_emb", zero, (9, feat))
        col_emb = self.param("col_emb", zero, (9, feat))
        box_emb = self.param("box_emb", zero, (9, feat))
        group_type_emb = self.param("group_type_emb", zero, (3, feat))
        group_idx_emb = self.param("group_idx_emb", zero, (9, feat))

        cell_struct = (
            row_emb[CELL_ROW_IDX]
            + col_emb[CELL_COL_IDX]
            + box_emb[CELL_BOX_IDX]
        )  # (81, feat)
        slack_struct = (
            group_type_emb[SLACK_GROUP_TYPE_IDX]
            + group_idx_emb[SLACK_GROUP_IDX]
        )  # (27, feat)
        cell_proj = cell_proj + cell_struct[None, :, :]
        slack_proj = slack_proj + slack_struct[None, :, :]

        return jnp.concatenate([cell_proj, slack_proj], axis=1)
