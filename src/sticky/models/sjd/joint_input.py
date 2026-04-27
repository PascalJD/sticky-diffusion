"""Joint input projection for the slack-augmented Sudoku SJD task.

Projects (cell_x_t, slack_x_t) into a single (B, 108, feature_dim) sequence
that the existing ContinuousClassifier can consume in sequence mode. A
learned 2-row site-type embedding distinguishes cell rows from slack rows;
per-position information is provided by the backbone's own learned
pos_embed (the GPT2-like backbone allocates one of length sequence_max_length).
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


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

        cell_proj = nn.Dense(int(self.feature_dim), name="cell_proj")(cell_x_t)
        slack_proj = nn.Dense(int(self.feature_dim), name="slack_proj")(slack_x_t)

        site_type_emb = self.param(
            "site_type_emb",
            nn.initializers.normal(stddev=0.02),
            (2, int(self.feature_dim)),
        )
        cell_proj = cell_proj + site_type_emb[0][None, None, :]
        slack_proj = slack_proj + site_type_emb[1][None, None, :]

        return jnp.concatenate([cell_proj, slack_proj], axis=1)
