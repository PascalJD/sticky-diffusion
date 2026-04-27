"""Generic multi-axis input projection for SJD models.

Given a `StateLayout` and a state dict `{axis_name: (B, site_count,
embedding_dim)}`, produces a single `(B, total_site_count, feature_dim)`
sequence ready for the existing `ContinuousClassifier` in sequence mode.

The module:

1. Projects each axis's per-site values via a per-axis `Dense(feature_dim)`
   named `<axis_name>_proj`.
2. Adds a per-axis site-type embedding (one row per axis, learned, init
   normal(0.02)) so the backbone can distinguish axes without relying on
   position alone.
3. Concatenates the projected per-axis tensors along the sequence axis in
   the order declared by the layout.

It is intentionally task-agnostic. Sudoku-specific structural embeddings
(row/col/box for cells, within-type group index for slacks) live in a
sibling adapter (`sudoku_structural.SudokuStructuralAdapter`) which
provides a `structural_offsets` dict that is added per-axis before concat.

Bit-equivalence note: this module's per-axis site-type embedding has one row
per axis (e.g., 4 for the SUDOKU_SLACK_LAYOUT), replacing the legacy
2-row site-type embedding from `SudokuJointInputProj`. Init values therefore
differ from the legacy module's, but the layout-aware structure makes a
single-axis (cell-only) layout reduce to one Dense + one site-type bias —
exactly the same shape and depth as a cell-only path.
"""

from __future__ import annotations

from typing import Mapping, Optional

import flax.linen as nn
import jax.numpy as jnp

from .state_layout import StateLayout


class MultiAxisInputProj(nn.Module):
    layout: StateLayout
    feature_dim: int

    @nn.compact
    def __call__(
        self,
        state: Mapping[str, jnp.ndarray],
        *,
        structural_offsets: Optional[Mapping[str, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        feat = int(self.feature_dim)
        n_axes = len(self.layout.axes)
        site_type_emb = self.param(
            "site_type_emb",
            nn.initializers.normal(stddev=0.02),
            (n_axes, feat),
        )

        pieces: list[jnp.ndarray] = []
        for i, axis in enumerate(self.layout.axes):
            if axis.name not in state:
                raise KeyError(
                    f"MultiAxisInputProj: state is missing axis {axis.name!r}; "
                    f"expected keys {[a.name for a in self.layout.axes]}."
                )
            arr = state[axis.name]
            if arr.ndim != 3:
                raise ValueError(
                    f"axis {axis.name!r} expects rank-3 input "
                    f"(B, site_count, embedding_dim); got shape {tuple(arr.shape)}."
                )
            if arr.shape[1] != axis.site_count:
                raise ValueError(
                    f"axis {axis.name!r} expects site_count={axis.site_count}; "
                    f"got {arr.shape[1]}."
                )
            if arr.shape[2] != axis.embedding_dim:
                raise ValueError(
                    f"axis {axis.name!r} expects embedding_dim={axis.embedding_dim}; "
                    f"got {arr.shape[2]}."
                )
            proj = nn.Dense(feat, name=f"{axis.name}_proj")(arr)
            proj = proj + site_type_emb[i][None, None, :]
            if structural_offsets is not None and axis.name in structural_offsets:
                offset = structural_offsets[axis.name]
                if offset.shape != (axis.site_count, feat):
                    raise ValueError(
                        f"structural_offsets[{axis.name!r}] expected shape "
                        f"({axis.site_count}, {feat}); got {tuple(offset.shape)}."
                    )
                proj = proj + offset[None, :, :]
            pieces.append(proj)

        return jnp.concatenate(pieces, axis=1)
