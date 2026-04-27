"""State-space partitioning for SJD models.

Today the SJD code path treats the state as a single tensor of anchored sites
(every site participates in the SJD forward kernel and contributes to the
NLL). The slack-augmented Sudoku task adds a second class of site that has
no anchor table and follows a pure VP-SDE — the joint state is the
concatenation of *anchored* and *unanchored* axes. This module surfaces that
partitioning as data so callers can declare the layout once and have the
input projection, forward sampling, and loss adapt automatically.

A `StateLayout` is an ordered tuple of `AxisSpec`s. Each `AxisSpec` carries:

  * `name`               — string identifier used as the key in the state dict
                           and in per-axis metric prefixes.
  * `site_count`         — number of sites along this axis (e.g., 81 for cells,
                           9 for row slacks).
  * `embedding_dim`      — feature width of the per-site continuous state
                           (must equal anchor_dim when anchored).
  * `anchor_table_name`  — name of the registered anchor family, or None for
                           unanchored axes.
  * `dynamics`           — forward-law dial:
                             "sjd"           — paired-corruption from
                                                models/sjd/corruption.py.
                             "vp"            — pure VP perturbation
                                                (vp_perturb), the current slack
                                                behavior.
                             "deterministic" — value at t > 0 is computed from
                                                other axes via a per-layout
                                                callback. Stubbed in this PR;
                                                see the Phase D TODO.
  * `contributes_to_nll` — True iff this axis enters the NLL term of the loss.
                           Anchored axes always do; unanchored axes never do.

The Sudoku slack experiment is the reference layout (`SUDOKU_SLACK_LAYOUT`);
the cell-only Sudoku layout is a one-axis projection of it. Single-axis tasks
(CIFAR / ImageNet64 / OpenWebText) can wrap their existing data shape in a
one-axis layout if they want to opt into the new abstraction. Migration of
those tasks is intentionally out of scope for this PR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


Dynamics = Literal["sjd", "vp", "deterministic"]
_VALID_DYNAMICS: Tuple[Dynamics, ...] = ("sjd", "vp", "deterministic")


@dataclass(frozen=True)
class AxisSpec:
    name: str
    site_count: int
    embedding_dim: int
    anchor_table_name: Optional[str]
    dynamics: Dynamics
    contributes_to_nll: bool

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("AxisSpec.name must be a non-empty string.")
        if self.site_count <= 0:
            raise ValueError(
                f"AxisSpec.site_count must be positive; got {self.site_count}."
            )
        if self.embedding_dim <= 0:
            raise ValueError(
                f"AxisSpec.embedding_dim must be positive; got {self.embedding_dim}."
            )
        if self.dynamics not in _VALID_DYNAMICS:
            raise ValueError(
                f"AxisSpec.dynamics must be one of {_VALID_DYNAMICS}; "
                f"got {self.dynamics!r}."
            )
        if self.contributes_to_nll and self.anchor_table_name is None:
            raise ValueError(
                f"AxisSpec({self.name}): contributes_to_nll=True requires "
                "an anchor_table_name (NLL is computed against anchors)."
            )
        if self.contributes_to_nll and self.dynamics != "sjd":
            raise ValueError(
                f"AxisSpec({self.name}): contributes_to_nll=True is only "
                "supported with dynamics='sjd'."
            )


@dataclass(frozen=True)
class StateLayout:
    axes: Tuple[AxisSpec, ...]

    def __post_init__(self) -> None:
        if len(self.axes) == 0:
            raise ValueError("StateLayout must have at least one axis.")
        names = [a.name for a in self.axes]
        if len(set(names)) != len(names):
            raise ValueError(
                f"StateLayout axes must have unique names; got {names}."
            )

    @property
    def total_site_count(self) -> int:
        return sum(a.site_count for a in self.axes)

    @property
    def anchored_axes(self) -> Tuple[AxisSpec, ...]:
        return tuple(a for a in self.axes if a.anchor_table_name is not None)

    @property
    def unanchored_axes(self) -> Tuple[AxisSpec, ...]:
        return tuple(a for a in self.axes if a.anchor_table_name is None)

    def axis(self, name: str) -> AxisSpec:
        for a in self.axes:
            if a.name == name:
                return a
        raise KeyError(f"No axis named {name!r} in layout (have {[a.name for a in self.axes]}).")

    def offset_of(self, name: str) -> int:
        """Cumulative site offset of `name` in the concatenated sequence."""
        offset = 0
        for a in self.axes:
            if a.name == name:
                return offset
            offset += a.site_count
        raise KeyError(f"No axis named {name!r} in layout.")

    def slice_of(self, name: str) -> slice:
        start = self.offset_of(name)
        return slice(start, start + self.axis(name).site_count)


# --- Reference layouts -----------------------------------------------------


SUDOKU_SLACK_LAYOUT = StateLayout(
    axes=(
        AxisSpec(
            name="cells",
            site_count=81,
            embedding_dim=9,
            anchor_table_name="simplex_vertex",
            dynamics="sjd",
            contributes_to_nll=True,
        ),
        AxisSpec(
            name="row_slacks",
            site_count=9,
            embedding_dim=9,
            anchor_table_name=None,
            dynamics="vp",
            contributes_to_nll=False,
        ),
        AxisSpec(
            name="col_slacks",
            site_count=9,
            embedding_dim=9,
            anchor_table_name=None,
            dynamics="vp",
            contributes_to_nll=False,
        ),
        AxisSpec(
            name="box_slacks",
            site_count=9,
            embedding_dim=9,
            anchor_table_name=None,
            dynamics="vp",
            contributes_to_nll=False,
        ),
    )
)


SUDOKU_CELL_ONLY_LAYOUT = StateLayout(
    axes=(
        AxisSpec(
            name="cells",
            site_count=81,
            embedding_dim=9,
            anchor_table_name="simplex_vertex",
            dynamics="sjd",
            contributes_to_nll=True,
        ),
    )
)
