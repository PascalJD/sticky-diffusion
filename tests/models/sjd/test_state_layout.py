from __future__ import annotations

import pytest

from sticky.models.sjd.state_layout import (
    SUDOKU_CELL_ONLY_LAYOUT,
    SUDOKU_SLACK_LAYOUT,
    AxisSpec,
    StateLayout,
)


def test_state_layout_sudoku_slack_shapes():
    assert SUDOKU_SLACK_LAYOUT.total_site_count == 108
    assert len(SUDOKU_SLACK_LAYOUT.anchored_axes) == 1
    assert SUDOKU_SLACK_LAYOUT.anchored_axes[0].name == "cells"
    assert len(SUDOKU_SLACK_LAYOUT.unanchored_axes) == 3
    assert {a.name for a in SUDOKU_SLACK_LAYOUT.unanchored_axes} == {
        "row_slacks",
        "col_slacks",
        "box_slacks",
    }
    for a in SUDOKU_SLACK_LAYOUT.unanchored_axes:
        assert a.site_count == 9
        assert a.dynamics == "vp"
        assert a.contributes_to_nll is False


def test_cell_only_layout_is_single_axis():
    assert SUDOKU_CELL_ONLY_LAYOUT.total_site_count == 81
    assert len(SUDOKU_CELL_ONLY_LAYOUT.anchored_axes) == 1
    assert len(SUDOKU_CELL_ONLY_LAYOUT.unanchored_axes) == 0


def test_slice_offsets_match_concatenation_order():
    offsets = [SUDOKU_SLACK_LAYOUT.offset_of(a.name) for a in SUDOKU_SLACK_LAYOUT.axes]
    assert offsets == [0, 81, 90, 99]
    assert SUDOKU_SLACK_LAYOUT.slice_of("cells") == slice(0, 81)
    assert SUDOKU_SLACK_LAYOUT.slice_of("row_slacks") == slice(81, 90)
    assert SUDOKU_SLACK_LAYOUT.slice_of("col_slacks") == slice(90, 99)
    assert SUDOKU_SLACK_LAYOUT.slice_of("box_slacks") == slice(99, 108)


def test_axis_lookup_raises_for_unknown_name():
    with pytest.raises(KeyError, match="No axis named"):
        SUDOKU_SLACK_LAYOUT.axis("missing")


def test_axisspec_validates_dynamics_value():
    with pytest.raises(ValueError, match="dynamics must be one of"):
        AxisSpec(
            name="x",
            site_count=4,
            embedding_dim=2,
            anchor_table_name=None,
            dynamics="something",  # type: ignore[arg-type]
            contributes_to_nll=False,
        )


def test_axisspec_rejects_nll_without_anchors():
    with pytest.raises(ValueError, match="anchor_table_name"):
        AxisSpec(
            name="x",
            site_count=4,
            embedding_dim=2,
            anchor_table_name=None,
            dynamics="sjd",
            contributes_to_nll=True,
        )


def test_axisspec_rejects_nll_without_sjd_dynamics():
    with pytest.raises(ValueError, match="dynamics='sjd'"):
        AxisSpec(
            name="x",
            site_count=4,
            embedding_dim=2,
            anchor_table_name="simplex_vertex",
            dynamics="vp",
            contributes_to_nll=True,
        )


def test_state_layout_rejects_duplicate_axis_names():
    with pytest.raises(ValueError, match="unique names"):
        StateLayout(
            axes=(
                AxisSpec("a", 1, 1, None, "vp", False),
                AxisSpec("a", 1, 1, None, "vp", False),
            )
        )


def test_state_layout_rejects_empty_axes():
    with pytest.raises(ValueError, match="at least one axis"):
        StateLayout(axes=())
