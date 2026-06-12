"""Clue-count-stratified sudoku eval metrics (lo <= 23, mid == 24, hi >= 25).

Covers:
  - `_evaluate_board_batch_counts`: boards with 22/24/26 clues land in the
    right bucket with hand-built expected counts; bucket boundaries (23, 25)
    and out-of-range counts (19, 30) fall into the open-ended lo/hi buckets;
    the three buckets exactly partition the unstratified totals.
  - `_base_totals`: registers all 12 stratified keys (regression for the
    KeyError-on-accumulate landmine in the eval loop's `totals[key] += value`).
  - `_finalize_metrics`: emits the 9 stratified metrics with exact ratios,
    including the zero-denominator convention (_safe_ratio -> 0.0).
  - CSV round-trip: the new columns survive `_normalize_csv_row` (regression
    for the silent-drop landmine in the closed column whitelist), and the
    `blur_score` column reflects `blur_score_effective` (the effective state),
    not the requested `blur_score` flag.

CPU-only, numpy-level; no sampler or GPU required.
"""
from __future__ import annotations

import numpy as np

from sticky.eval.sudoku import (
    _base_totals,
    _evaluate_board_batch_counts,
    _finalize_metrics,
    _sjd_run_row,
)
from sticky.training.csv_report import (
    SUDOKU_SJD_PROGRESS_COLUMNS,
    _normalize_csv_row,
)


BUCKETS = ("lo", "mid", "hi")
STRATIFIED_COUNT_KEYS = [
    f"{stem}_clues_{bucket}"
    for bucket in BUCKETS
    for stem in (
        "num_examples",
        "board_exact",
        "unknown_cell_correct",
        "unknown_cell_total",
    )
]


def _solved_board_flat() -> np.ndarray:
    """A canonical valid solved sudoku, flattened to (81,)."""
    rows = []
    for r in range(9):
        shift = (r * 3 + r // 3) % 9
        rows.append([(shift + c) % 9 + 1 for c in range(9)])
    board = np.asarray(rows, dtype=np.int32).reshape(-1)
    assert board.shape == (81,)
    return board


def _wrong_digit(value: int) -> int:
    """A digit in 1..9 guaranteed different from `value`."""
    return value % 9 + 1


def _make_batch(clue_counts: list[int], wrong_unknown_cells: list[list[int]]):
    """Boards sharing one solution; board b has clue_counts[b] leading clues
    and its prediction corrupted at the given unknown-cell indices."""
    solution = _solved_board_flat()
    batch = len(clue_counts)
    solution_board = np.tile(solution, (batch, 1))
    clue_mask = np.zeros((batch, 81), dtype=bool)
    for b, n_clues in enumerate(clue_counts):
        clue_mask[b, :n_clues] = True
    clue_board = np.where(clue_mask, solution_board, 0).astype(np.int32)
    pred_board = solution_board.copy()
    for b, cells in enumerate(wrong_unknown_cells):
        for cell in cells:
            assert not clue_mask[b, cell], "test bug: corrupting a clue cell"
            pred_board[b, cell] = _wrong_digit(int(solution[cell]))
    return pred_board, solution_board, clue_board, clue_mask


# ---------------------------------------------------------------------------
# _evaluate_board_batch_counts: bucket membership + exact counts
# ---------------------------------------------------------------------------

def test_counts_boards_land_in_correct_buckets_with_exact_counts():
    # Board 0: 22 clues (lo), exact (0 wrong unknown cells).
    # Board 1: 24 clues (mid), 2 wrong unknown cells.
    # Board 2: 26 clues (hi), 5 wrong unknown cells.
    pred, sol, clue_board, clue_mask = _make_batch(
        clue_counts=[22, 24, 26],
        wrong_unknown_cells=[[], [30, 40], [40, 50, 60, 70, 80]],
    )
    counts = _evaluate_board_batch_counts(
        pred_board=pred,
        solution_board=sol,
        clue_board=clue_board,
        clue_mask=clue_mask,
    )

    # Hand-built expectations (unknown totals are 81 - clue_count).
    assert counts["num_examples_clues_lo"] == 1
    assert counts["num_examples_clues_mid"] == 1
    assert counts["num_examples_clues_hi"] == 1

    assert counts["board_exact_clues_lo"] == 1
    assert counts["board_exact_clues_mid"] == 0
    assert counts["board_exact_clues_hi"] == 0

    assert counts["unknown_cell_total_clues_lo"] == 81 - 22  # 59
    assert counts["unknown_cell_total_clues_mid"] == 81 - 24  # 57
    assert counts["unknown_cell_total_clues_hi"] == 81 - 26  # 55

    assert counts["unknown_cell_correct_clues_lo"] == 59
    assert counts["unknown_cell_correct_clues_mid"] == 57 - 2  # 55
    assert counts["unknown_cell_correct_clues_hi"] == 55 - 5  # 50

    # Existing unstratified keys are untouched and consistent.
    assert counts["num_examples"] == 3
    assert counts["board_exact"] == 1
    assert counts["unknown_cell_total"] == 59 + 57 + 55
    assert counts["unknown_cell_correct"] == 59 + 55 + 50


def test_counts_bucket_boundaries_and_open_ends():
    # 23 -> lo (boundary), 25 -> hi (boundary), 19 -> lo (below the split's
    # min of 20), 30 -> hi (above the split's max of 29). No mid board.
    pred, sol, clue_board, clue_mask = _make_batch(
        clue_counts=[23, 25, 19, 30],
        wrong_unknown_cells=[[], [], [], []],
    )
    counts = _evaluate_board_batch_counts(
        pred_board=pred,
        solution_board=sol,
        clue_board=clue_board,
        clue_mask=clue_mask,
    )
    assert counts["num_examples_clues_lo"] == 2
    assert counts["num_examples_clues_mid"] == 0
    assert counts["num_examples_clues_hi"] == 2
    assert counts["unknown_cell_total_clues_lo"] == (81 - 23) + (81 - 19)
    assert counts["unknown_cell_total_clues_mid"] == 0
    assert counts["unknown_cell_total_clues_hi"] == (81 - 25) + (81 - 30)
    assert counts["board_exact_clues_lo"] == 2
    assert counts["board_exact_clues_mid"] == 0
    assert counts["board_exact_clues_hi"] == 2


def test_counts_buckets_partition_unstratified_totals():
    rng = np.random.default_rng(0)
    clue_counts = [20, 22, 23, 24, 24, 25, 27, 29]
    wrong = [
        sorted(rng.choice(np.arange(n, 81), size=k, replace=False).tolist())
        for n, k in zip(clue_counts, [0, 1, 3, 0, 2, 4, 1, 7])
    ]
    pred, sol, clue_board, clue_mask = _make_batch(clue_counts, wrong)
    counts = _evaluate_board_batch_counts(
        pred_board=pred,
        solution_board=sol,
        clue_board=clue_board,
        clue_mask=clue_mask,
    )
    for stem, total_key in (
        ("num_examples", "num_examples"),
        ("board_exact", "board_exact"),
        ("unknown_cell_correct", "unknown_cell_correct"),
        ("unknown_cell_total", "unknown_cell_total"),
    ):
        bucket_sum = sum(counts[f"{stem}_clues_{b}"] for b in BUCKETS)
        assert bucket_sum == counts[total_key], stem


# ---------------------------------------------------------------------------
# _base_totals: KeyError-on-accumulate regression
# ---------------------------------------------------------------------------

def test_base_totals_registers_all_12_stratified_keys():
    totals = _base_totals()
    assert len(STRATIFIED_COUNT_KEYS) == 12
    for key in STRATIFIED_COUNT_KEYS:
        assert key in totals, key
        assert totals[key] == 0


def test_accumulation_into_base_totals_does_not_keyerror():
    """Mimics the eval loop: `totals[key] += int(value)` over batch counts."""
    pred, sol, clue_board, clue_mask = _make_batch(
        clue_counts=[22, 24, 26],
        wrong_unknown_cells=[[], [30], []],
    )
    counts = _evaluate_board_batch_counts(
        pred_board=pred,
        solution_board=sol,
        clue_board=clue_board,
        clue_mask=clue_mask,
    )
    totals = _base_totals()
    for key, value in counts.items():
        totals[key] += int(value)  # KeyErrors if any count key is unregistered
    for key in STRATIFIED_COUNT_KEYS:
        assert totals[key] == counts[key]


def test_chained_pipeline_two_batches_through_finalize():
    """Full-chain regression: counts from TWO batches accumulated into
    `_base_totals()` via the eval loop's `totals[key] += int(value)` pattern,
    then finalized. The hi bucket is empty in BOTH batches, so the 0.0
    zero-denominator convention must survive the whole chain."""
    # Batch 1: board 0 = 22 clues (lo, exact); board 1 = 24 clues (mid, 1 wrong).
    # Batch 2: board 0 = 23 clues (lo boundary, 2 wrong); board 1 = 24 (mid, exact).
    batches = [
        _make_batch(clue_counts=[22, 24], wrong_unknown_cells=[[], [30]]),
        _make_batch(clue_counts=[23, 24], wrong_unknown_cells=[[40, 50], []]),
    ]
    totals = _base_totals()
    for pred, sol, clue_board, clue_mask in batches:
        counts = _evaluate_board_batch_counts(
            pred_board=pred,
            solution_board=sol,
            clue_board=clue_board,
            clue_mask=clue_mask,
        )
        for key, value in counts.items():
            totals[key] += int(value)
        totals["num_batches"] += 1

    prefix = "eval/chained"
    metrics = _finalize_metrics(
        totals=totals,
        metric_prefix=prefix,
        n_steps=50,
        checkpoint_source="live",
    )

    # lo: (22-clue exact) + (23-clue, 2 wrong): 1/2 boards exact,
    # cells (59 + 58 - 2) / (59 + 58).
    assert metrics[f"{prefix}/num_examples_clues_lo"] == 2.0
    assert metrics[f"{prefix}/board_acc_exact_clues_lo"] == 1.0 / 2.0
    assert metrics[f"{prefix}/cell_acc_unknown_clues_lo"] == 115.0 / 117.0
    # mid: two 24-clue boards, one exact, cells (57 - 1 + 57) / (57 + 57).
    assert metrics[f"{prefix}/num_examples_clues_mid"] == 2.0
    assert metrics[f"{prefix}/board_acc_exact_clues_mid"] == 1.0 / 2.0
    assert metrics[f"{prefix}/cell_acc_unknown_clues_mid"] == 113.0 / 114.0
    # hi: empty in both batches — 0.0 convention through the full chain.
    assert metrics[f"{prefix}/num_examples_clues_hi"] == 0.0
    assert metrics[f"{prefix}/board_acc_exact_clues_hi"] == 0.0
    assert metrics[f"{prefix}/cell_acc_unknown_clues_hi"] == 0.0
    # Unstratified totals stay consistent with the bucket partition.
    assert metrics[f"{prefix}/board_acc_exact"] == 2.0 / 4.0
    assert metrics[f"{prefix}/cell_acc_unknown"] == (115.0 + 113.0) / (117.0 + 114.0)
    assert metrics[f"{prefix}/num_batches"] == 2.0


# ---------------------------------------------------------------------------
# _finalize_metrics: exact ratios + zero-denominator convention
# ---------------------------------------------------------------------------

def _totals_for_finalize() -> dict[str, float]:
    totals = _base_totals()
    totals.update(
        {
            "num_examples": 10,
            "num_batches": 1,
            "board_exact": 5,
            "unknown_cell_correct": 40,
            "unknown_cell_total": 90,
            # lo: 4 boards, 2 exact, 30/40 unknown cells correct.
            "num_examples_clues_lo": 4,
            "board_exact_clues_lo": 2,
            "unknown_cell_correct_clues_lo": 30,
            "unknown_cell_total_clues_lo": 40,
            # mid: 6 boards, 3 exact, 10/50 unknown cells correct.
            "num_examples_clues_mid": 6,
            "board_exact_clues_mid": 3,
            "unknown_cell_correct_clues_mid": 10,
            "unknown_cell_total_clues_mid": 50,
            # hi: empty bucket (zero denominators).
            "num_examples_clues_hi": 0,
            "board_exact_clues_hi": 0,
            "unknown_cell_correct_clues_hi": 0,
            "unknown_cell_total_clues_hi": 0,
        }
    )
    return totals


def test_finalize_metrics_emits_stratified_keys_with_exact_ratios():
    prefix = "eval/run_a"
    metrics = _finalize_metrics(
        totals=_totals_for_finalize(),
        metric_prefix=prefix,
        n_steps=50,
        checkpoint_source="live",
    )
    assert metrics[f"{prefix}/board_acc_exact_clues_lo"] == 2.0 / 4.0
    assert metrics[f"{prefix}/board_acc_exact_clues_mid"] == 3.0 / 6.0
    assert metrics[f"{prefix}/cell_acc_unknown_clues_lo"] == 30.0 / 40.0
    assert metrics[f"{prefix}/cell_acc_unknown_clues_mid"] == 10.0 / 50.0
    assert metrics[f"{prefix}/num_examples_clues_lo"] == 4.0
    assert metrics[f"{prefix}/num_examples_clues_mid"] == 6.0
    assert metrics[f"{prefix}/num_examples_clues_hi"] == 0.0
    # Zero-denominator bucket: _safe_ratio convention is 0.0, not NaN/raise.
    assert metrics[f"{prefix}/board_acc_exact_clues_hi"] == 0.0
    assert metrics[f"{prefix}/cell_acc_unknown_clues_hi"] == 0.0
    # Existing unstratified metrics are untouched.
    assert metrics[f"{prefix}/board_acc_exact"] == 0.5
    assert metrics[f"{prefix}/cell_acc_unknown"] == 40.0 / 90.0


# ---------------------------------------------------------------------------
# CSV round-trip: silent-drop regression + effective blur_score contract
# ---------------------------------------------------------------------------

NEW_CSV_COLUMNS = [
    "blur_score",
    "board_acc_exact_clues_lo",
    "board_acc_exact_clues_mid",
    "board_acc_exact_clues_hi",
    "cell_acc_unknown_clues_lo",
    "cell_acc_unknown_clues_mid",
    "cell_acc_unknown_clues_hi",
    "num_examples_clues_lo",
    "num_examples_clues_mid",
    "num_examples_clues_hi",
]


def test_new_columns_registered_in_whitelist():
    for column in NEW_CSV_COLUMNS:
        assert column in SUDOKU_SJD_PROGRESS_COLUMNS, column


def test_sjd_run_row_csv_round_trip_survives_normalization():
    prefix = "eval/run_a"
    metrics = _finalize_metrics(
        totals=_totals_for_finalize(),
        metric_prefix=prefix,
        n_steps=50,
        checkpoint_source="live",
    )
    # blur_score requested but no kernel attached -> effective state is False
    # (this is what build_sudoku_eval_logger records on the spec).
    spec = {
        "kind": "policy",
        "label": "run_a",
        "metrics_prefix": prefix,
        "policy": "linear_topk_probability",
        "eta": 1.0,
        "blur_score": True,
        "blur_score_effective": False,
    }
    row = _sjd_run_row(spec, metrics)
    normalized = _normalize_csv_row(row)

    # Silent-drop regression: every new column survives normalization with the
    # exact finalized value (none coerced to the empty-string None marker).
    assert normalized["board_acc_exact_clues_lo"] == 0.5
    assert normalized["board_acc_exact_clues_mid"] == 0.5
    assert normalized["board_acc_exact_clues_hi"] == 0.0
    assert normalized["cell_acc_unknown_clues_lo"] == 0.75
    assert normalized["cell_acc_unknown_clues_mid"] == 0.2
    assert normalized["cell_acc_unknown_clues_hi"] == 0.0
    assert normalized["num_examples_clues_lo"] == 4.0
    assert normalized["num_examples_clues_mid"] == 6.0
    assert normalized["num_examples_clues_hi"] == 0.0

    # Effective-state contract: requested True + no kernel -> column is False,
    # NOT the requested flag.
    assert normalized["blur_score"] is False


def test_sjd_run_row_blur_score_effective_true_when_kernel_attached():
    prefix = "eval/run_b"
    metrics = _finalize_metrics(
        totals=_totals_for_finalize(),
        metric_prefix=prefix,
        n_steps=50,
        checkpoint_source="live",
    )
    spec = {
        "kind": "policy",
        "label": "run_b",
        "metrics_prefix": prefix,
        "policy": "linear_topk_probability",
        "eta": 1.0,
        "blur_score": True,
        "blur_score_effective": True,
    }
    row = _sjd_run_row(spec, metrics)
    assert row["blur_score"] is True
    assert _normalize_csv_row(row)["blur_score"] is True


def test_sjd_run_row_sampler_spec_blur_score_blank():
    """Sampler-kind specs never get blur_score_effective: the row carries None
    and the CSV cell normalizes to the blank-string marker."""
    prefix = "eval/predictor_only"
    metrics = _finalize_metrics(
        totals=_totals_for_finalize(),
        metric_prefix=prefix,
        n_steps=50,
        checkpoint_source="live",
    )
    spec = {
        "kind": "sampler",
        "label": "predictor_only",
        "metrics_prefix": prefix,
    }
    row = _sjd_run_row(spec, metrics)
    assert row["blur_score"] is None
    assert _normalize_csv_row(row)["blur_score"] == ""
    # Stratified metrics are emitted for every spec kind (shared finalizer).
    assert _normalize_csv_row(row)["board_acc_exact_clues_lo"] == 0.5
