from __future__ import annotations

import numpy as np

from sticky.data.sudoku import make_sudoku_iterator


def _solved_board_triples() -> np.ndarray:
    triples = np.zeros((81, 3), dtype=np.int32)
    idx = 0
    for row in range(9):
        for col in range(9):
            triples[idx] = (row, col, ((row * 3 + row // 3 + col) % 9) + 1)
            idx += 1
    return triples


def _pack_row(*, start_index: int, triples: np.ndarray, strategy_offset: int) -> np.ndarray:
    strategy = (np.arange(81, dtype=np.int32) + int(strategy_offset)).reshape(81, 1)
    quads = np.concatenate([triples.astype(np.int32), strategy], axis=1)
    return np.concatenate(
        [
            np.asarray([start_index], dtype=np.int32),
            quads.reshape(-1),
        ],
        axis=0,
    )


def _reference_transform(
    *,
    triples: np.ndarray,
    start_index: int,
    seq_order: str,
    rng: np.random.Generator | None,
) -> np.ndarray:
    if seq_order == "dataset":
        return triples.reshape(-1)

    transformed = np.empty_like(triples)
    prefix = triples[:start_index].copy()
    suffix = triples[start_index:].copy()

    if seq_order == "fixed":
        prefix = prefix[np.lexsort(prefix[:, ::-1].T)] if len(prefix) > 1 else prefix
        suffix = suffix[np.lexsort(suffix[:, ::-1].T)] if len(suffix) > 1 else suffix
    elif seq_order == "random":
        assert rng is not None
        prefix = prefix[rng.permutation(len(prefix))] if len(prefix) > 1 else prefix
        suffix = suffix[rng.permutation(len(suffix))] if len(suffix) > 1 else suffix
    else:
        raise AssertionError(f"Unexpected seq_order={seq_order!r}")

    transformed[:start_index] = prefix
    transformed[start_index:] = suffix
    return transformed.reshape(-1)


def test_make_sudoku_iterator_matches_original_prefix_suffix_ordering(tmp_path):
    base = _solved_board_triples()
    triples_a = base[::-1]
    triples_b = np.roll(base, 11, axis=0)
    table = np.stack(
        [
            _pack_row(start_index=2, triples=triples_a, strategy_offset=0),
            _pack_row(start_index=5, triples=triples_b, strategy_offset=100),
        ],
        axis=0,
    )

    train_path = tmp_path / "synthetic-sudoku.npy"
    np.save(train_path, table)

    expected_board = np.zeros((81,), dtype=np.int32)
    for row, col, value in base:
        expected_board[(row * 9) + col] = value

    for seq_order in ("dataset", "fixed", "random"):
        batch = next(
            make_sudoku_iterator(
                split="train",
                batch_size=2,
                seed=17,
                data_dir=str(tmp_path),
                train_file=str(train_path.name),
                shuffle=False,
                repeat=False,
                drop_remainder=False,
                seq_order=seq_order,
                mmap=False,
                auto_download=False,
            )
        )

        rng = np.random.default_rng(17) if seq_order == "random" else None
        expected = np.stack(
            [
                _reference_transform(
                    triples=triples_a,
                    start_index=2,
                    seq_order=seq_order,
                    rng=rng,
                ),
                _reference_transform(
                    triples=triples_b,
                    start_index=5,
                    seq_order=seq_order,
                    rng=rng,
                ),
            ],
            axis=0,
        )

        np.testing.assert_array_equal(batch["image"], expected)
        np.testing.assert_array_equal(batch["start_index"], np.asarray([[2], [5]], dtype=np.int32))
        np.testing.assert_array_equal(
            batch["puzzle"],
            np.stack([expected_board, expected_board], axis=0),
        )
