from __future__ import annotations

from typing import Mapping

import numpy as np


SUDOKU_NUM_CELLS = 81
SUDOKU_TOKENS_PER_TRIPLE = 3
SUDOKU_TRIPLET_SEQ_LEN = SUDOKU_NUM_CELLS * SUDOKU_TOKENS_PER_TRIPLE

# Original Sudoku row/col/value tokens stay in the 0..9 range.
SUDOKU_BASE_VOCAB_SIZE = 10
SUDOKU_SEP_TOKEN_ID = 10
SUDOKU_EOS_TOKEN_ID = 11
SUDOKU_VOCAB_SIZE = 12
SUDOKU_PACKED_SEQ_LEN = SUDOKU_TRIPLET_SEQ_LEN + 2


def _normalize_triplet_batch(
    *,
    triplet_seq: np.ndarray,
    start_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    seq = np.asarray(triplet_seq, dtype=np.int32)
    if seq.ndim == 1:
        seq = seq.reshape(1, -1)
    if seq.ndim != 2 or seq.shape[1] != SUDOKU_TRIPLET_SEQ_LEN:
        raise ValueError(
            "Expected Sudoku triplet token sequence with shape "
            f"(batch, {SUDOKU_TRIPLET_SEQ_LEN}) or ({SUDOKU_TRIPLET_SEQ_LEN},), "
            f"got {seq.shape}."
        )

    if seq.size > 0 and ((seq.min() < 0) or (seq.max() > 9)):
        raise ValueError(
            "Sudoku triplet sequences must preserve the original token ids in 0..9."
        )

    start = np.asarray(start_index, dtype=np.int32).reshape(-1)
    if start.shape[0] != seq.shape[0]:
        raise ValueError(
            "start_index must have one entry per Sudoku example: "
            f"got {start.shape[0]} for batch size {seq.shape[0]}."
        )
    if np.any(start < 0) or np.any(start > SUDOKU_NUM_CELLS):
        raise ValueError(
            f"Invalid Sudoku start_index; expected values in 0..{SUDOKU_NUM_CELLS}."
        )
    return seq, start


def packed_sudoku_positions(*, start_index: np.ndarray) -> Mapping[str, np.ndarray]:
    start = np.asarray(start_index, dtype=np.int32).reshape(-1)
    if np.any(start < 0) or np.any(start > SUDOKU_NUM_CELLS):
        raise ValueError(
            f"Invalid Sudoku start_index; expected values in 0..{SUDOKU_NUM_CELLS}."
        )

    prompt_token_count = SUDOKU_TOKENS_PER_TRIPLE * start
    sep_index = prompt_token_count
    response_start_index = sep_index + 1
    eos_index = np.full_like(sep_index, SUDOKU_PACKED_SEQ_LEN - 1)

    return {
        "prompt_token_count": prompt_token_count.reshape(-1, 1),
        "response_token_count": (
            SUDOKU_TRIPLET_SEQ_LEN - prompt_token_count
        ).reshape(-1, 1),
        "sep_index": sep_index.reshape(-1, 1),
        "response_start_index": response_start_index.reshape(-1, 1),
        "eos_index": eos_index.reshape(-1, 1),
    }


def pack_sudoku_seq2seq(
    *,
    triplet_seq: np.ndarray,
    start_index: np.ndarray,
) -> Mapping[str, np.ndarray]:
    seq, start = _normalize_triplet_batch(
        triplet_seq=triplet_seq,
        start_index=start_index,
    )
    positions = packed_sudoku_positions(start_index=start)

    batch_size = int(seq.shape[0])
    packed_seq = np.empty((batch_size, SUDOKU_PACKED_SEQ_LEN), dtype=np.int32)
    prompt_mask = np.zeros((batch_size, SUDOKU_PACKED_SEQ_LEN), dtype=np.bool_)
    response_mask = np.zeros((batch_size, SUDOKU_PACKED_SEQ_LEN), dtype=np.bool_)

    sep_index = positions["sep_index"].reshape(-1)
    eos_index = positions["eos_index"].reshape(-1)
    response_start_index = positions["response_start_index"].reshape(-1)

    for i in range(batch_size):
        prompt_token_count = int(sep_index[i])
        response_start = int(response_start_index[i])
        eos = int(eos_index[i])

        packed_seq[i, :prompt_token_count] = seq[i, :prompt_token_count]
        packed_seq[i, prompt_token_count] = SUDOKU_SEP_TOKEN_ID
        packed_seq[i, response_start:eos] = seq[i, prompt_token_count:]
        packed_seq[i, eos] = SUDOKU_EOS_TOKEN_ID

        # Prompt-side positions are clamped during decoding, including the [SEP].
        prompt_mask[i, :response_start] = True
        # Response-side positions are predicted/scored, including the [EOS].
        response_mask[i, response_start:] = True

    return {
        "packed_seq": packed_seq,
        "prompt_mask": prompt_mask,
        "response_mask": response_mask,
        **positions,
    }
