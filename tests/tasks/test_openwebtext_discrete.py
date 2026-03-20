from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sticky.data.openwebtext import load_token_sequences, make_openwebtext_iterator
from sticky.tasks.openwebtext_discrete import OpenWebTextDiscreteTask


@pytest.fixture
def synthetic_text_tokens(tmp_path: Path):
    train = (np.arange(6 * 8, dtype=np.int64).reshape(6, 8) % 97).astype(np.int64)
    eval_ = (np.arange(3 * 8, dtype=np.int32).reshape(3, 8) + 11).astype(np.int32)

    train_path = tmp_path / "train.npy"
    eval_path = tmp_path / "eval.npz"
    np.save(train_path, train)
    np.savez(eval_path, tokens=eval_)

    return {
        "train": train,
        "eval": eval_,
        "train_path": train_path,
        "eval_path": eval_path,
    }


def test_openwebtext_loader_supports_npy_and_npz(synthetic_text_tokens):
    train_tokens = load_token_sequences(
        str(synthetic_text_tokens["train_path"]),
        seq_len=8,
        mmap=True,
    )
    eval_tokens = load_token_sequences(
        str(synthetic_text_tokens["eval_path"]),
        seq_len=8,
        mmap=True,
    )

    assert train_tokens.shape == (6, 8)
    assert eval_tokens.shape == (3, 8)
    assert np.issubdtype(train_tokens.dtype, np.integer)
    assert np.issubdtype(eval_tokens.dtype, np.integer)


def test_openwebtext_iterator_and_task_smoke(synthetic_text_tokens):
    iterator = make_openwebtext_iterator(
        split="train",
        batch_size=2,
        seq_len=8,
        train_tokens_path=str(synthetic_text_tokens["train_path"]),
        eval_tokens_path=str(synthetic_text_tokens["eval_path"]),
        seed=0,
        shuffle=False,
        repeat=False,
        drop_remainder=True,
        mmap=True,
    )
    batch = next(iterator)
    assert batch["image"].shape == (2, 8)
    assert batch["image"].dtype == np.int32
    assert np.array_equal(batch["image"], synthetic_text_tokens["train"][:2].astype(np.int32))

    task = OpenWebTextDiscreteTask(
        task_name="openwebtext_discrete",
        train_tokens_path=str(synthetic_text_tokens["train_path"]),
        eval_tokens_path=str(synthetic_text_tokens["eval_path"]),
        batch_size=2,
        eval_batch_size=2,
        seq_len=8,
        vocab_size=128,
        tokenizer_name=None,
        shuffle=False,
        drop_remainder=True,
    )

    train_iter, eval_iter = task.make_dataloaders(seed=0)
    train_batch = next(train_iter)
    eval_batch = next(eval_iter)

    assert task.spec.task_type == "text"
    assert task.spec.data_shape == (8,)
    assert train_batch["image"].shape == (2, 8)
    assert eval_batch["image"].shape == (2, 8)
    assert task.format_samples_for_logging(train_batch["image"][:1]) == [
        "0 1 2 3 4 5 6 7"
    ]
