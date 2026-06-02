"""Text8 data loader + char-map round-trip."""
import numpy as np

from sticky.data.text8 import (
    TEXT8_CHARS,
    TEXT8_VOCAB_SIZE,
    decode_ids,
    encode_text8,
    make_text8_iterator,
)


def test_char_map_ordering_is_candi_verbatim():
    assert TEXT8_CHARS == "abcdefghijklmnopqrstuvwxyz "
    assert TEXT8_VOCAB_SIZE == 27
    assert encode_text8("a")[0] == 0
    assert encode_text8("z")[0] == 25
    assert encode_text8(" ")[0] == 26


def test_encode_decode_round_trip():
    text = "the quick brown fox jumps"
    ids = encode_text8(text)
    assert ids.dtype == np.int32
    assert decode_ids(ids) == text


def test_loader_yields_int_batches(tmp_path):
    seq_len = 8
    tokens = np.arange(TEXT8_VOCAB_SIZE * seq_len, dtype=np.int32).reshape(
        TEXT8_VOCAB_SIZE, seq_len
    ) % TEXT8_VOCAB_SIZE
    path = tmp_path / "train.npy"
    np.save(path, tokens)

    it = make_text8_iterator(
        split="train",
        batch_size=4,
        seq_len=seq_len,
        train_tokens_path=str(path),
        eval_tokens_path=None,
        seed=0,
        shuffle=False,
        repeat=False,
        drop_remainder=True,
    )
    batch = next(iter(it))
    arr = np.asarray(batch["image"])
    assert arr.shape == (4, seq_len)
    assert np.issubdtype(arr.dtype, np.integer)
    assert arr.min() >= 0 and arr.max() < TEXT8_VOCAB_SIZE
