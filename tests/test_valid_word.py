"""Valid-word metric matches CANDI calculate_word_dictionary_match."""
import json

import numpy as np

from sticky.eval.valid_word import (
    load_test_vocab,
    score_samples,
    valid_word_frontier_report,
    word_dictionary_match,
)

# Mini reference vocab; all entries are len>4, mirroring the test-split filter.
MINI_VOCAB = {"quick", "brown", "jumps"}


def test_per_sample_score_is_unique_matches_over_total_tokens():
    # 5 tokens; "the"/"fox" not in vocab; 3 unique matches -> 3/5 = 0.6.
    score = word_dictionary_match("the quick brown fox jumps", MINI_VOCAB)
    assert abs(score - 0.6) < 1e-9


def test_denominator_is_total_tokens_not_unique():
    # Repeated non-matching tokens inflate the denominator (total tokens),
    # so a single match over 4 tokens scores 1/4, not 1/(unique=2).
    score = word_dictionary_match("quick the the the", MINI_VOCAB)
    assert abs(score - 0.25) < 1e-9


def test_empty_sample_scores_zero():
    assert word_dictionary_match("", MINI_VOCAB) == 0.0


def test_frontier_report_mean_and_max_along_temperature():
    # nfe=32: temp 0.5 all-match (1.0), temp 1.0 no-match (0.0). Vocab via ids.
    # Build samples with an explicit id->char map for two single-word rows.
    id_to_char = {0: "q", 1: "u", 2: "i", 3: "c", 4: "k", 5: " "}
    vocab = {"quick"}
    quick = np.array([[0, 1, 2, 3, 4]])  # decodes to "quick" -> 1 token, match
    nope = np.array([[5, 5, 5, 5, 5]])   # decodes to "" -> 0 tokens -> 0.0
    samples = {
        (32, 0.5): quick,
        (32, 1.0): nope,
        (64, 1.0): quick,
    }
    report = valid_word_frontier_report(samples, id_to_char=id_to_char, vocab=vocab)
    assert abs(report["eval/valid_word@nfe32_temp0.5"] - 1.0) < 1e-9
    assert abs(report["eval/valid_word@nfe32_temp1"] - 0.0) < 1e-9
    # Max along temperature for nfe=32 picks the 1.0 gridpoint.
    assert abs(report["eval/valid_word_max@nfe32"] - 1.0) < 1e-9
    assert abs(report["eval/valid_word_max@nfe64"] - 1.0) < 1e-9


def test_score_samples_uses_char_map():
    id_to_char = {0: "q", 1: "u", 2: "i", 3: "c", 4: "k"}
    samples = np.array([[0, 1, 2, 3, 4]])
    assert abs(score_samples(samples, id_to_char=id_to_char, vocab={"quick"}) - 1.0) < 1e-9


def test_load_test_vocab_round_trip(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(sorted(MINI_VOCAB)), encoding="utf-8")
    assert load_test_vocab(str(path)) == MINI_VOCAB
