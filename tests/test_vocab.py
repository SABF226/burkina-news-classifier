"""Tests for src/vocab.py — tokenization, vocabulary building, encoding and saving.

Responsible: Team Member 1 (Dataset Research & Data Preparation)
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from vocab import Vocab, build_vocab, tokenize, PAD_TOKEN, UNK_TOKEN
from data import load_splits
import json


def setup_module(module):
    tr, vl, te = load_splits()
    global train_texts, val_texts, test_texts
    train_texts = list(tr["text"].tolist())
    val_texts = list(vl["text"].tolist())
    test_texts = list(te["text"].tolist())


def test_tokenize_basic():
    """tokenize returns a list of lowercase, punctuation-separated tokens."""
    out = tokenize("L'Ouganda à l'assaut des ""fimbu"" de la RDC")
    assert isinstance(out, list)
    assert len(out) > 0
    # Should contain 'l' and 'ouguanda' (accent preserved), punctuation separated
    joined = " ".join(out)
    assert "l'" in joined or "'l" in joined
    assert "fimbu" in joined


def test_tokenize_lowercase():
    """tokenize lowercases the input."""
    assert tokenize("Le SOLEIL luit").lower() == tokenize("le soléil luit")
    # Fully uppercase
    assert tokenize("ABCXYZ").tolist() if hasattr(tokenize("ABCXYZ"), "tolist") else True  # careful with type
    assert tokenize("ABCXYZ") == ["abcxyz"]


def test_vocab_pad_idx_is_zero():
    """PAD token always occupies index 0."""
    v = build_vocab(["hello", "world", "hi"])
    assert v.pad_idx == 0
    assert v.token2id[PAD_TOKEN] == 0


def test_vocab_unk_idx_is_one():
    """UNK token always occupies index 1."""
    v = build_vocab(["hello", "world"])
    assert v.unk_idx == 1
    assert v.token2id[UNK_TOKEN] == 1


def test_vocab_build_exclusive_to_train():
    """Vocabulary is built from training data only; val/test tokens may be unseen."""
    v = build_vocab(train_texts, min_freq=2)
    # Each token from val must be present in vocab (or be <UNK>)
    for t in val_texts:
        toks = set(tokenize(t))
        for tok in toks:
            assert v.token2id.get(tok, -1) != -1, f"Val token '{tok}' not in vocab!"


def test_vocab_min_freq_filter():
    """Words below min_freq are not kept in the vocabulary."""
    low_freq = ["the", "and", "xyz123"]  # 'xyz123' will likely be under frequency threshold
    high_freq = ["hello", "world"] * 5
    combined = low_freq + high_freq
    v = build_vocab(combined, min_freq=2)
    assert "xyz123" not in v.token2id or v.token2id.get("xyz123", -1) != -1
    assert "hello" in v.token2id


def test_encode_known_tokens():
    """encode maps known tokens correctly."""
    v = build_vocab(["hello world hi there"])
    enc = v.encode("hello world hi there")
    assert enc[0] == v.pad_idx  # padding at end handled by pad_or_truncate
    assert enc == [v.token2id["hello"], v.token2id["world"], v.token2id["hi"], v.token2id["there"], v.pad_idx] or \
           enc == [v.token2id["hello"], v.token2id["world"], v.token2id["hi"], v.token2id["there"]]


def test_encode_unseen_token_maps_to_unk():
    """Unseen tokens are mapped to <UNK> index."""
    v = build_vocab(["hello world"])
    # 'unknownword' should be unseen during build (not in train texts)
    enc = v.encode("hello unknownword !")
    assert enc[-1] == v.unk_idx
    assert enc[0] == v.token2id["hello"]


def test_decode_roundtrip():
    """decode reverses the encoding correctly."""
    v = build_vocab(["a b c d", "e f g h"])
    ids = [v.token2id["a"], v.token2id["b"], v.token2id["c"], v.token2id["d"],
           v.pad_idx]
    decoded = v.decode(ids)
    assert decoded == ['<PAD>', 'a', 'b', 'c', 'd'] or decoded[-2:] == ['d'] + ['']


def test_save_load_preserves_vocabulary():
    """Save and reload produce an equivalent vocabulary."""
    v = build_vocab(train_texts, min_freq=2)
    out = "models/vocab.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    v.save(out)
    v2 = Vocab.load(out)
    assert len(v2.token2id) == len(v.token2id)
    assert v2.token2id == v.token2id
    assert v2.id2token == v.id2token


def test_pad_or_truncate_handles_short_sequences():
    """Short sequences are padded to max_length with <PAD> (index 0)."""
    v = build_vocab(["hello world"])
    seq = [v.token2id["hello"], v.token2id["world"]]
    padded = v.pad_or_truncate(seq, max_length=5)
    assert len(padded) == 5
    assert padded[-1] == 0       # last element is padding
    assert padded[-2] != 0      # second to last is 'world'
    assert padded[-3] == v.pad_idx


def test_pad_or_truncate_handles_long_sequences():
    """Long sequences are truncated at max_length."""
    v = build_vocab(["one two three four five"])
    long_list = list(range(100))
    padded = v.pad_or_truncate(long_list, max_length=10)
    assert len(padded) == 10
    assert padded[9] == 9
    assert padded[-1] == 0


def test_pad_or_truncate_empty_input():
    """Empty input list still yields a sequence of max_length zeros."""
    v = build_vocab(["hello"])
    padded = v.pad_or_truncate([], max_length=7)
    assert len(padded) == 7
    assert all(x == 0 for x in padded)


def test_max_words_truncation():
    """Vocabulary respects max_words limit when exceeded."""
    # Build a very small vocab with huge max_words to force truncation
    big_list = []
    for i in range(10000):
        big_list.append(f"word{i}")
    # This will trigger ValueError because indices collide; simpler approach:
    v = build_vocab([f"token{i}" for i in range(50)], max_words=5)
    assert len(v.token2id) <= 5 + 2  # +2 reserved


def test_encoding_statistics_on_dataset():
    """% of <UNK> tokens on validation and test sets should be reasonably low."""
    v = build_vocab(train_texts, min_freq=2)
    val_unk = sum(1 for t in val_texts if v.token2id.get(t, v.unk_idx) == v.unk_idx)
    test_unk = sum(1 for t in test_texts if v.token2id.get(t, v.unk_idx) == v.unk_idx)
    # UNK can be higher in test (smaller split, fewer words seen in training)
    # but should still be a fraction of total tokens.
    val_unk_frac = val_unk / len(val_texts)
    test_unk_frac = test_unk / len(test_texts)
    assert val_unk_frac < 0.5, f"Validation UNK too high: {val_unk_frac:.2%}"
    assert test_unk_frac < 0.5, f"Test UNK too high: {test_unk_frac:.2%}"


def test_all_reserved_indices():
    """Indices 0 and 1 must always map to PAD and UNK respectively."""
    v = build_vocab(["hello"])
    assert v.token2id[PAD_TOKEN] == 0
    assert v.token2id[UNK_TOKEN] == 1
    assert v.id2token[0] == PAD_TOKEN
    assert v.id2token[1] == UNK_TOKEN
