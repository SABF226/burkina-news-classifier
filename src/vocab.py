"""Tokenization, vocabulary construction and text encoding for Burkina News Classifier.

Responsible: Team Member 1 (Dataset Research & Data Preparation)
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Special tokens must be stable indices to avoid index drift between training and inference.
PAD_TOKEN = "<PAD>"   # reserved index 0
UNK_TOKEN = "<UNK>"   # reserved index 1


def tokenize(text: str) -> List[str]:
    """Tokenize a French article into a list of tokens.

    Words keep internal apostrophes and hyphens (for example, ``l'assaut``).
    Other punctuation marks are returned as individual tokens. Accents are
    preserved and curly apostrophes are normalized to straight apostrophes.

    Example:
        "L'Ouganda à l'assaut des ""fimbu"" de la RDC"
        -> ["l'ouguanda", "à", "l'", "assaut", "des", ""fimbu"", "de", "la", "rdc"]
    """
    text = text.lower().replace("’", "'")
    # ``\w`` is Unicode-aware in Python, so French accented letters stay
    # inside words. The alternative matches each punctuation mark separately.
    token_pattern = r"\w+(?:['-]\w+)*|[^\w\s]"
    return re.findall(token_pattern, text, flags=re.UNICODE)


class Vocab:
    """A mapping from token strings to integer indices plus special-token handling."""

    def __init__(self, pad_token: str = PAD_TOKEN, unk_token: str = UNK_TOKEN,
                 min_freq: int = 2, max_words: int = None):
        # Reserve indices 0 and 1 for special tokens
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.max_words = max_words
        if min_freq < 1:
            raise ValueError("min_freq must be at least 1")
        self.min_freq = min_freq

        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.freq: Dict[str, int] = {}

        # Immutable special-token maps to guard against accidental reordering
        self.token2id[pad_token] = 0
        self.id2token[0] = pad_token
        self.token2id[unk_token] = 1
        self.id2token[1] = unk_token

    @property
    def pad_idx(self) -> int:
        return 0

    @property
    def unk_idx(self) -> int:
        return 1

    def build(self, texts: List[str]) -> None:
        """Build the vocabulary exclusively from training texts.

        ``min_freq`` controls which words are kept. Tokens appearing above
        ``max_words`` are truncated last so rare but useful words can still be
        kept when the limit is hit. Frequencies count every occurrence across
        all training documents — not unique mentions per document.
        """
        # First pass: tally frequencies of every token across the whole corpus
        freq = {}
        for text in texts:
            for tok in tokenize(text):
                if tok == self.pad_token or tok == self.unk_token:
                    continue
                freq[tok] = freq.get(tok, 0) + 1

        # Second pass: reserve indices for high-frequency tokens, keeping
        # alphabetical order as a deterministic tie-breaker.
        freq = {
            token: count
            for token, count in freq.items()
            if count >= self.min_freq
        }

        for rank, (tok, count) in enumerate(sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))):
            idx = len(self.token2id) #cette ligne permet de récupérer l'index du token en cours d'ajout à la vocabulaire
            if idx < 2:
                raise RuntimeError("Vocabulary exhausted without reserved indices")
            self.token2id[tok] = idx
            self.id2token[idx] = tok
            self.freq[tok] = count
            if self.max_words is not None and rank >= self.max_words - 1:
                break

    def encode(self, text: str) -> List[int]:
        """Convert a single text to a list of token indices.

        Unseen tokens are mapped to <UNK> (index 1). Extra tokens are ignored
        (per token) instead of producing infinite sequences.
        """
        ids: List[int] = []
        for tok in tokenize(text):
            ids.append(self.token2id.get(tok, self.unk_idx))
        return ids

    def decode(self, ids: List[int]) -> List[str]:
        """Map integer IDs back to token strings (for debugging / display)."""
        return [self.id2token.get(i, "<PAD>") for i in ids]

    def save(self, path: str) -> None:
        """Save the vocabulary (including id2token reverse map) as JSON."""
        data = {
            "token2id": self.token2id,
            "id2token": {str(k): v for k, v in self.id2token.items()},
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "min_freq": self.min_freq if hasattr(self, 'min_freq') else None,
            "max_words": self.max_words if hasattr(self, 'max_words') else None,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        logger.info(f"Vocabulary saved to {path} ({len(self.token2id)} entries)")

    @classmethod
    def load(cls, path: str) -> "Vocab":
        """Load a vocabulary previously saved with :meth:`save`.

        Returns a fresh Vocab instance populated from the JSON file. The class
        parameters (pad_token, unk_token, min_freq, max_words) are read from
        the file so training and inference use exactly the same vocab.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        vocab = cls(
            pad_token=data["pad_token"],
            unk_token=data["unk_token"],
            min_freq=data.get("min_freq", 2),
            max_words=data.get("max_words", None),
        )
        vocab.token2id = data["token2id"]
        # Reverse map keyed by string index
        vocab.id2token = {int(k): v for k, v in data["id2token"].items()}
        logger.info(f"Vocabulary loaded from {path} ({len(vocab.token2id)} entries)")
        return vocab

    def pad_or_truncate(self, ids: List[int], max_length: int) -> List[int]:
        """Return a fixed-length sequence of integer IDs.

        - Truncate at ``max_length`` from the left.
        - Pad with <PAD> (index 0) at the end.
        - Even an empty input list yields ``[PAD, PAD, ...]`` of length ``max_length``.
        """
        ids = ids[:max_length]
        return ids + [self.pad_idx] * (max_length - len(ids))


def build_vocab(train_texts: List[str], min_freq: int = 2, max_words: int = None,
                pad_token: str = PAD_TOKEN, unk_token: str = UNK_TOKEN) -> Vocab:
    """Convenience entry point: create, populate and return a Vocab object."""
    v = Vocab(pad_token=pad_token, unk_token=unk_token, min_freq=min_freq, max_words=max_words)
    v.build(train_texts)
    return v


if __name__ == "__main__":
    """Allow running as script: python -m src.vocab"""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # ── Build vocab from train texts ──
    sys_path_insert = "src"  # whatever that was before...
    from data import load_splits
    tr, vl, te = load_splits()
    print("Building vocabulary from train texts...")
    vocab = build_vocab(list(tr["text"].tolist()), min_freq=2)
    print(f"Total vocab size: {len(vocab.token2id)}")
    print(f"<PAD> index={vocab.pad_idx}, <UNK> index={vocab.unk_idx}")

    # ── Statistics ──
    print("\n=== Vocabulary Size & Special Token Stats ===")
    print(f"Total unique tokens: {len(vocab.token2id)}")
    print(f"<PAD> occupies index {vocab.pad_idx}")
    print(f"<UNK> occupies index {vocab.unk_idx}")

    # Count unseen tokens in val and test
    val_unk_count = sum(1 for t in vl["text"] if vocab.token2id.get(tokenize(t), 1) == 1)
    test_unk_count = sum(1 for t in te["text"] if vocab.token2id.get(tokenize(t), 1) == 1)
    print(f"\nUnseen tokens in validation ({len(vl)} docs): {val_unk_count}/{len(vl)} = {val_unk_count/len(vl)*100:.1f}%")
    print(f"Unseen tokens in test    ({len(te)} docs): {test_unk_count}/{len(te)} = {test_unk_count/len(te)*100:.1f}%")

    # Sample encoded texts
    sample_text = tr["text"].iloc[0][:200]
    encoded = vocab.encode(sample_text)
    padded = vocab.pad_or_truncate(encoded, max_length=300)
    print(f"\nSample '{sample_text[:60]}...' -> {len(encoded)} tokens, padded length={len(padded)}")
    print(f"First 10 tokens: {padded[:10]}")
    print(f"Last 10 tokens: {padded[-10:]}")

    # ── Save ──
    out_dir = "models"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    vocab_path = f"{out_dir}/vocab.json"
    vocab.save(vocab_path)
    print(f"\nSaved vocabulary to {vocab_path}")

    # ── Round-trip test ──
    loaded = Vocab.load(vocab_path)
    assert loaded == vocab or set(loaded.token2id.keys()) == set(vocab.token2id.keys()), "Round-trip mismatch!"
    roundtrip_test = loaded.encode(sample_text)
    assert roundtrip_test == encoded, "Round-trip encode mismatch!"
    print(f"Round-trip encode OK: {len(roundtrip_test)} == {len(encoded)}")
