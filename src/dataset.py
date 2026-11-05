"""PyTorch Dataset and batch-building utilities for the MasakhaNEWS articles."""

from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, Dataset

try:  # pragma: no cover - supports both package and direct-script imports
    from .data import load_splits
except ImportError:  # pragma: no cover
    from data import load_splits

try:  # pragma: no cover
    from .vocab import Vocab, build_vocab
except ImportError:  # pragma: no cover
    from vocab import Vocab, build_vocab


class NewsDataset(Dataset):
    """Convert articles and labels into tensors for PyTorch."""

    def __init__(self, texts, labels, vocab, max_length: int):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_length = max_length

        if len(self.texts) != len(self.labels):
            raise ValueError("texts and labels must have the same length")

        if max_length <= 0:
            raise ValueError("max_length must be greater than 0")

    def __len__(self) -> int:
        """Return the number of articles."""
        return len(self.texts)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one encoded article and its label as PyTorch tensors."""
        token_ids = self.vocab.encode(self.texts[index])
        token_ids = self.vocab.pad_or_truncate(token_ids, self.max_length)

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)

        return input_ids, label


def make_loaders(
    data_dir: str = "datasets",
    batch_size: int = 32,
    max_length: int = 1800,
    min_freq: int = 2,
    shuffle_train: bool = True,
    return_vocab: bool = False,
):
    """Create train/validation/test datasets and DataLoaders from the cleaned CSV splits.

    Args:
        data_dir: directory containing the train/validation/test CSV files.
        batch_size: number of examples per batch.
        max_length: fixed token length after padding/truncation.
        min_freq: minimum token frequency used to build the vocabulary from the train split.
        shuffle_train: whether the training loader shuffles batches.
        return_vocab: if True, also return the built vocabulary as the final item.

    Returns:
        train_loader, val_loader, test_loader[, vocab]
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    train_df, val_df, test_df = load_splits(data_dir=data_dir)

    vocab = build_vocab(train_df["text"].astype(str).tolist(), min_freq=min_freq)

    train_ds = NewsDataset(train_df["text"].astype(str), train_df["label"], vocab, max_length=max_length)
    val_ds = NewsDataset(val_df["text"].astype(str), val_df["label"], vocab, max_length=max_length)
    test_ds = NewsDataset(test_df["text"].astype(str), test_df["label"], vocab, max_length=max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    if return_vocab:
        return train_loader, val_loader, test_loader, vocab
    return train_loader, val_loader, test_loader


__all__ = ["NewsDataset", "make_loaders", "Vocab", "build_vocab"]