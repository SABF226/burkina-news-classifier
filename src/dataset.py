"""PyTorch Dataset for the MasakhaNEWS articles."""

from typing import Tuple

import torch
from torch.utils.data import Dataset


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