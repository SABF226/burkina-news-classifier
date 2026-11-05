"""A simple text classifier built from embeddings and mean pooling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPoolTextClassifier(nn.Module):
    """Embed token IDs, average the word vectors, and classify the result."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 5,
        pad_idx: int = 0,
    ):
        super().__init__()
        if vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1")
        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embedding_dim and hidden_dim must be positive")

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of token ids with shape [batch, seq_len]."""
        embedded = self.embedding(input_ids)
        pooled = embedded.mean(dim=1)
        hidden = self.relu(self.fc1(pooled))
        logits = self.fc2(hidden)
        return logits

    def predict_proba(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities for the class scores."""
        logits = self(input_ids)
        return F.softmax(logits, dim=-1)


def build_model(vocab_size: int, num_classes: int = 5, **kwargs) -> MeanPoolTextClassifier:
    """Factory for the default text-classification architecture."""
    return MeanPoolTextClassifier(vocab_size=vocab_size, num_classes=num_classes, **kwargs)


__all__ = ["MeanPoolTextClassifier", "build_model"]
