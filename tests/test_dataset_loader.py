import sys
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from dataset import make_loaders
from model import MeanPoolTextClassifier


def test_make_loaders_generates_expected_batch_shapes():
    train_loader, val_loader, test_loader = make_loaders(batch_size=8, max_length=12)

    batch_texts, batch_labels = next(iter(train_loader))
    assert batch_texts.shape == (8, 12)
    assert batch_labels.shape == (8,)

    val_texts, val_labels = next(iter(val_loader))
    assert val_texts.shape[1] == 12
    assert val_labels.shape[0] == val_texts.shape[0]

    test_texts, test_labels = next(iter(test_loader))
    assert test_texts.shape[1] == 12
    assert test_labels.shape[0] == test_texts.shape[0]


def test_model_forward_returns_class_logits():
    model = MeanPoolTextClassifier(vocab_size=50, embedding_dim=16, hidden_dim=8, num_classes=5)
    batch = torch.randint(0, 49, (4, 12), dtype=torch.long)

    logits = model(batch)

    assert logits.shape == (4, 5)
    assert logits.dtype == torch.float32
