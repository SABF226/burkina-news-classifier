"""Training loop for the Burkina News text classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

try:  # pragma: no cover - keep direct and package imports working
    from .dataset import make_loaders
    from .model import build_model
except ImportError:  # pragma: no cover
    from dataset import make_loaders
    from model import build_model


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds for Python, NumPy and PyTorch."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader, device: torch.device):
    """Compute loss, accuracy and macro-F1 for a validation/test loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=-1)
            all_targets.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average="macro")
    return avg_loss, acc, f1


def train_one_epoch(model: nn.Module, loader, optimizer, device: torch.device):
    """Run one training epoch and return the mean loss."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a text classifier on MasakhaNEWS French data")
    parser.add_argument("--data-dir", default="datasets", help="Directory containing the cleaned CSV files")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for the training and validation loaders")
    parser.add_argument("--max-length", type=int, default=1800, help="Maximum token length after padding/truncation")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save-dir", default="models", help="Directory for model and metrics artifacts")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, vocab = make_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        return_vocab=True,
    )

    model = build_model(
        vocab_size=len(vocab.token2id),
        num_classes=5,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        pad_idx=vocab.pad_idx,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "test_loss": [], "test_acc": [], "test_f1": []}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

    test_loss, test_acc, test_f1 = evaluate(model, test_loader, device)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)
    history["test_f1"].append(test_f1)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": len(vocab.token2id),
            "num_classes": 5,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "pad_idx": vocab.pad_idx,
            "history": history,
        },
        save_dir / "classifier.pt",
    )

    with open(save_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    vocab.save(str(save_dir / "vocab.json"))

    print(f"\nFinal test metrics | loss={test_loss:.4f} | acc={test_acc:.4f} | macro_f1={test_f1:.4f}")
    print(f"Saved model to {save_dir / 'classifier.pt'}")


if __name__ == "__main__":
    main()
