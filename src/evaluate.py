"""Evaluation utilities for the text classifier.

Après l'entraînement
--------------------
- On compare les prédictions du modèle aux vraies classes.
- Accuracy = proportion de bons prédictions globales.
- Macro-F1 = moyenne des F1 par classe, utile quand les classes sont
  déséquilibrées.
- La matrice de confusion montre précisément quels types d'erreurs sont faits.
- La baseline TF-IDF sert de point de comparaison avec un modèle classique.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

try:  # pragma: no cover
    from .dataset import make_loaders
    from .model import MeanPoolTextClassifier
except ImportError:  # pragma: no cover
    from dataset import make_loaders
    from model import MeanPoolTextClassifier


def save_confusion_matrix(matrix: np.ndarray, save_dir: Path) -> None:
    """Save a readable confusion matrix plot for the final evaluation."""
    fig_dir = save_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="Counts")
    plt.xticks(range(matrix.shape[1]), [str(i) for i in range(matrix.shape[1])])
    plt.yticks(range(matrix.shape[0]), [str(i) for i in range(matrix.shape[0])])
    plt.title("Confusion matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black" if matrix[i, j] < matrix.max() / 2 else "white")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=200)
    plt.close()


def evaluate_model(model: torch.nn.Module, loader, device: torch.device):
    """Return loss, accuracy and macro-F1 for a PyTorch model on a loader."""
    # On évalue le modèle en mode inference, sans rétropropagation.
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            losses.append(criterion(logits, labels).item())
            predictions = logits.argmax(dim=-1)
            all_targets.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    loss = float(np.mean(losses))
    acc = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average="macro")
    return loss, acc, f1


def confusion_report(model, loader, device: torch.device):
    """Return the confusion matrix and a list of predicted/true class labels."""
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            predictions = logits.argmax(dim=-1)
            all_targets.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    matrix = confusion_matrix(all_targets, all_predictions, labels=list(range(5)))
    return matrix, all_targets, all_predictions


def tfidf_baseline(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels):
    """Train a TF-IDF + logistic regression baseline and return metrics."""
    model = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_train = model.fit_transform(train_texts)
    X_val = model.transform(val_texts)
    X_test = model.transform(test_texts)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_labels)

    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    return {
        "val_accuracy": accuracy_score(val_labels, val_pred),
        "val_macro_f1": f1_score(val_labels, val_pred, average="macro"),
        "test_accuracy": accuracy_score(test_labels, test_pred),
        "test_macro_f1": f1_score(test_labels, test_pred, average="macro"),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained text classifier")
    parser.add_argument("--model-path", default="models/classifier.pt", help="Checkpoint file produced by train.py")
    parser.add_argument("--data-dir", default="datasets", help="Dataset directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--max-length", type=int, default=1800, help="Fixed sequence length")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, vocab = make_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        return_vocab=True,
    )

    checkpoint = torch.load(args.model_path, map_location=device)
    model = MeanPoolTextClassifier(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=checkpoint["num_classes"],
        pad_idx=checkpoint["pad_idx"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, device)
    test_loss, test_acc, test_f1 = evaluate_model(model, test_loader, device)
    matrix, _, _ = confusion_report(model, test_loader, device)

    save_dir = Path("models")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(matrix, save_dir)

    print(f"Validation  | loss={val_loss:.4f} | acc={val_acc:.4f} | macro_f1={val_f1:.4f}")
    print(f"Test        | loss={test_loss:.4f} | acc={test_acc:.4f} | macro_f1={test_f1:.4f}")
    print("Confusion matrix:")
    print(matrix)
    print(f"Saved confusion matrix plot to {save_dir / 'figures' / 'confusion_matrix.png'}")

    train_df, val_df, test_df = __import__("data").load_splits(data_dir=args.data_dir)
    baseline = tfidf_baseline(
        train_df["text"].astype(str).tolist(),
        train_df["label"].tolist(),
        val_df["text"].astype(str).tolist(),
        val_df["label"].tolist(),
        test_df["text"].astype(str).tolist(),
        test_df["label"].tolist(),
    )
    print("TF-IDF baseline:")
    print(json.dumps(baseline, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
