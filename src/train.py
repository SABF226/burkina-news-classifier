"""Training loop for the Burkina News text classifier.

Avant l'entraînement
-------------------
- On charge les données nettoyées et on construit un vocabulaire sur le split
  train uniquement.
- On transforme chaque article en séquences d'indices numériques.
- On crée des DataLoaders pour envoyer les batchs au modèle.

Pendant l'entraînement
----------------------
- Le modèle calcule des logits sur chaque batch.
- La fonction de perte CrossEntropyLoss compare ces logits aux vraies classes.
- L'optimiseur Adam met à jour les poids du modèle.

Après l'entraînement
--------------------
- On évalue le modèle sur la validation puis le test.
- On sauvegarde le meilleur état du modèle + historique des métriques.
- On produit aussi des courbes visuelles pour suivre l'évolution de la perte et
  des scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def save_training_plots(history: dict, save_dir: Path) -> None:
    """Generate simple training curves to visualize the optimization process."""
    fig_dir = save_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history["train_loss"], label="train loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="validation loss", marker="s")
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "train_validation_loss.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history["val_acc"], label="validation accuracy", marker="o")
    plt.plot(epochs, history["val_f1"], label="validation macro-F1", marker="s")
    plt.title("Validation metrics over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "validation_metrics.png", dpi=200)
    plt.close()


def evaluate(model: nn.Module, loader, device: torch.device):
    """Compute loss, accuracy and macro-F1 for a validation/test loader."""
    # On passe le modèle en mode évaluation : les couches de dropout/batchnorm
    # ne calculent plus de mises à jour stochastiques.
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
    # On repasse le modèle en mode entraînement pour autoriser les calculs de
    # gradient et les couches sensibles à l'apprentissage.
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 1) remise à zéro des gradients précédents
        optimizer.zero_grad()
        # 2) forward pass : le modèle produit des logits pour chaque classe
        logits = model(inputs)
        # 3) calcul de la perte
        loss = criterion(logits, labels)
        # 4) rétropropagation : calcul des gradients
        loss.backward()
        # 5) mise à jour des poids par Adam
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

    # On conserve l'historique de la courbe d'apprentissage pour pouvoir visualiser
    # l'évolution sur les époques et vérifier si le modèle apprend ou surapprend.
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "test_loss": [], "test_acc": [], "test_f1": []}

    for epoch in range(1, args.epochs + 1):
        # Une époque = un passage complet sur tout le dataset d'entraînement.
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        # Evaluation régulière sur la validation pour surveiller la généralisation.
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

    # Le test est utilisé seulement en fin d'entraînement pour mesurer la
    # performance réelle dans des données jamais vues.
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

    # On sauvegarde aussi le vocabulaire pour qu'une future inférence utilise les
    # mêmes index et le même mapping que lors de l'entraînement.
    vocab.save(str(save_dir / "vocab.json"))
    save_training_plots(history, save_dir)

    print(f"\nFinal test metrics | loss={test_loss:.4f} | acc={test_acc:.4f} | macro_f1={test_f1:.4f}")
    print(f"Saved model to {save_dir / 'classifier.pt'}")
    print(f"Saved plots to {save_dir / 'figures'}")


if __name__ == "__main__":
    main()
