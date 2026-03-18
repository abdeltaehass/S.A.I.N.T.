"""
SAINT classifier — PyTorch feed-forward network for NSL-KDD 5-class detection.

Architecture: BatchNorm → [Linear → BatchNorm → ReLU → Dropout] × N → Linear
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    MODEL_DROPOUT, MODEL_HIDDEN_DIMS, MODEL_INPUT_DIM,
    MODEL_NUM_CLASSES, MODEL_PATH, IDX_TO_CATEGORY,
)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class SAINTClassifier(nn.Module):
    """
    Multi-layer perceptron with BatchNorm + Dropout.
    Input dim is determined at runtime from actual feature count.
    """

    def __init__(
        self,
        input_dim: int = MODEL_INPUT_DIM,
        hidden_dims: list[int] = MODEL_HIDDEN_DIMS,
        num_classes: int = MODEL_NUM_CLASSES,
        dropout: float = MODEL_DROPOUT,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.BatchNorm1d(input_dim)]
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 1e-3,
    device: str | None = None,
) -> SAINTClassifier:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    input_dim = X_train.shape[1]
    model = SAINTClassifier(input_dim=input_dim).to(device)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True
    )

    # Class weights to handle NSL-KDD imbalance (U2R/R2L are rare)
    class_counts = np.bincount(y_train, minlength=MODEL_NUM_CLASSES).astype(float)
    class_weights = torch.tensor(
        1.0 / (class_counts + 1e-6), dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
        scheduler.step()

        acc = correct / len(X_train)
        val_str = ""
        if X_val is not None and y_val is not None:
            val_acc = evaluate(model, X_val, y_val, device=device, verbose=False)
            val_str = f"  val_acc={val_acc:.4f}"
        print(
            f"Epoch {epoch:03d}/{epochs}  loss={total_loss/len(X_train):.4f}"
            f"  train_acc={acc:.4f}{val_str}"
        )

    return model


def evaluate(
    model: SAINTClassifier,
    X: np.ndarray,
    y: np.ndarray,
    device: str | None = None,
    verbose: bool = True,
) -> float:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    X_t = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_t)
        preds = logits.argmax(1).cpu().numpy()

    acc = (preds == y).mean()
    if verbose:
        labels = list(IDX_TO_CATEGORY.values())
        print(classification_report(y, preds, target_names=labels, zero_division=0))
        print("Confusion matrix:")
        print(confusion_matrix(y, preds))
    return float(acc)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: SAINTClassifier, path: str = MODEL_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "input_dim": model.net[0].num_features},
        path,
    )
    print(f"Model saved → {path}")


def load_model(path: str = MODEL_PATH, device: str | None = None) -> SAINTClassifier:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = SAINTClassifier(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
