"""
Train the SAINT classifier on NSL-KDD and persist all artifacts.

Usage:
    python scripts/train.py
"""

import json
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.loader import load_dataset, save_artifacts
from model.classifier import evaluate, save_model, train

if __name__ == "__main__":
    print("=== Loading NSL-KDD training data ===")
    X_train_full, y_train_full, scaler, encoders, feature_cols = load_dataset("train")
    print(f"Train shape: {X_train_full.shape}  |  Classes: {set(y_train_full)}")

    # Persist feature column order so the API can align live inputs
    feat_path = Path("model/feature_cols.json")
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f)
    print(f"Feature columns saved → {feat_path}")

    # Hold-out split (10% validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
    )

    print("\n=== Training ===")
    model = train(X_train, y_train, X_val, y_val, epochs=30, batch_size=1024)

    print("\n=== Validation evaluation ===")
    val_acc = evaluate(model, X_val, y_val)
    print(f"\nValidation accuracy: {val_acc:.4f}")

    print("\n=== Test evaluation ===")
    X_test, y_test, *_ = load_dataset("test", scaler=scaler, encoders=encoders, reference_cols=feature_cols)
    test_acc = evaluate(model, X_test, y_test)
    print(f"\nTest accuracy: {test_acc:.4f}")

    print("\n=== Saving artifacts ===")
    save_model(model)
    save_artifacts(scaler, encoders)
    print("Done.")
