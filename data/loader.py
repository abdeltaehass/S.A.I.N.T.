"""
NSL-KDD dataset loader and preprocessor.

Downloads KDDTrain+.txt / KDDTest+.txt from the UNB repository if not present,
then engineers the 41-feature pipeline used by the SAINT classifier.
"""

import os
import pickle
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    ATTACK_TYPE_MAP, CATEGORICAL_FEATURES, FEATURE_NAMES,
    NUMERIC_FEATURES, ATTACK_CATEGORIES, SCALER_PATH, ENCODER_PATH,
)

# ---------------------------------------------------------------------------
# NSL-KDD column schema (41 features + label + difficulty score)
# ---------------------------------------------------------------------------
_COLUMNS = FEATURE_NAMES + ["attack_type", "difficulty"]

# Remote URLs (UNB public mirror)
_TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
_TEST_URL  = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"

DATA_DIR = Path(__file__).resolve().parent / "raw"


def _download(url: str, dest: Path) -> None:
    print(f"Downloading {dest.name} ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")


def _load_raw(split: str = "train") -> pd.DataFrame:
    """Load raw NSL-KDD CSV; download if missing."""
    url  = _TRAIN_URL if split == "train" else _TEST_URL
    dest = DATA_DIR / ("KDDTrain+.txt" if split == "train" else "KDDTest+.txt")
    if not dest.exists():
        _download(url, dest)
    df = pd.read_csv(dest, header=None, names=_COLUMNS)
    return df


def _map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Convert specific attack names → 5-class category integers."""
    df = df.copy()
    df["attack_type"] = df["attack_type"].str.strip().str.lower()
    df["label"] = df["attack_type"].map(ATTACK_TYPE_MAP).map(ATTACK_CATEGORIES)
    # Drop rows with unknown attack types (shouldn't happen, but just in case)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df


def _encode_categoricals(
    df: pd.DataFrame,
    encoders: dict | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """One-hot encode protocol_type, service, flag."""
    df = df.copy()
    if fit:
        encoders = {}
        for col in CATEGORICAL_FEATURES:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES)
    else:
        for col in CATEGORICAL_FEATURES:
            le = encoders[col]
            # Handle unseen labels gracefully
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES)
    return df, encoders


def _scale_numerics(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """StandardScale numeric columns (present in df)."""
    df = df.copy()
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    if fit:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])
    return df, scaler


def load_dataset(
    split: str = "train",
    scaler: StandardScaler | None = None,
    encoders: dict | None = None,
    reference_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, dict, list[str]]:
    """
    Full pipeline: raw CSV → (X, y, scaler, encoders, feature_cols).

    When split='train', fits and returns new scaler/encoders.
    When split='test', expects pre-fit scaler/encoders (or loads from disk).
    Pass reference_cols (from training) to align test feature dimensions.
    """
    fit = (split == "train")

    if not fit and (scaler is None or encoders is None):
        scaler, encoders = load_artifacts()

    df = _load_raw(split)
    df = _map_labels(df)

    y = df["label"].values
    df = df.drop(columns=["attack_type", "difficulty", "label"])

    df, encoders = _encode_categoricals(df, encoders, fit=fit)
    df, scaler = _scale_numerics(df, scaler, fit=fit)

    # Align to training feature columns when provided (fills missing with 0)
    if reference_cols is not None:
        df = df.reindex(columns=reference_cols, fill_value=0)

    feature_cols = df.columns.tolist()
    X = df.values.astype(np.float32)

    return X, y, scaler, encoders, feature_cols


def save_artifacts(scaler: StandardScaler, encoders: dict) -> None:
    """Persist fitted scaler and encoders to disk."""
    Path(SCALER_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoders, f)
    print(f"Artifacts saved → {SCALER_PATH}, {ENCODER_PATH}")


def load_artifacts() -> tuple[StandardScaler, dict]:
    """Load persisted scaler and encoders from disk."""
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoders = pickle.load(f)
    return scaler, encoders


def preprocess_single(
    raw: dict,
    scaler: StandardScaler,
    encoders: dict,
    feature_cols: list[str],
) -> np.ndarray:
    """
    Preprocess a single connection dict (live inference path).

    raw: dict with keys matching FEATURE_NAMES
    Returns: float32 array of shape (1, num_features)
    """
    df = pd.DataFrame([raw])
    # Fill any missing features with 0
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0

    df, _ = _encode_categoricals(df, encoders, fit=False)
    df, _ = _scale_numerics(df, scaler, fit=False)

    # Align to training feature columns (fill missing with 0)
    df = df.reindex(columns=feature_cols, fill_value=0)
    return df.values.astype(np.float32)
