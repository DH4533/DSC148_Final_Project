"""utils.py — shared helpers."""

import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def temporal_split(df: pd.DataFrame, test_year: int = 2023):
    """
    Train on seasons before test_year, test on test_year.
    This mimics real deployment: you never train on future data.
    """
    df["game_date"] = pd.to_datetime(df["game_date"])
    mask = df["game_date"].dt.year < test_year
    return df[mask], df[~mask]


def stratified_split(X, y, test_size: float = 0.15, val_size: float = 0.10,
                     random_state: int = 42):
    """Train / val / test split preserving class balance."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, stratify=y_temp, random_state=random_state
    )
    print(f"Split: train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def model_save_path(model_name: str) -> str:
    ext = {
        "logistic":      ".joblib",
        "random_forest": ".joblib",
        "xgboost":       ".ubj",
        "neural_net":    ".pt",
    }
    return str(MODELS_DIR / f"{model_name}{ext.get(model_name, '.pkl')}")
