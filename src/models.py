"""
models.py
---------
Four models for swing-and-miss prediction:
  1. Logistic Regression  (sklearn)
  2. Random Forest         (sklearn)
  3. XGBoost               (xgboost)
  4. Neural Network (MLP)  (PyTorch)

Each model is wrapped in a common interface:
  model.fit(X_train, y_train)
  model.predict(X)          → binary labels
  model.predict_proba(X)    → probability of swinging strike
  model.save(path)
  model.load(path)          (class method)
"""

import os
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# 1. Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────
class LogisticModel:
    name = "logistic"

    def __init__(self, C: float = 0.1, max_iter: int = 1000):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=C,
                max_iter=max_iter,
                class_weight="balanced",   # handle class imbalance
                solver="lbfgs",
                n_jobs=-1,
            )),
        ])

    def fit(self, X, y):
        print("  Training Logistic Regression …")
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)[:, 1]

    def save(self, path: str):
        joblib.dump(self.pipeline, path)
        print(f"  Saved → {path}")

    @classmethod
    def load(cls, path: str):
        obj = cls.__new__(cls)
        obj.pipeline = joblib.load(path)
        return obj

    def feature_importance(self, feature_names):
        coef = self.pipeline.named_steps["clf"].coef_[0]
        return pd.Series(np.abs(coef), index=feature_names).sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Random Forest
# ─────────────────────────────────────────────────────────────────────────────
class RandomForestModel:
    name = "random_forest"

    def __init__(self, n_estimators: int = 300, max_depth: int = 12,
                 min_samples_leaf: int = 50):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )

    def fit(self, X, y):
        print("  Training Random Forest …")
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)[:, 1]

    def save(self, path: str):
        joblib.dump(self.clf, path)
        print(f"  Saved → {path}")

    @classmethod
    def load(cls, path: str):
        obj = cls.__new__(cls)
        obj.clf = joblib.load(path)
        return obj

    def feature_importance(self, feature_names):
        return pd.Series(
            self.clf.feature_importances_, index=feature_names
        ).sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# 3. XGBoost
# ─────────────────────────────────────────────────────────────────────────────
class XGBoostModel:
    name = "xgboost"

    def __init__(self, n_estimators: int = 500, max_depth: int = 6,
                 learning_rate: float = 0.05, subsample: float = 0.8,
                 colsample_bytree: float = 0.8):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",   # fast even on CPU
        )
        self.clf = xgb.XGBClassifier(**self.params)

    def fit(self, X, y, X_val=None, y_val=None):
        print("  Training XGBoost …")
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.clf.fit(
            X, y,
            eval_set=eval_set,
            verbose=100,
        )
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)[:, 1]

    def save(self, path: str):
        self.clf.save_model(path)
        print(f"  Saved → {path}")

    @classmethod
    def load(cls, path: str):
        obj = cls.__new__(cls)
        obj.clf = xgb.XGBClassifier()
        obj.clf.load_model(path)
        return obj

    def feature_importance(self, feature_names):
        scores = self.clf.get_booster().get_fscore()
        s = pd.Series(scores).reindex(
            [f"f{i}" for i in range(len(feature_names))]
        ).fillna(0)
        s.index = feature_names
        return s.sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Neural Network (PyTorch MLP)
# ─────────────────────────────────────────────────────────────────────────────
class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class NeuralNetModel:
    name = "neural_net"

    def __init__(self, hidden_dims: list[int] = None, dropout: float = 0.3,
                 lr: float = 1e-3, batch_size: int = 2048, epochs: int = 30,
                 device: str = None):
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.dropout     = dropout
        self.lr          = lr
        self.batch_size  = batch_size
        self.epochs      = epochs
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model       = None
        self.scaler      = StandardScaler()

    # -- helpers --------------------------------------------------------------
    def _to_tensor(self, X):
        return torch.FloatTensor(X).to(self.device)

    def _pos_weight(self, y):
        """Class imbalance correction via pos_weight in BCEWithLogitsLoss."""
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        return torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32).to(self.device)

    # -- public interface -----------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None):
        print(f"  Training Neural Net on {self.device} …")
        X_sc = self.scaler.fit_transform(X)

        y_arr  = np.array(y, dtype=np.float32)
        Xt, yt = self._to_tensor(X_sc), torch.FloatTensor(y_arr).to(self.device)

        input_dim = X_sc.shape[1]
        self.model = _MLP(input_dim, self.hidden_dims, self.dropout).to(self.device)
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight(y_arr))

        dataset = TensorDataset(Xt, yt)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)
            scheduler.step()

            if epoch % 5 == 0 or epoch == 1:
                avg_loss = total_loss / len(dataset)
                print(f"    Epoch {epoch:3d}/{self.epochs}  loss={avg_loss:.4f}")

        return self

    def predict_proba(self, X):
        self.model.eval()
        X_sc = self.scaler.transform(X)
        with torch.no_grad():
            logits = self.model(self._to_tensor(X_sc))
            probs  = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "hidden_dims":  self.hidden_dims,
            "dropout":      self.dropout,
            "scaler":       self.scaler,
        }, path)
        print(f"  Saved → {path}")

    @classmethod
    def load(cls, path: str):
        ckpt = torch.load(path, map_location="cpu")
        # Determine input dim from first weight matrix
        first_weight = ckpt["model_state"]["net.0.weight"]
        input_dim    = first_weight.shape[1]

        obj = cls(hidden_dims=ckpt["hidden_dims"], dropout=ckpt["dropout"])
        obj.model  = _MLP(input_dim, ckpt["hidden_dims"], ckpt["dropout"])
        obj.model.load_state_dict(ckpt["model_state"])
        obj.model.eval()
        obj.scaler = ckpt["scaler"]
        return obj

    def feature_importance(self, feature_names):
        """Gradient-based feature importance (mean |grad| on first layer weights)."""
        w = self.model.net[0].weight.detach().cpu().numpy()
        importance = np.abs(w).mean(axis=0)
        return pd.Series(importance, index=feature_names).sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "logistic":     LogisticModel,
    "random_forest": RandomForestModel,
    "xgboost":      XGBoostModel,
    "neural_net":   NeuralNetModel,
}

def get_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
