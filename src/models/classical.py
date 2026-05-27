from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.config import RandomForestConfig, KNNConfig


class RandomForestModel:
    name = "random_forest"

    def __init__(self, cfg: RandomForestConfig):
        self.cfg = cfg
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                random_state=42,
                n_jobs=-1,
            )),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray, **_) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def feature_importances(self) -> np.ndarray:
        return self.model.named_steps["clf"].feature_importances_

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: Path, cfg: RandomForestConfig) -> "RandomForestModel":
        obj = cls(cfg)
        with open(path, "rb") as f:
            obj.model = pickle.load(f)
        return obj


class KNNModel:
    name = "knn"

    def __init__(self, cfg: KNNConfig):
        self.cfg = cfg
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=cfg.k,
                metric=cfg.metric,
                n_jobs=-1,
            )),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray, **_) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: Path, cfg: KNNConfig) -> "KNNModel":
        obj = cls(cfg)
        with open(path, "rb") as f:
            obj.model = pickle.load(f)
        return obj
