from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from src.config import MLPConfig


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int], dropout: float, n_classes: int):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPModel:
    name = "mlp"

    def __init__(self, cfg: MLPConfig, n_classes: int = 4):
        self.cfg = cfg
        self.n_classes = n_classes
        self.scaler = StandardScaler()
        self.net: _MLP | None = None
        self.device = torch.device("cpu")

    def _to_device(self, device: torch.device) -> None:
        self.device = device
        if self.net:
            self.net.to(device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        device: torch.device | None = None,
    ) -> None:
        if device:
            self._to_device(device)

        X = self.scaler.fit_transform(X)
        self.net = _MLP(X.shape[1], self.cfg.hidden_layers, self.cfg.dropout, self.n_classes)
        self.net.to(self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        criterion = nn.CrossEntropyLoss()
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        for epoch in range(self.cfg.epochs):
            self.net.train()
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.net(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if X_val is not None and (epoch + 1) % 10 == 0:
                val_acc = (self.predict(X_val, already_scaled=True) == y_val).mean()
                print(f"  [MLP] epoch {epoch+1}/{self.cfg.epochs}  "
                      f"loss={total_loss/len(loader):.4f}  val_acc={val_acc:.4f}")

    def predict(self, X: np.ndarray, already_scaled: bool = False) -> np.ndarray:
        if not already_scaled:
            X = self.scaler.transform(X)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.tensor(X, dtype=torch.float32).to(self.device))
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self.scaler.transform(X)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.tensor(X, dtype=torch.float32).to(self.device))
        return torch.softmax(logits, dim=1).cpu().numpy()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(), "scaler": self.scaler}, path)

    def load(self, path: Path, input_dim: int) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.net = _MLP(input_dim, self.cfg.hidden_layers, self.cfg.dropout, self.n_classes)
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.to(self.device)
        self.scaler = checkpoint["scaler"]
