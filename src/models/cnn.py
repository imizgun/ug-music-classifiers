from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.config import CNNConfig


class _CNN(nn.Module):
    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CNNModel:
    name = "cnn"

    def __init__(self, cfg: CNNConfig, n_classes: int = 4):
        self.cfg = cfg
        self.n_classes = n_classes
        self.net: _CNN | None = None
        self.device = torch.device("cpu")

    def _to_device(self, device: torch.device) -> None:
        self.device = device
        if self.net:
            self.net.to(device)

    def fit(
        self,
        X: np.ndarray,  # (N, 1, H, W)
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        device: torch.device | None = None,
    ) -> None:
        if device:
            self._to_device(device)

        self.net = _CNN(self.n_classes).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.epochs
        )
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
            scheduler.step()

            if X_val is not None and (epoch + 1) % 5 == 0:
                val_acc = (self.predict(X_val) == y_val).mean()
                print(f"  [CNN] epoch {epoch+1}/{self.cfg.epochs}  "
                      f"loss={total_loss/len(loader):.4f}  val_acc={val_acc:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.tensor(X, dtype=torch.float32).to(self.device))
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.tensor(X, dtype=torch.float32).to(self.device))
        return torch.softmax(logits, dim=1).cpu().numpy()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def load(self, path: Path) -> None:
        self.net = _CNN(self.n_classes).to(self.device)
        self.net.load_state_dict(torch.load(path, map_location=self.device))
