from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.config import LSTMConfig


class _LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float, n_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])  # последний слой, последний шаг


class LSTMModel:
    name = "lstm"

    def __init__(self, cfg: LSTMConfig, n_classes: int = 4):
        self.cfg = cfg
        self.n_classes = n_classes
        self.net: _LSTM | None = None
        self.device = torch.device("cpu")

    def _to_device(self, device: torch.device) -> None:
        self.device = device
        if self.net:
            self.net.to(device)

    def fit(
        self,
        X: np.ndarray,  # (N, T, n_mfcc)
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        device: torch.device | None = None,
    ) -> None:
        if device:
            self._to_device(device)

        input_size = X.shape[2]
        self.net = _LSTM(
            input_size, self.cfg.hidden_size, self.cfg.num_layers,
            self.cfg.dropout, self.n_classes
        ).to(self.device)

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
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            if X_val is not None and (epoch + 1) % 10 == 0:
                val_acc = (self.predict(X_val) == y_val).mean()
                print(f"  [LSTM] epoch {epoch+1}/{self.cfg.epochs}  "
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

    def load(self, path: Path, input_size: int) -> None:
        self.net = _LSTM(
            input_size, self.cfg.hidden_size, self.cfg.num_layers,
            self.cfg.dropout, self.n_classes
        ).to(self.device)
        self.net.load_state_dict(torch.load(path, map_location=self.device))
