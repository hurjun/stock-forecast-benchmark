"""
GRU (Gated Recurrent Unit) forecaster built with PyTorch.

GRU is a lighter variant of LSTM: it uses two gates (reset + update) instead
of three (input + forget + output), which reduces parameter count and can
train faster while achieving comparable accuracy on shorter sequences.

Training and prediction logic is identical to lstm.py — only the recurrent
cell type differs.
"""

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseForecaster

logger = logging.getLogger(__name__)
_SEED = 42


class _GRUNet(nn.Module):
    """Two-layer GRU followed by a single linear output neuron."""

    def __init__(self, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.gru = nn.GRU(1, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)   # use last time-step


class GRUForecaster(BaseForecaster):

    def __init__(self, config: dict) -> None:
        self._cfg    = config
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model  = None
        self._mean   = 0.0
        self._std    = 1.0
        self._last_seq: np.ndarray = np.array([])

    def fit(self, train: pd.DataFrame) -> None:
        torch.manual_seed(_SEED)
        series = train["Close"].values.astype(np.float32)

        # Z-score normalisation for stable gradient training
        self._mean = series.mean()
        self._std  = series.std()
        norm       = (series - self._mean) / self._std

        seq_len = self._cfg["seq_len"]
        X = np.array([norm[i : i + seq_len] for i in range(len(norm) - seq_len)])
        y = norm[seq_len:]

        loader = DataLoader(
            TensorDataset(
                torch.tensor(X).unsqueeze(-1),
                torch.tensor(y),
            ),
            batch_size=self._cfg["batch_size"],
            shuffle=True,
        )

        self._model = _GRUNet(self._cfg["hidden_size"], self._cfg["num_layers"]).to(self._device)
        opt     = torch.optim.Adam(self._model.parameters(), lr=self._cfg["lr"])
        loss_fn = nn.MSELoss()

        self._model.train()
        for _ in range(self._cfg["epochs"]):
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                opt.zero_grad()
                loss_fn(self._model(xb), yb).backward()
                opt.step()

        self._last_seq = norm[-seq_len:]
        logger.info("%s fitted successfully", self.name)

    def predict(self, n_steps: int) -> np.ndarray:
        self._model.eval()
        seq  = list(self._last_seq)
        preds: list[float] = []

        with torch.no_grad():
            for _ in range(n_steps):
                x = (
                    torch.tensor(seq[-self._cfg["seq_len"] :], dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .to(self._device)
                )
                val = self._model(x).item()
                preds.append(val)
                seq.append(val)   # recursive: prediction becomes next input

        # Denormalise back to original price scale
        return np.array(preds) * self._std + self._mean

    @property
    def name(self) -> str:
        return "GRU"
