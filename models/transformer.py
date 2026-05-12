"""
Transformer forecaster built with PyTorch.

Architecture:
  Input → Linear projection (1 → d_model) → TransformerEncoder → Linear head

The Transformer processes the entire input sequence in parallel via
self-attention, making it theoretically capable of capturing long-range
dependencies without the vanishing-gradient problem of RNNs.

Training and prediction use the same normalisation + recursive strategy
as the LSTM and GRU models.
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


class _TransformerNet(nn.Module):
    """Transformer encoder that maps a price sequence to a single prediction."""

    def __init__(self, d_model: int, nhead: int, num_layers: int) -> None:
        super().__init__()
        # Project each scalar input value into d_model dimensions
        # so the Transformer encoder has enough capacity to learn representations
        self.input_proj = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=d_model * 4,   # standard 4× expansion in the FFN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Map the last sequence position's representation to a scalar output
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1)
        x   = self.input_proj(x)   # → (batch, seq_len, d_model)
        out = self.encoder(x)      # self-attention across the sequence
        return self.fc(out[:, -1, :]).squeeze(-1)   # predict from the last position


class TransformerForecaster(BaseForecaster):

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

        # Z-score normalisation: zero mean, unit variance
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

        self._model = _TransformerNet(
            self._cfg["d_model"], self._cfg["nhead"], self._cfg["num_layers"]
        ).to(self._device)
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
                seq.append(val)   # recursive: feed prediction back as next input

        # Denormalise: reverse the z-score applied before training
        return np.array(preds) * self._std + self._mean

    @property
    def name(self) -> str:
        return "Transformer"
