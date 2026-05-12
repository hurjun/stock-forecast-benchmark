"""
LSTM (Long Short-Term Memory) forecaster built with PyTorch.

Architecture:
  Input (1 feature per time step) → LSTM layers → Linear output head

Training:
  - Input sequences of length seq_len (default 60 trading days)
  - Predict the next single value (one-step-ahead supervised learning)
  - Loss: MSE   Optimizer: Adam

Prediction:
  - Recursive / auto-regressive: predict one step, append to sequence, repeat.
  - All computation is done in normalised (z-score) space; predictions are
    denormalised before returning.
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


class _LSTMNet(nn.Module):
    """Two-layer LSTM followed by a single linear output neuron."""

    def __init__(self, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        # input_size=1 because we feed one price value per time step
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
        # Map the last hidden state to a scalar prediction
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        # out: (batch, seq_len, hidden_size) — take the last time step
        return self.fc(out[:, -1, :]).squeeze(-1)   # → (batch,)


class LSTMForecaster(BaseForecaster):

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

        # Z-score normalisation: zero mean, unit variance.
        # This stabilises gradient updates and helps the model converge faster.
        self._mean = series.mean()
        self._std  = series.std()
        norm       = (series - self._mean) / self._std

        seq_len = self._cfg["seq_len"]

        # Slide a window of length seq_len to create (input, target) pairs
        # X[i] = norm[i : i+seq_len]   (the context window)
        # y[i] = norm[i+seq_len]        (the next value to predict)
        X = np.array([norm[i : i + seq_len] for i in range(len(norm) - seq_len)])
        y = norm[seq_len:]

        loader = DataLoader(
            TensorDataset(
                torch.tensor(X).unsqueeze(-1),  # (n, seq_len, 1)
                torch.tensor(y),                 # (n,)
            ),
            batch_size=self._cfg["batch_size"],
            shuffle=True,   # shuffle within each epoch for better generalisation
        )

        self._model = _LSTMNet(self._cfg["hidden_size"], self._cfg["num_layers"]).to(self._device)
        opt     = torch.optim.Adam(self._model.parameters(), lr=self._cfg["lr"])
        loss_fn = nn.MSELoss()

        self._model.train()
        for _ in range(self._cfg["epochs"]):
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                opt.zero_grad()                  # clear gradients
                loss_fn(self._model(xb), yb).backward()  # compute gradients
                opt.step()                       # update weights

        # Store the last seq_len normalised values as the seed for prediction
        self._last_seq = norm[-seq_len:]
        logger.info("%s fitted successfully", self.name)

    def predict(self, n_steps: int) -> np.ndarray:
        self._model.eval()
        seq  = list(self._last_seq)   # mutable copy; will grow during recursion
        preds: list[float] = []

        with torch.no_grad():
            for _ in range(n_steps):
                # Build the input tensor from the most recent seq_len values
                x = (
                    torch.tensor(seq[-self._cfg["seq_len"] :], dtype=torch.float32)
                    .unsqueeze(0)   # add batch dimension
                    .unsqueeze(-1)  # add feature dimension
                    .to(self._device)
                )
                val = self._model(x).item()   # normalised prediction
                preds.append(val)
                seq.append(val)               # feed prediction back as the next input

        # Reverse z-score normalisation to get real price values
        return np.array(preds) * self._std + self._mean

    @property
    def name(self) -> str:
        return "LSTM"
