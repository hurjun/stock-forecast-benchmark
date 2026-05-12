"""
TCN (Temporal Convolutional Network) forecaster built with PyTorch.

Architecture:
  Input (1 feature) → stack of Residual Blocks → Linear output head

Each Residual Block contains two dilated causal Conv1d layers:
  - Causal: the convolution only looks at past and present, never future values.
  - Dilated: dilation doubles at each block (1, 2, 4, …), so the receptive
    field grows exponentially with depth — a seq_len of 60 is covered in just
    a few blocks without the vanishing-gradient problems of deep RNNs.
  - Residual connection: allows gradients to flow directly to early layers,
    making training more stable.

Why TCN instead of more LSTM/GRU layers?
  - Parallelisable: Conv1d operations run in parallel over the time axis;
    RNNs must step sequentially. TCNs train ~3× faster than LSTM on CPU.
  - Fixed receptive field: easier to reason about what context the model sees.
  - Competitive accuracy: often matches or beats LSTMs on financial sequences.

Prediction:
  Recursive (auto-regressive): predict one step → append → repeat.
  All computation is done in normalised (z-score) space.
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


class _CausalConv1d(nn.Conv1d):
    """
    Conv1d with left-only padding so the model never sees future time steps.

    Standard Conv1d with padding='same' pads both sides, which would leak
    future information. Here we pad only the left (past) side and trim the
    extra right-side output that PyTorch appends.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
        # Left padding = (kernel_size - 1) * dilation ensures the output length
        # equals the input length with no right-side lookahead.
        self._pad = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, dilation=dilation, padding=self._pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        # Remove the extra frames appended on the right by symmetric padding
        return out[:, :, : -self._pad] if self._pad > 0 else out


class _ResidualBlock(nn.Module):
    """
    Two stacked causal dilated convolutions with a residual (skip) connection.

    The skip connection lets gradients bypass the two conv layers entirely,
    so the network can learn both 'apply a transformation' and 'do nothing'
    at each block — this is key for stable training with many blocks.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(in_ch,  out_ch, kernel_size, dilation)
        self.conv2 = _CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.relu  = nn.ReLU()
        # 1×1 conv to project channels when in_ch ≠ out_ch (only the first block)
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)   # residual addition before final activation


class _TCNNet(nn.Module):
    """
    Full TCN: a stack of Residual Blocks with exponentially growing dilation,
    followed by a linear head that reads the last time step's features.
    """

    def __init__(self, num_channels: list[int], kernel_size: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1                        # single input feature (price)
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i            # 1, 2, 4, 8, … exponential receptive field
            layers.append(_ResidualBlock(in_ch, out_ch, kernel_size, dilation))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.fc      = nn.Linear(num_channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, seq_len)  — channels-first format required by Conv1d
        out = self.network(x)            # → (batch, channels, seq_len)
        return self.fc(out[:, :, -1]).squeeze(-1)  # read only the last time step


class TCNForecaster(BaseForecaster):

    def __init__(self, config: dict) -> None:
        self._cfg     = config
        self._device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model   = None
        self._mean    = 0.0
        self._std     = 1.0
        self._last_seq: np.ndarray = np.array([])

    def fit(self, train: pd.DataFrame) -> None:
        torch.manual_seed(_SEED)
        series = train["Close"].values.astype(np.float32)

        # Z-score normalisation so all tickers live on the same scale
        self._mean = series.mean()
        self._std  = series.std()
        norm       = (series - self._mean) / self._std

        seq_len = self._cfg["seq_len"]

        # Build sliding-window supervised dataset
        X = np.array([norm[i : i + seq_len] for i in range(len(norm) - seq_len)])
        y = norm[seq_len:]

        # TCN expects channels-first: (batch, 1, seq_len)
        loader = DataLoader(
            TensorDataset(
                torch.tensor(X).unsqueeze(1),   # (n, 1, seq_len)
                torch.tensor(y),                 # (n,)
            ),
            batch_size=self._cfg["batch_size"],
            shuffle=True,
        )

        self._model = _TCNNet(
            num_channels=self._cfg["num_channels"],
            kernel_size=self._cfg["kernel_size"],
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
        seq   = list(self._last_seq)
        preds: list[float] = []

        with torch.no_grad():
            for _ in range(n_steps):
                x = (
                    torch.tensor(seq[-self._cfg["seq_len"] :], dtype=torch.float32)
                    .unsqueeze(0)    # batch dimension
                    .unsqueeze(0)    # channel dimension — TCN is channels-first
                    .to(self._device)
                )
                val = self._model(x).item()
                preds.append(val)
                seq.append(val)     # auto-regressive: feed prediction back

        return np.array(preds) * self._std + self._mean

    @property
    def name(self) -> str:
        return "TCN"
