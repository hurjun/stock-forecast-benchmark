"""
Naive (random-walk / persistence) forecaster.

The naive baseline predicts that every future price equals the last observed
training price.  Under the efficient-market / random-walk hypothesis this is
the theoretically optimal point forecast for a price series, which makes it the
standard reference floor in financial forecasting: any model that cannot beat
persistence is not adding value.

It is also fast (no fitting), fully deterministic, and dependency-free, so it
doubles as the model used by the test suite and the ``--smoke`` quick run.
"""

import logging

import numpy as np
import pandas as pd

from .base import BaseForecaster

logger = logging.getLogger(__name__)


class NaiveForecaster(BaseForecaster):
    """Persistence model: forecast the last observed Close for every horizon."""

    def __init__(self, config: dict | None = None) -> None:
        # Accepts an (unused) config dict for a uniform constructor signature.
        self._config = config or {}
        self._last_value: float = 0.0

    def fit(self, train: pd.DataFrame) -> None:
        """Record the final training Close price; there is nothing to optimise."""
        self._last_value = float(train["Close"].values[-1])
        logger.info("%s fitted (last value=%.4f)", self.name, self._last_value)

    def predict(self, n_steps: int) -> np.ndarray:
        """Return a constant array of the last observed price, length ``n_steps``."""
        return np.full(n_steps, self._last_value, dtype=float)

    @property
    def name(self) -> str:
        return "Naive"
