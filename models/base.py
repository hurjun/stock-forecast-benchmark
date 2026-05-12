"""Abstract base class that every forecaster must implement."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseForecaster(ABC):

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> None:
        """Train the model on the given DataFrame (must contain a 'Close' column)."""

    @abstractmethod
    def predict(self, n_steps: int) -> np.ndarray:
        """Return a 1-D array of exactly n_steps future price predictions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name used in the leaderboard."""
