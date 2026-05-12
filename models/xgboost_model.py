"""
XGBoost forecaster with recursive multi-step prediction.

Training:
  - Features: lag_1…lag_30, rolling_mean/std for windows [7, 14, 30]
  - Target:   next-day Close price

Prediction strategy (recursive / auto-regressive):
  1. Predict one step ahead using the last known values.
  2. Append the prediction to history.
  3. Repeat from step 1 — each new prediction becomes the next lag.

  Recursive prediction accumulates error over long horizons, but it is
  the standard approach when a single model is used for all horizons.
"""

import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from .base import BaseForecaster

logger = logging.getLogger(__name__)

_FEATURE_PREFIX = ("lag_", "rolling_")


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all column names that are lag or rolling features."""
    return [c for c in df.columns if c.startswith(_FEATURE_PREFIX)]


class XGBoostForecaster(BaseForecaster):

    def __init__(self, config: dict, feat_config: dict) -> None:
        self._config      = config
        self._lags        = feat_config["lags"]
        self._roll_wins   = feat_config["rolling_windows"]
        self._model       = None
        self._history: list[float] = []  # tail of training series used for prediction

    def fit(self, train: pd.DataFrame) -> None:
        cols = _feature_cols(train)
        X = train[cols].values          # shape: (n_train, n_features)
        y = train["Close"].values.astype(float)

        self._model = xgb.XGBRegressor(
            n_estimators=self._config["n_estimators"],
            learning_rate=self._config["learning_rate"],
            max_depth=self._config["max_depth"],
            random_state=42,
            verbosity=0,                # suppress XGBoost console output
        )
        self._model.fit(X, y)

        # Store only as many historical values as needed to reconstruct features:
        # we need max(lags, max_rolling_window) past observations
        lookback = max(self._lags, max(self._roll_wins))
        self._history = list(train["Close"].values[-lookback:].astype(float))
        logger.info("%s fitted successfully", self.name)

    def predict(self, n_steps: int) -> np.ndarray:
        history = list(self._history)   # work on a copy to leave self._history intact
        preds: list[float] = []

        for _ in range(n_steps):
            feat = self._build_features(history)
            val  = float(self._model.predict(np.array([feat]))[0])
            preds.append(val)
            history.append(val)         # feed prediction back as the next lag input

        return np.array(preds)

    def _build_features(self, history: list[float]) -> list[float]:
        """
        Reconstruct the feature vector from the running history list.

        The order must exactly match the column order produced by
        data/loader.py::_add_features():
          lag_1, lag_2, ..., lag_lags,
          rolling_mean_w, rolling_std_w  (for each w in rolling_windows)
        """
        feats: list[float] = []

        # Lag features: history[-1] is the most recent known value (lag_1)
        for lag in range(1, self._lags + 1):
            feats.append(history[-lag])

        # Rolling statistics from the tail of history
        for w in self._roll_wins:
            window = np.array(history[-w:])
            feats.append(float(window.mean()))
            feats.append(float(window.std()) if len(window) > 1 else 0.0)

        return feats

    @property
    def name(self) -> str:
        return "XGBoost"
