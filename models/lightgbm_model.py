"""
LightGBM forecaster with recursive multi-step prediction.

Identical interface and feature set as XGBoost — exists as a separate model
so the leaderboard can compare the two gradient-boosting libraries head-to-head.
LightGBM uses a histogram-based algorithm that is typically faster to train
than XGBoost on large datasets.
"""

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from .base import BaseForecaster

logger = logging.getLogger(__name__)

_FEATURE_PREFIX = ("lag_", "rolling_")


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all column names that are lag or rolling features."""
    return [c for c in df.columns if c.startswith(_FEATURE_PREFIX)]


class LightGBMForecaster(BaseForecaster):

    def __init__(self, config: dict, feat_config: dict) -> None:
        self._config    = config
        self._lags      = feat_config["lags"]
        self._roll_wins = feat_config["rolling_windows"]
        self._model     = None
        self._history: list[float] = []

    def fit(self, train: pd.DataFrame) -> None:
        cols = _feature_cols(train)
        X = train[cols].values
        y = train["Close"].values.astype(float)

        self._model = lgb.LGBMRegressor(
            n_estimators=self._config["n_estimators"],
            learning_rate=self._config["learning_rate"],
            max_depth=self._config["max_depth"],
            random_state=42,
            verbosity=-1,   # suppress LightGBM console output
        )
        self._model.fit(X, y)

        lookback = max(self._lags, max(self._roll_wins))
        self._history = list(train["Close"].values[-lookback:].astype(float))
        logger.info("%s fitted successfully", self.name)

    def predict(self, n_steps: int) -> np.ndarray:
        history = list(self._history)
        preds: list[float] = []

        for _ in range(n_steps):
            feat = self._build_features(history)
            val  = float(self._model.predict(np.array([feat]))[0])
            preds.append(val)
            history.append(val)   # use prediction as the next lag

        return np.array(preds)

    def _build_features(self, history: list[float]) -> list[float]:
        """
        Rebuild the feature vector from history in the same order as
        data/loader.py so the model sees the same input schema it was trained on.
        """
        feats: list[float] = []
        for lag in range(1, self._lags + 1):
            feats.append(history[-lag])
        for w in self._roll_wins:
            window = np.array(history[-w:])
            feats.append(float(window.mean()))
            feats.append(float(window.std()) if len(window) > 1 else 0.0)
        return feats

    @property
    def name(self) -> str:
        return "LightGBM"
