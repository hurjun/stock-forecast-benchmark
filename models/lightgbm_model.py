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
from .features import build_recursive_features, feature_columns

logger = logging.getLogger(__name__)


class LightGBMForecaster(BaseForecaster):

    def __init__(self, config: dict, feat_config: dict) -> None:
        self._config    = config
        self._lags      = feat_config["lags"]
        self._roll_wins = feat_config["rolling_windows"]
        self._model     = None
        self._history: list[float] = []

    def fit(self, train: pd.DataFrame) -> None:
        cols = feature_columns(train)
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
            feat = build_recursive_features(history, self._lags, self._roll_wins)
            val  = float(self._model.predict(np.array([feat]))[0])
            preds.append(val)
            history.append(val)   # use prediction as the next lag

        return np.array(preds)

    @property
    def name(self) -> str:
        return "LightGBM"
