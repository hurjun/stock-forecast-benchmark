"""
ETS (Exponential Smoothing) forecaster using statsmodels Holt-Winters.

ETS stands for Error, Trend, Seasonality. This configuration uses:
  - Additive trend:    captures a linear upward or downward drift
  - Damped trend:      dampens the trend as the horizon grows to avoid
                       explosive extrapolation (a known failure mode of
                       undamped Holt's linear method on stock data)
  - No seasonality:   daily stock prices do not follow a clean periodic
                       seasonal pattern, so adding a seasonal component
                       would overfit noise

ETS is one of the fastest and most reliable univariate baselines in
practice — it is the default benchmark in many forecasting competitions.
"""

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .base import BaseForecaster

logger = logging.getLogger(__name__)


class ETSForecaster(BaseForecaster):

    def __init__(self, config: dict) -> None:
        # trend:        'add' = additive (linear) trend component
        # damped_trend: True  = attenuate the trend projection at long horizons
        self._trend        = config.get("trend", "add")
        self._damped_trend = config.get("damped_trend", True)
        self._result       = None

    def fit(self, train: pd.DataFrame) -> None:
        series = train["Close"].astype(float)

        model = ExponentialSmoothing(
            series,
            trend=self._trend,
            damped_trend=self._damped_trend,
            seasonal=None,           # no seasonal component for daily stock data
            initialization_method="estimated",  # let statsmodels optimise initial states
        )
        self._result = model.fit(optimized=True)
        logger.info(
            "%s fitted (AIC=%.2f, alpha=%.4f, beta=%.4f)",
            self.name,
            self._result.aic,
            self._result.params["smoothing_level"],
            self._result.params["smoothing_trend"],
        )

    def predict(self, n_steps: int) -> np.ndarray:
        # forecast() produces h-step-ahead predictions from the end of training
        return self._result.forecast(steps=n_steps).to_numpy()

    @property
    def name(self) -> str:
        return "ETS"
