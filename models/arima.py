"""
ARIMA forecaster using statsmodels.

ARIMA(p, d, q):
  p = number of autoregressive lags  (how many past values to use)
  d = degree of differencing          (d=1 removes a linear trend / unit root)
  q = number of moving-average terms  (set to 0 here — pure AR after differencing)

Default config: ARIMA(5, 1, 0)
"""

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .base import BaseForecaster

logger = logging.getLogger(__name__)


class ARIMAForecaster(BaseForecaster):

    def __init__(self, config: dict) -> None:
        self._config = config  # expects {'order': [p, d, q]}
        self._result = None

    def fit(self, train: pd.DataFrame) -> None:
        # ARIMA only needs the univariate Close price series
        series = train["Close"].astype(float)
        model = ARIMA(series, order=tuple(self._config["order"]))
        self._result = model.fit()
        logger.info("%s fitted (AIC=%.2f)", self.name, self._result.aic)

    def predict(self, n_steps: int) -> np.ndarray:
        # forecast() extends from the end of the fitted training series
        return self._result.forecast(steps=n_steps).to_numpy()

    @property
    def name(self) -> str:
        return "ARIMA"
