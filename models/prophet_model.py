"""Prophet forecaster (Meta's decomposable time-series model)."""

import logging

import numpy as np
import pandas as pd
from prophet import Prophet

from .base import BaseForecaster

logger = logging.getLogger(__name__)


class ProphetForecaster(BaseForecaster):

    def __init__(self, config: dict) -> None:
        self._config = config
        self._model = None

    def fit(self, train: pd.DataFrame) -> None:
        df = (
            train[["Close"]]
            .reset_index()
            .rename(columns={train.index.name or "Date": "ds", "Close": "y"})
        )
        df["ds"] = pd.to_datetime(df["ds"])

        self._model = Prophet(
            weekly_seasonality=self._config["weekly_seasonality"],
            yearly_seasonality=self._config["yearly_seasonality"],
        )
        # Suppress verbose Stan output
        import logging as _logging
        _logging.getLogger("prophet").setLevel(_logging.WARNING)
        _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)

        self._model.fit(df)
        logger.info("%s fitted successfully", self.name)

    def predict(self, n_steps: int) -> np.ndarray:
        future   = self._model.make_future_dataframe(periods=n_steps, freq="B")
        forecast = self._model.predict(future)
        return forecast["yhat"].values[-n_steps:]

    @property
    def name(self) -> str:
        return "Prophet"
