"""
End-to-end pipeline test on a small synthetic series.

Exercises the full path - synthetic data -> feature engineering -> date split
-> fit -> recursive predict -> metric computation - for the fast, network-free
baselines (Naive and ARIMA).  This is the smoke guarantee that the harness
itself is wired together correctly, independent of the heavy DL stack.
"""

import math

import numpy as np
import pytest

from data.loader import add_features, make_synthetic_prices, split_train_test
from evaluation.metrics import compute_all
from models.arima import ARIMAForecaster
from models.naive import NaiveForecaster

SMOKE_CFG = {
    "train_start": "1990-01-01", "train_end": "1992-12-31",
    "test_start":  "1993-01-01", "test_end":  "1994-12-31",
}


@pytest.fixture
def synthetic_split():
    df = make_synthetic_prices(n_days=1200, start="1990-01-01", seed=123)
    feat = add_features(df, lags=5, rolling_windows=[7, 14])
    train, test = split_train_test(feat, SMOKE_CFG)
    return train, test


def test_naive_forecaster_predicts_last_value(synthetic_split):
    train, test = synthetic_split
    model = NaiveForecaster()
    model.fit(train)
    preds = model.predict(len(test))

    assert preds.shape == (len(test),)
    last_train_close = float(train["Close"].values[-1])
    assert np.allclose(preds, last_train_close)


def test_naive_end_to_end_metrics_are_finite(synthetic_split):
    train, test = synthetic_split
    model = NaiveForecaster()
    model.fit(train)
    preds = model.predict(len(test))

    metrics = compute_all(test["Close"].values.astype(float), preds)
    assert set(metrics) == {"MAE", "RMSE", "MAPE", "DA"}
    assert all(math.isfinite(v) for v in metrics.values())
    assert metrics["MAE"] >= 0.0


def test_arima_end_to_end_runs_and_returns_correct_length(synthetic_split):
    train, test = synthetic_split
    model = ARIMAForecaster({"order": [1, 1, 0]})
    model.fit(train)
    preds = model.predict(len(test))

    assert preds.shape == (len(test),)
    assert np.all(np.isfinite(preds))

    metrics = compute_all(test["Close"].values.astype(float), preds)
    assert all(math.isfinite(v) for v in metrics.values())


def test_naive_baseline_is_a_sensible_reference_floor(synthetic_split):
    """On a (nearly) random walk, persistence should not be wildly off-scale."""
    train, test = synthetic_split
    model = NaiveForecaster()
    model.fit(train)
    preds = model.predict(len(test))
    actual = test["Close"].values.astype(float)

    # MAPE should be a finite percentage, not exploding to absurd values.
    metrics = compute_all(actual, preds)
    assert 0.0 <= metrics["MAPE"] < 500.0
