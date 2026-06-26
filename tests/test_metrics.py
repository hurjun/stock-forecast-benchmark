"""Known-answer tests for the forecast evaluation metrics."""

import math

import numpy as np
import pytest

from evaluation.metrics import (
    compute_all,
    directional_accuracy,
    mae,
    mape,
    rmse,
)


def test_perfect_prediction_has_zero_error():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mae(y, y) == pytest.approx(0.0)
    assert rmse(y, y) == pytest.approx(0.0)
    assert mape(y, y) == pytest.approx(0.0)


def test_mae_known_answer():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 5.0])  # abs errors: 1, 0, 2 -> mean = 1.0
    assert mae(y_true, y_pred) == pytest.approx(1.0)


def test_rmse_known_answer():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([3.0, 0.0, 4.0])  # squared errors: 9, 0, 16 -> mean 25/3
    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt(25.0 / 3.0))


def test_rmse_penalises_large_errors_more_than_mae():
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([0.0, 0.0, 0.0, 10.0])  # one big miss
    assert rmse(y_true, y_pred) > mae(y_true, y_pred)


def test_mape_known_answer_and_percentage_scale():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])  # 10% and 10% -> 10
    assert mape(y_true, y_pred) == pytest.approx(10.0)


def test_mape_skips_zero_denominators():
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([5.0, 90.0])  # row 0 skipped; only 10% from row 1
    assert mape(y_true, y_pred) == pytest.approx(10.0)


def test_directional_accuracy_all_correct():
    y_true = np.array([1.0, 2.0, 3.0, 2.0])  # up, up, down
    y_pred = np.array([5.0, 6.0, 9.0, 1.0])  # up, up, down
    assert directional_accuracy(y_true, y_pred) == pytest.approx(100.0)


def test_directional_accuracy_all_wrong():
    y_true = np.array([1.0, 2.0, 3.0])  # up, up
    y_pred = np.array([3.0, 2.0, 1.0])  # down, down
    assert directional_accuracy(y_true, y_pred) == pytest.approx(0.0)


def test_directional_accuracy_is_a_percentage():
    y_true = np.array([1.0, 2.0, 1.0, 2.0])  # up, down, up
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])  # up, up, up  -> 2 of 3 correct
    assert directional_accuracy(y_true, y_pred) == pytest.approx(200.0 / 3.0)


def test_compute_all_returns_all_metric_keys():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2])
    out = compute_all(y_true, y_pred)
    assert set(out) == {"MAE", "RMSE", "MAPE", "DA"}
    assert all(isinstance(v, float) and math.isfinite(v) for v in out.values())


def test_compute_all_matches_individual_functions():
    """The aggregator must return exactly what each metric function computes."""
    y_true = np.array([10.0, 11.0, 9.0, 12.0, 13.0])
    y_pred = np.array([10.5, 10.0, 9.5, 11.0, 14.0])
    out = compute_all(y_true, y_pred)
    assert out["MAE"] == pytest.approx(mae(y_true, y_pred))
    assert out["RMSE"] == pytest.approx(rmse(y_true, y_pred))
    assert out["MAPE"] == pytest.approx(mape(y_true, y_pred))
    assert out["DA"] == pytest.approx(directional_accuracy(y_true, y_pred))


def test_error_metrics_are_symmetric_in_their_arguments():
    """MAE and RMSE depend only on the residual magnitude, so swapping the
    two arrays must leave them unchanged."""
    a = np.array([1.0, 5.0, 2.0, 8.0])
    b = np.array([2.0, 4.0, 2.5, 6.0])
    assert mae(a, b) == pytest.approx(mae(b, a))
    assert rmse(a, b) == pytest.approx(rmse(b, a))


def test_rmse_is_always_at_least_mae():
    """By Jensen's inequality RMSE >= MAE for any residual vector."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        y_true = rng.normal(size=50)
        y_pred = rng.normal(size=50)
        assert rmse(y_true, y_pred) >= mae(y_true, y_pred) - 1e-12


def test_mape_is_scale_invariant():
    """Scaling true and predicted prices by the same factor leaves the
    percentage error unchanged — the property that makes MAPE comparable
    across tickers of very different price levels."""
    y_true = np.array([100.0, 200.0, 150.0])
    y_pred = np.array([110.0, 180.0, 165.0])
    assert mape(y_true * 7.5, y_pred * 7.5) == pytest.approx(mape(y_true, y_pred))


def test_flat_forecast_has_zero_directional_accuracy_on_a_trend():
    """A constant (persistence) forecast never signals up/down, so against a
    strictly rising series its directional accuracy is 0% — the caveat behind
    the flat smoke-run baselines and the ~0% tree-model DA in the README."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # strictly increasing
    y_pred = np.full_like(y_true, 3.0)            # flat forecast
    assert directional_accuracy(y_true, y_pred) == pytest.approx(0.0)
