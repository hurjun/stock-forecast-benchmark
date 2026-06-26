"""
Tests for the leakage-safe feature engineering and train/test split.

These guard the most subtle correctness property of the benchmark: no feature
at row ``i`` may contain information from row ``i`` or later (look-ahead bias),
and the date-based split must be performed on the fully-featurised series so
the first test rows reference real training history.
"""

import numpy as np
import pandas as pd
import pytest

from data.loader import (
    _feature_names,
    add_features,
    make_synthetic_prices,
    split_train_test,
)
from models.features import build_recursive_features, feature_columns

LAGS = 3
WINDOWS = [2, 4]


@pytest.fixture
def linear_frame():
    """A simple, fully-predictable Close series (1, 2, 3, ...) on a bday index."""
    close = np.arange(1, 51, dtype=float)
    idx = pd.bdate_range("2000-01-03", periods=len(close))
    return pd.DataFrame({"Close": close}, index=idx), close, idx


def test_add_features_has_no_lookahead_leakage(linear_frame):
    df, close, idx = linear_frame
    feat = add_features(df, lags=LAGS, rolling_windows=WINDOWS)

    # No NaNs should remain after the warm-up rows are dropped.
    assert not feat.isna().any().any()

    for row_date, row in feat.iterrows():
        pos = idx.get_loc(row_date)

        # Lag features look strictly into the past.
        for lag in range(1, LAGS + 1):
            assert row[f"lag_{lag}"] == pytest.approx(close[pos - lag])

        # Rolling stats use the w values BEFORE the current row (shift(1)),
        # so they never include close[pos] itself.
        for w in WINDOWS:
            past = close[pos - w : pos]
            assert row[f"rolling_mean_{w}"] == pytest.approx(past.mean())
            assert row[f"rolling_std_{w}"] == pytest.approx(past.std(ddof=1))

        # Explicit anti-leakage check: nothing equals the current target by
        # construction of this strictly increasing series.
        assert (row.drop(labels=["Close"]) < close[pos]).all()


def test_add_features_drops_warmup_rows(linear_frame):
    df, close, _ = linear_frame
    feat = add_features(df, lags=LAGS, rolling_windows=WINDOWS)
    # First valid row needs max(lags, max_window) past observations available.
    expected_dropped = max(LAGS, max(WINDOWS))
    assert len(feat) == len(close) - expected_dropped


def test_feature_column_order_is_canonical(linear_frame):
    df, _, _ = linear_frame
    feat = add_features(df, lags=LAGS, rolling_windows=WINDOWS)
    assert feature_columns(feat) == _feature_names(LAGS, WINDOWS)


def test_recursive_reconstruction_matches_training_schema(linear_frame):
    """The prediction-time feature vector must line up with the trained columns."""
    df, close, _ = linear_frame
    feat = add_features(df, lags=LAGS, rolling_windows=WINDOWS)

    history = list(close)
    vector = build_recursive_features(history, LAGS, WINDOWS)

    # Same length and positional meaning as the trained feature columns.
    assert len(vector) == len(feature_columns(feat))

    # First `LAGS` entries are the most recent values, most-recent-first.
    for lag in range(1, LAGS + 1):
        assert vector[lag - 1] == pytest.approx(history[-lag])


def test_split_is_leakage_safe_across_the_boundary():
    """First test row's lag_1 must equal the last train row's Close."""
    df = make_synthetic_prices(n_days=1100, start="1990-01-01", seed=7)
    feat = add_features(df, lags=LAGS, rolling_windows=WINDOWS)

    data_cfg = {
        "train_start": "1990-01-01", "train_end": "1992-12-31",
        "test_start":  "1993-01-01", "test_end":  "1994-12-31",
    }
    train, test = split_train_test(feat, data_cfg)

    assert len(train) > 0 and len(test) > 0
    # Windows do not overlap.
    assert train.index.max() < pd.Timestamp("1993-01-01")
    assert test.index.min() >= pd.Timestamp("1993-01-01")
    # Continuity: the first test row references the final training Close,
    # which is only possible because features were built before splitting.
    assert test.iloc[0]["lag_1"] == pytest.approx(train.iloc[-1]["Close"])
