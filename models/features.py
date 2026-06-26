"""
Shared feature engineering for the recursive tree-based forecasters.

Both :class:`~models.xgboost_model.XGBoostForecaster` and
:class:`~models.lightgbm_model.LightGBMForecaster` consume the lag/rolling
feature schema produced by :func:`data.loader.add_features`.  The helpers below
live in one place so that the training-time column selection and the
prediction-time feature reconstruction can never drift apart (and so that the
reconstruction order can be unit-tested without importing the heavy gradient
boosting libraries).
"""

from typing import List, Sequence

import numpy as np
import pandas as pd

#: Column-name prefixes that identify engineered model features.
FEATURE_PREFIX = ("lag_", "rolling_")


def feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the names of every lag/rolling feature column in ``df``.

    Args:
        df: A DataFrame produced by :func:`data.loader.add_features`.

    Returns:
        Feature column names in their original left-to-right order, i.e.
        ``lag_1 … lag_n`` followed by ``rolling_mean_w``/``rolling_std_w``
        for each rolling window ``w``.
    """
    return [c for c in df.columns if c.startswith(FEATURE_PREFIX)]


def build_recursive_features(
    history: Sequence[float],
    lags: int,
    rolling_windows: Sequence[int],
) -> List[float]:
    """
    Reconstruct a single feature vector from a running price history.

    The element order **must** match the column order created by
    :func:`data.loader.add_features` so the model sees the same input schema
    at prediction time that it was trained on::

        lag_1, lag_2, …, lag_lags,
        rolling_mean_w, rolling_std_w   (for each w in rolling_windows)

    Args:
        history: Running list of past (and previously predicted) Close values;
            ``history[-1]`` is the most recent value (``lag_1``).
        lags: Number of lag features to emit.
        rolling_windows: Rolling-window sizes, in the same order used to build
            the training features.

    Returns:
        A flat list of ``lags + 2 * len(rolling_windows)`` feature values.
    """
    feats: List[float] = []

    # Lag features: history[-1] is the most recent known value (lag_1).
    for lag in range(1, lags + 1):
        feats.append(float(history[-lag]))

    # Rolling statistics from the tail of history.
    for w in rolling_windows:
        window = np.asarray(history[-w:], dtype=float)
        feats.append(float(window.mean()))
        feats.append(float(window.std()) if len(window) > 1 else 0.0)

    return feats
