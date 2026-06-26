"""
Data loading and feature engineering for the stock forecast benchmark.

Download flow:
  yfinance -> raw OHLCV DataFrame -> add lag + rolling features -> train/test split

The full date range (train + test) is downloaded before splitting so that
lag features for the first test rows correctly reference the training period.
The feature-engineering and split steps are exposed as standalone, network-free
functions (:func:`add_features`, :func:`split_train_test`) so the leakage-safe
behaviour can be unit-tested on synthetic data.
"""

import logging
import os
import pickle
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _download(ticker: str, start: str, end: str, cache_dir: str) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker and cache the result as a pickle.

    On subsequent calls the cached file is loaded instead of hitting the
    network, which saves time during repeated runs.

    Args:
        ticker: yfinance ticker symbol (e.g. ``"^GSPC"``).
        start: Inclusive start date, ``YYYY-MM-DD``.
        end: Inclusive end date, ``YYYY-MM-DD``.
        cache_dir: Directory used to store/lookup the pickled download.

    Returns:
        Raw OHLCV DataFrame indexed by a timezone-naive ``DatetimeIndex``.
    """
    os.makedirs(cache_dir, exist_ok=True)
    # Replace ^ so file names are valid on all operating systems
    cache_path = os.path.join(cache_dir, f"{ticker.replace('^', '')}_{start}_{end}.pkl")

    if os.path.exists(cache_path):
        logger.info("Loading %s from cache (%s)", ticker, cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info("Downloading %s from yfinance [%s -> %s]", ticker, start, end)
    # Imported lazily so the network-free helpers (add_features, split_train_test,
    # make_synthetic_prices) can be used and tested without yfinance installed.
    import yfinance as yf

    # auto_adjust=True applies splits and dividend adjustments automatically
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    # Newer versions of yfinance return a MultiIndex when multiple tickers
    # are downloaded at once.  Flatten it for a single-ticker download.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)

    # Strip timezone info so comparisons with plain date strings work cleanly
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    with open(cache_path, "wb") as f:
        pickle.dump(df, f)

    return df


def add_features(
    df: pd.DataFrame, lags: int, rolling_windows: Sequence[int]
) -> pd.DataFrame:
    """
    Append lag and rolling-window features to a Close-price DataFrame.

    Why ``shift(1)`` on rolling features?
        Without the shift, ``rolling_mean_7[i]`` would include ``Close[i]``
        itself, leaking the target value into the input features (look-ahead
        bias).  Shifting by 1 guarantees that every feature at row ``i`` only
        contains information strictly before row ``i``.

    After all NaN rows (created by shifting / rolling) are dropped, the
    effective start date moves forward by roughly ``max(lags, max_window)``
    rows.

    Args:
        df: DataFrame containing at least a ``Close`` column.
        lags: Number of lag features to create (``lag_1 … lag_lags``).
        rolling_windows: Window sizes for rolling mean/std features.

    Returns:
        A copy of ``df`` with the engineered feature columns appended and all
        rows containing NaN features removed.
    """
    df = df.copy()
    target = df["Close"]

    # lag_k[i] = Close[i - k]  ->  'what was the price k days ago?'
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = target.shift(lag)

    # Rolling statistics: .shift(1) avoids look-ahead bias (see docstring above)
    for w in rolling_windows:
        df[f"rolling_mean_{w}"] = target.rolling(w).mean().shift(1)
        df[f"rolling_std_{w}"]  = target.rolling(w).std().shift(1)

    # Remove rows where any feature is still NaN (the first ~max(lags, window) rows)
    return df.dropna()


def split_train_test(
    df: pd.DataFrame, data_cfg: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Slice a feature DataFrame into train and test windows by calendar date.

    The split is performed *after* feature engineering on the full series so
    that the first test rows' lag/rolling features reference real training
    history rather than NaNs — a subtle but important leakage-safety property.

    Args:
        df: Feature DataFrame indexed by a ``DatetimeIndex``.
        data_cfg: The ``data`` section of the config, providing
            ``train_start``, ``train_end``, ``test_start`` and ``test_end``.

    Returns:
        ``(train_df, test_df)``. Note that ``pandas`` ``.loc`` date slicing is
        end-inclusive.
    """
    train = df.loc[data_cfg["train_start"] : data_cfg["train_end"]]
    test  = df.loc[data_cfg["test_start"]  : data_cfg["test_end"]]
    return train, test


def load_ticker(ticker: str, cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return ``(train_df, test_df)`` for a single ticker.

    Both DataFrames contain the ``Close`` column plus all engineered features.
    The ML models use the feature columns; statistical and DL models read only
    ``Close`` and ignore the rest.

    Args:
        ticker: yfinance ticker symbol.
        cfg: The full parsed configuration dict.

    Returns:
        ``(train_df, test_df)`` sliced to the configured train/test windows.
    """
    data_cfg  = cfg["data"]
    feat_cfg  = cfg["features"]
    cache_dir = os.path.join("data", "cache")

    # Download the full date range so features are consistent across the split
    raw = _download(ticker, data_cfg["train_start"], data_cfg["test_end"], cache_dir)

    # Keep only the target column before adding derived features
    df = raw[[data_cfg["target_col"]]].copy()
    df = add_features(df, feat_cfg["lags"], feat_cfg["rolling_windows"])

    train, test = split_train_test(df, data_cfg)

    logger.info(
        "%s - train: %d rows (%s -> %s), test: %d rows (%s -> %s)",
        ticker,
        len(train), train.index[0].date(), train.index[-1].date(),
        len(test),  test.index[0].date(),  test.index[-1].date(),
    )
    return train, test


def make_synthetic_prices(
    n_days: int = 900,
    start: str = "1990-01-01",
    seed: int = 42,
    drift: float = 0.0003,
    volatility: float = 0.012,
    initial_price: float = 100.0,
) -> pd.DataFrame:
    """
    Generate a deterministic synthetic Close-price series for tests/smoke runs.

    The series is a geometric random walk with mild upward drift on a
    business-day calendar — structurally similar to a real equity price track
    but fully offline and reproducible. It lets the test suite and the
    ``--smoke`` quick run exercise the full load -> feature -> split -> fit ->
    evaluate path without contacting yfinance.

    Args:
        n_days: Number of business days to generate.
        start: First date of the series, ``YYYY-MM-DD``.
        seed: RNG seed controlling the (otherwise random) returns.
        drift: Mean daily log-return.
        volatility: Standard deviation of the daily log-returns.
        initial_price: Price on the first day.

    Returns:
        A single-column (``Close``) DataFrame indexed by a business-day
        ``DatetimeIndex``.
    """
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=drift, scale=volatility, size=n_days)
    prices = initial_price * np.exp(np.cumsum(log_returns))
    index = pd.bdate_range(start=start, periods=n_days)
    return pd.DataFrame({"Close": prices}, index=index)


def _feature_names(lags: int, rolling_windows: Sequence[int]) -> List[str]:
    """Return the canonical feature-column order for the given config."""
    names = [f"lag_{lag}" for lag in range(1, lags + 1)]
    for w in rolling_windows:
        names.append(f"rolling_mean_{w}")
        names.append(f"rolling_std_{w}")
    return names
