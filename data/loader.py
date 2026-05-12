"""
Data loading and feature engineering for the stock forecast benchmark.

Download flow:
  yfinance → raw OHLCV DataFrame → add lag + rolling features → train/test split

The full date range (train + test) is downloaded before splitting so that
lag features for the first test rows correctly reference the training period.
"""

import logging
import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _download(ticker: str, start: str, end: str, cache_dir: str) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker and cache the result as a pickle.

    On subsequent calls the cached file is loaded instead of hitting the
    network, which saves time during repeated runs.
    """
    os.makedirs(cache_dir, exist_ok=True)
    # Replace ^ so file names are valid on all operating systems
    cache_path = os.path.join(cache_dir, f"{ticker.replace('^', '')}_{start}_{end}.pkl")

    if os.path.exists(cache_path):
        logger.info("Loading %s from cache (%s)", ticker, cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info("Downloading %s from yfinance [%s → %s]", ticker, start, end)
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


def _add_features(df: pd.DataFrame, lags: int, rolling_windows: list[int]) -> pd.DataFrame:
    """
    Append lag and rolling-window features to the DataFrame.

    WHY shift(1) on rolling features?
      Without shift, rolling_mean_7[i] would include Close[i] itself,
      leaking the target value into the input features (look-ahead bias).
      Shifting by 1 ensures every feature at row i only contains information
      from rows before i.

    After all NaN rows (created by shifting / rolling) are dropped, the
    effective start date moves forward by roughly max(lags, max_window) days.
    """
    df = df.copy()
    target = df["Close"]

    # lag_k[i] = Close[i - k]  →  'what was the price k days ago?'
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = target.shift(lag)

    # Rolling statistics: .shift(1) avoids look-ahead bias (see docstring above)
    for w in rolling_windows:
        df[f"rolling_mean_{w}"] = target.rolling(w).mean().shift(1)
        df[f"rolling_std_{w}"]  = target.rolling(w).std().shift(1)

    # Remove rows where any feature is still NaN (the first ~30 rows)
    return df.dropna()


def load_ticker(
    ticker: str, cfg: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (train_df, test_df) for a single ticker.

    Both DataFrames contain the 'Close' column plus all engineered features.
    The ML models use the feature columns; statistical and DL models read
    only 'Close' and ignore the rest.
    """
    data_cfg = cfg["data"]
    feat_cfg  = cfg["features"]
    cache_dir = os.path.join("data", "cache")

    # Download the full date range so features are consistent across the split
    raw = _download(ticker, data_cfg["train_start"], data_cfg["test_end"], cache_dir)

    # Keep only the target column before adding derived features
    df  = raw[[data_cfg["target_col"]]].copy()
    df  = _add_features(df, feat_cfg["lags"], feat_cfg["rolling_windows"])

    # Slice by date after feature engineering (pandas loc is end-inclusive)
    train = df.loc[data_cfg["train_start"] : data_cfg["train_end"]]
    test  = df.loc[data_cfg["test_start"]  : data_cfg["test_end"]]

    logger.info(
        "%s — train: %d rows (%s → %s), test: %d rows (%s → %s)",
        ticker,
        len(train), train.index[0].date(), train.index[-1].date(),
        len(test),  test.index[0].date(),  test.index[-1].date(),
    )
    return train, test
