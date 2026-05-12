"""
Entry point for the Stock Forecast Benchmark.

Usage:
    python main.py

What this script does:
  1. Load configuration from config.yaml
  2. Download and preprocess stock data for each ticker (cached locally)
  3. Train all 7 forecasting models on the training period (1962–1990)
  4. Generate predictions for the test period (1991–2000)
  5. Compute MAE, RMSE, MAPE, and Directional Accuracy for every model/ticker pair
  6. Save a leaderboard (CSV + Markdown) and inject it into README.md
  7. Save forecast comparison and metrics bar charts to results/
"""

import logging
import warnings

import numpy as np
import yaml

# Suppress noisy third-party warnings (statsmodels, prophet, etc.)
warnings.filterwarnings("ignore")

from data.loader import load_ticker
from evaluation.leaderboard import build_leaderboard, inject_into_readme, save_leaderboard
from evaluation.metrics import compute_all
from models.arima import ARIMAForecaster
from models.gru import GRUForecaster
from models.lightgbm_model import LightGBMForecaster
from models.lstm import LSTMForecaster
from models.prophet_model import ProphetForecaster
from models.transformer import TransformerForecaster
from models.xgboost_model import XGBoostForecaster
from visualization.plots import plot_forecast_comparison, plot_metrics_bar

# Configure root logger: INFO level shows training progress without debug noise
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    """Parse config.yaml into a plain Python dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_models(cfg: dict) -> list:
    """
    Instantiate all forecasters from the loaded config.

    Each model receives only the slice of config it needs so that
    model files never import from config.yaml directly.
    """
    feat = cfg["features"]
    m    = cfg["models"]
    return [
        ARIMAForecaster(m["arima"]),
        ProphetForecaster(m["prophet"]),
        XGBoostForecaster(m["xgboost"], feat),
        LightGBMForecaster(m["lightgbm"], feat),
        LSTMForecaster(m["lstm"]),
        GRUForecaster(m["gru"]),
        TransformerForecaster(m["transformer"]),
    ]


def _align(arr: np.ndarray, n: int) -> np.ndarray:
    """
    Make sure predict() output is exactly n elements long.

    Some models (e.g. Prophet with business-day calendars) may return
    slightly more or fewer steps than requested.
    """
    if len(arr) > n:
        return arr[:n]
    if len(arr) < n:
        # Pad by repeating the last value rather than introducing zeros
        return np.pad(arr, (0, n - len(arr)), mode="edge")
    return arr


def run_pipeline(cfg: dict):
    """
    Main training + evaluation loop across all tickers and models.

    Returns:
        results      — list of per-(model, ticker) metric dicts
        all_forecasts — {ticker: {model_name: predictions}}
        all_actuals   — {ticker: actual_close_array}
        all_dates     — {ticker: DatetimeIndex of test period}
    """
    tickers = cfg["data"]["tickers"]

    results: list[dict]             = []
    all_forecasts: dict[str, dict]  = {}
    all_actuals:   dict[str, np.ndarray] = {}
    all_dates:     dict[str, object]     = {}

    for ticker in tickers:
        logger.info("=" * 60)
        logger.info("Ticker: %s", ticker)
        logger.info("=" * 60)

        # Load pre-processed train and test DataFrames
        train_df, test_df = load_ticker(ticker, cfg)
        actual  = test_df["Close"].values.astype(float)
        n_steps = len(test_df)

        all_actuals[ticker]   = actual
        all_dates[ticker]     = test_df.index
        all_forecasts[ticker] = {}

        # Train each model independently on the same training set
        for model in build_models(cfg):
            try:
                logger.info("  [%s] Training...", model.name)
                model.fit(train_df)

                preds = _align(model.predict(n_steps), n_steps)
                metrics = compute_all(actual, preds)

                results.append({"model": model.name, "ticker": ticker, **metrics})
                all_forecasts[ticker][model.name] = preds

                logger.info(
                    "  [%s] RMSE=%.4f  MAE=%.4f  MAPE=%.2f%%  DA=%.1f%%",
                    model.name, metrics["RMSE"], metrics["MAE"],
                    metrics["MAPE"], metrics["DA"],
                )

            except Exception as exc:
                # Log failures but keep the pipeline running for the remaining models
                logger.warning("  [%s] FAILED on %s: %s", model.name, ticker, exc)

    return results, all_forecasts, all_actuals, all_dates


def main() -> None:
    cfg = load_config()

    results, all_forecasts, all_actuals, all_dates = run_pipeline(cfg)

    # Build the aggregated leaderboard (average metrics across all tickers)
    leaderboard = build_leaderboard(results)
    save_leaderboard(leaderboard)
    inject_into_readme()

    # Visualise results for the first ticker (used as the showcase in README)
    primary = cfg["data"]["tickers"][0]
    if primary in all_forecasts:
        plot_forecast_comparison(
            all_dates[primary],
            all_actuals[primary],
            all_forecasts[primary],
            ticker=primary,
        )
        plot_metrics_bar(leaderboard)

    # Print the final leaderboard to stdout
    print("\n" + "=" * 60)
    print("  FINAL LEADERBOARD (averaged across tickers)")
    print("=" * 60)
    print(leaderboard.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
