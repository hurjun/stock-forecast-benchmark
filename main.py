"""
Entry point for the Stock Forecast Benchmark.

Usage:
    python main.py                       # full benchmark on real data (downloads via yfinance)
    python main.py --smoke               # fast synthetic smoke run (no network, seconds)
    python main.py --smoke --save-plots  # smoke run + save a forecast-vs-actual PNG
    python main.py --plot-leaderboard    # regen results/metrics_bar.png from committed CSV

What the full run does:
  1. Load configuration from config.yaml
  2. Download and preprocess stock data for each ticker (cached locally)
  3. Train all 9 forecasting models on the training period (1962-1992)
  4. Generate predictions for the test period (1993-2000)
  5. Compute MAE, RMSE, MAPE, and Directional Accuracy for every model/ticker pair
  6. Save a leaderboard (CSV + Markdown) and inject it into README.md
  7. Save forecast comparison and metrics bar charts to results/

The ``--smoke`` run generates a deterministic synthetic price series and trains
only the fast statistical baselines (Naive, ARIMA, ETS), letting a reviewer
verify the end-to-end pipeline in seconds without any network access.  It never
touches README.md or the committed results/.
"""

import argparse
import logging
import warnings

import numpy as np
import yaml

from evaluation.leaderboard import build_leaderboard, inject_into_readme, save_leaderboard
from evaluation.metrics import compute_all
from seed import SEED, set_global_seeds

# Silence the few high-volume but benign warning categories emitted by the
# statistical/deep-learning stack at import and fit time.  We deliberately keep
# this targeted (rather than a blanket ``filterwarnings("ignore")``) so that
# genuine numerical or runtime warnings still surface.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure root logger: INFO level shows training progress without debug noise
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
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

    Each model receives only the slice of config it needs so that model files
    never import from config.yaml directly.  The model classes are imported
    lazily here so that lightweight entry points (the smoke run, the test
    suite) do not need the full deep-learning stack installed.
    """
    from models.arima import ARIMAForecaster
    from models.ets_model import ETSForecaster
    from models.gru import GRUForecaster
    from models.lightgbm_model import LightGBMForecaster
    from models.lstm import LSTMForecaster
    from models.prophet_model import ProphetForecaster
    from models.tcn import TCNForecaster
    from models.transformer import TransformerForecaster
    from models.xgboost_model import XGBoostForecaster

    feat = cfg["features"]
    m    = cfg["models"]
    return [
        ARIMAForecaster(m["arima"]),
        ETSForecaster(m["ets"]),
        ProphetForecaster(m["prophet"]),
        XGBoostForecaster(m["xgboost"], feat),
        LightGBMForecaster(m["lightgbm"], feat),
        LSTMForecaster(m["lstm"]),
        GRUForecaster(m["gru"]),
        TCNForecaster(m["tcn"]),
        TransformerForecaster(m["transformer"]),
    ]


def _align(arr: np.ndarray, n: int) -> np.ndarray:
    """
    Make sure predict() output is exactly n elements long.

    Some models (e.g. Prophet with business-day calendars) may return slightly
    more or fewer steps than requested.
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
        results      - list of per-(model, ticker) metric dicts
        all_forecasts - {ticker: {model_name: predictions}}
        all_actuals   - {ticker: actual_close_array}
        all_dates     - {ticker: DatetimeIndex of test period}
    """
    from data.loader import load_ticker

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


def run_smoke(cfg: dict, save_plots: bool = False) -> "object":
    """
    Fast, network-free smoke benchmark on a deterministic synthetic series.

    Trains only the cheap statistical baselines (Naive, ARIMA, ETS) so a
    reviewer can confirm the full data -> feature -> split -> fit -> evaluate
    pipeline works in seconds.  Results are printed only; README.md and the
    committed leaderboard are left untouched.

    Args:
        cfg: The full parsed configuration dict.
        save_plots: When True, also write a forecast-vs-actual PNG
            (``results/smoke_forecast.png``) generated from this run's real
            outputs.  Off by default so the plain smoke run stays dependency-
            light and side-effect free.

    Returns:
        The smoke leaderboard DataFrame (for programmatic inspection / tests).
    """
    from data.loader import add_features, make_synthetic_prices, split_train_test
    from models.arima import ARIMAForecaster
    from models.ets_model import ETSForecaster
    from models.naive import NaiveForecaster

    logger.info("Running SMOKE benchmark on SYNTHETIC data (no network).")

    feat = cfg["features"]
    df = make_synthetic_prices(n_days=1300, start="1990-01-01", seed=SEED)
    df = add_features(df, feat["lags"], feat["rolling_windows"])

    smoke_cfg = {
        "train_start": "1990-01-01", "train_end": "1992-12-31",
        "test_start":  "1993-01-01", "test_end":  "1994-12-31",
        "target_col":  "Close",
    }
    train_df, test_df = split_train_test(df, smoke_cfg)
    actual  = test_df["Close"].values.astype(float)
    n_steps = len(test_df)
    logger.info("Synthetic split - train: %d rows, test: %d rows", len(train_df), n_steps)

    models = [
        NaiveForecaster(),
        ARIMAForecaster({"order": [1, 1, 0]}),
        ETSForecaster(cfg["models"]["ets"]),
    ]

    results: list[dict] = []
    forecasts: dict[str, np.ndarray] = {}
    for model in models:
        model.fit(train_df)
        preds = _align(model.predict(n_steps), n_steps)
        metrics = compute_all(actual, preds)
        results.append({"model": model.name, "ticker": "SYNTH", **metrics})
        forecasts[model.name] = preds
        logger.info(
            "  [%s] RMSE=%.4f  MAE=%.4f  MAPE=%.2f%%  DA=%.1f%%",
            model.name, metrics["RMSE"], metrics["MAE"], metrics["MAPE"], metrics["DA"],
        )

    if save_plots:
        # Render a forecast-vs-actual figure from this run's genuine outputs.
        # The title is explicit that the data is synthetic so the committed PNG
        # is never confused with the real ^GSPC/KO/IBM benchmark.
        from visualization.plots import plot_forecast_comparison

        plot_forecast_comparison(
            test_df.index,
            actual,
            forecasts,
            ticker="Synthetic",
            title="Smoke Run — Synthetic Random Walk (offline pipeline check, NOT the real benchmark)",
            filename="smoke_forecast.png",
        )
        logger.info("Wrote results/smoke_forecast.png from the synthetic smoke run.")

    leaderboard = build_leaderboard(results)
    print("\n" + "=" * 60)
    print("  SMOKE LEADERBOARD (synthetic data - NOT the real benchmark)")
    print("=" * 60)
    print(leaderboard.to_string(index=False))
    print("=" * 60)
    return leaderboard


def plot_committed_leaderboard(
    csv_path: str = "results/leaderboard.csv",
    out_dir: str = "results",
) -> None:
    """
    Regenerate ``results/metrics_bar.png`` from the committed leaderboard CSV.

    This lets a reviewer reproduce the per-model error-bar figure embedded in
    the README directly from the genuine committed results — offline, in
    seconds, without the ~3.5 h full benchmark or any network access.  It plots
    only the real numbers already stored in ``results/leaderboard.csv``; it
    never invents values.
    """
    import pandas as pd

    from visualization.plots import plot_metrics_bar

    df = pd.read_csv(csv_path)
    plot_metrics_bar(df, out_dir=out_dir)
    logger.info("Wrote %s/metrics_bar.png from %s", out_dir, csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock Forecast Benchmark")
    parser.add_argument(
        "--smoke",
        "--quick",
        action="store_true",
        dest="smoke",
        help="Run a fast synthetic smoke benchmark (no network) instead of the full run.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="With --smoke, also save a forecast-vs-actual PNG to results/.",
    )
    parser.add_argument(
        "--plot-leaderboard",
        action="store_true",
        help=(
            "Regenerate results/metrics_bar.png from the committed "
            "results/leaderboard.csv (no training, no network) and exit."
        ),
    )
    args = parser.parse_args()

    set_global_seeds(SEED)
    cfg = load_config()

    if args.plot_leaderboard:
        plot_committed_leaderboard()
        return

    if args.smoke:
        run_smoke(cfg, save_plots=args.save_plots)
        return

    results, all_forecasts, all_actuals, all_dates = run_pipeline(cfg)

    # Build the aggregated leaderboard (average metrics across all tickers)
    leaderboard = build_leaderboard(results)
    save_leaderboard(leaderboard)
    inject_into_readme()

    # Visualise results for the first ticker (used as the showcase in README)
    from visualization.plots import plot_forecast_comparison, plot_metrics_bar

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
