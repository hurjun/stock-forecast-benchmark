"""
Plotting functions for benchmark results.

All functions save figures to disk via plt.savefig() and immediately close
the figure with plt.close() — never plt.show() — so the code runs correctly
in headless environments (servers, Kaggle, CI pipelines).
"""

import os

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_PALETTE = "tab10"   # 10-colour qualitative palette, works well for up to 7 models


def plot_forecast_comparison(
    dates: pd.Index,
    actual: np.ndarray,
    forecasts: dict[str, np.ndarray],
    ticker: str,
    out_dir: str = "results",
) -> None:
    """
    Line chart: actual closing prices vs each model's forecast.

    Args:
        dates:     DatetimeIndex of the test period (x-axis)
        actual:    Array of true closing prices
        forecasts: {model_name: predictions} dict
        ticker:    Ticker symbol shown in the chart title
        out_dir:   Directory where the PNG is saved
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    # Plot actual prices in black so it stands out against the coloured forecasts
    ax.plot(dates, actual, label="Actual", color="black", linewidth=2)

    palette = plt.get_cmap(_PALETTE).colors
    for i, (model_name, preds) in enumerate(forecasts.items()):
        ax.plot(
            dates, preds,
            label=model_name,
            color=palette[i % len(palette)],
            linewidth=1.2,
            alpha=0.85,
        )

    ax.set_title(f"Forecast Comparison — {ticker} (Test Period: 1991–2000)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (USD)")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "forecast_comparison.png"), dpi=150)
    plt.close()   # free memory; required when running many plots in sequence


def plot_metrics_bar(
    leaderboard_df: pd.DataFrame,
    out_dir: str = "results",
) -> None:
    """
    Three side-by-side bar charts showing MAE, RMSE, and MAPE per model.

    Lower bars are better for all three metrics.

    Args:
        leaderboard_df: Output of evaluation.leaderboard.build_leaderboard()
        out_dir:        Directory where the PNG is saved
    """
    metrics = ["MAE", "RMSE", "MAPE"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))

    for ax, metric in zip(axes, metrics):
        sns.barplot(
            data=leaderboard_df,
            x="model",        # one bar per model
            y=metric,
            ax=ax,
            palette=_PALETTE,
        )
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=40)   # rotate labels to avoid overlap

    fig.suptitle("Model Comparison — Average Across Tickers (1991–2000)", y=1.03)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "metrics_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
