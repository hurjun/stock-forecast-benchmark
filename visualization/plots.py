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

# Colours used to highlight the winner and loser in bar charts
_COLOR_BEST  = "#2ca02c"   # green  — lowest error / highest DA
_COLOR_WORST = "#d62728"   # red    — highest error / lowest DA
_COLOR_MID   = "#aec7e8"   # light blue — all other models


def plot_forecast_comparison(
    dates: pd.Index,
    actual: np.ndarray,
    forecasts: dict[str, np.ndarray],
    ticker: str,
    out_dir: str = "results",
    title: str | None = None,
    filename: str = "forecast_comparison.png",
) -> None:
    """
    Line chart: actual closing prices vs each model's forecast.

    Args:
        dates:     DatetimeIndex of the test period (x-axis)
        actual:    Array of true closing prices
        forecasts: {model_name: predictions} dict
        ticker:    Ticker symbol shown in the (default) chart title
        out_dir:   Directory where the PNG is saved
        title:     Optional explicit chart title. When ``None`` (default) the
                   original real-benchmark title is used; callers running on a
                   different window (e.g. the synthetic smoke run) pass an
                   honest title so the figure is never mislabelled.
        filename:  Output PNG file name (lets the smoke run write a distinct
                   file without clobbering the full-run figure).
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    # Plot actual prices in black so it stands out against the coloured forecasts
    ax.plot(dates, actual, label="Actual", color="black", linewidth=2.5, zorder=10)

    palette = plt.get_cmap(_PALETTE).colors
    for i, (model_name, preds) in enumerate(forecasts.items()):
        ax.plot(
            dates, preds,
            label=model_name,
            color=palette[i % len(palette)],
            linewidth=1.2,
            alpha=0.85,
        )

    ax.set_title(
        title or f"Forecast Comparison — {ticker} (Test Period: 1993–2000)",
        fontsize=13,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (USD)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, filename), dpi=150)
    plt.close()   # free memory; required when running many plots in sequence


def _bar_colors(values: pd.Series, higher_is_better: bool) -> list[str]:
    """
    Return a colour list: green for the best bar, red for the worst, grey for the rest.

    Args:
        values:           Series of metric values aligned with the DataFrame rows.
        higher_is_better: True for DA (higher = better), False for MAE/RMSE/MAPE.
    """
    colors = [_COLOR_MID] * len(values)
    best_idx  = int(values.idxmax()) if higher_is_better else int(values.idxmin())
    worst_idx = int(values.idxmin()) if higher_is_better else int(values.idxmax())
    colors[best_idx]  = _COLOR_BEST
    colors[worst_idx] = _COLOR_WORST
    return colors


def _add_bar_labels(ax: plt.Axes, fmt: str = "{:.1f}") -> None:
    """Annotate each bar with its numeric value just above the bar top."""
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )


def plot_metrics_bar(
    leaderboard_df: pd.DataFrame,
    out_dir: str = "results",
) -> None:
    """
    Four side-by-side bar charts: MAE, RMSE, MAPE (lower = better) and DA (higher = better).

    The best-performing model is highlighted in green; the worst in red.
    Each bar is labelled with its numeric value.

    Args:
        leaderboard_df: Output of evaluation.leaderboard.build_leaderboard()
        out_dir:        Directory where the PNG is saved
    """
    # Sort by RMSE ascending so bars are ordered best → worst left to right
    df = leaderboard_df.sort_values("RMSE").reset_index(drop=True)

    metrics = [
        ("MAE",  False, "{:.1f}"),
        ("RMSE", False, "{:.1f}"),
        ("MAPE", False, "{:.1f}%"),
        ("DA",   True,  "{:.1f}%"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax, (metric, higher_is_better, fmt) in zip(axes, metrics):
        colors = _bar_colors(df[metric], higher_is_better)

        sns.barplot(
            data=df,
            x="model",
            y=metric,
            ax=ax,
            palette=colors,
            order=df["model"],   # preserve the RMSE-sorted order
        )
        _add_bar_labels(ax, fmt=fmt)

        direction = "higher = better" if higher_is_better else "lower = better"
        ax.set_title(f"{metric}\n({direction})", fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=40)

    # Shared legend explaining the colour coding
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=_COLOR_BEST,  label="Best model"),
        Patch(color=_COLOR_WORST, label="Worst model"),
        Patch(color=_COLOR_MID,   label="Others"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.02),
        framealpha=0.9,
    )

    fig.suptitle(
        "Model Comparison — Average Across ^GSPC, KO, IBM (Test: 1993–2000)",
        y=1.08,
        fontsize=12,
    )
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "metrics_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()
