"""
Regression metrics for forecast evaluation.

All functions take two 1-D NumPy arrays (actual, predicted) of the same length
and return a single float.
"""

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error — average absolute dollar deviation."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error — penalises large errors more heavily than MAE."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (%).

    Scale-independent, so models trained on different tickers can be compared
    fairly.  Rows where y_true == 0 are skipped to avoid division by zero.
    """
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional Accuracy (%).

    Measures how often the model correctly predicts whether the price goes
    up or down — more relevant for trading decisions than raw error magnitude.
    A random classifier scores 50 %; anything above is informative.
    """
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float(np.mean(true_dir == pred_dir) * 100)


def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all four metrics and return them as a dictionary."""
    return {
        "MAE":  mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "DA":   directional_accuracy(y_true, y_pred),
    }
