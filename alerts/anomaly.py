import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from configs.settings import get_model_settings

THRESHOLDS_PATH = "data/anomaly_thresholds.json"


def compute_residual_scores(
    actual: np.ndarray,
    predicted: np.ndarray,
    rolling_std_window: int | None = None,
) -> np.ndarray:
    settings = get_model_settings()
    if rolling_std_window is None:
        rolling_std_window = settings.anomaly.rolling_std_window
    residuals = actual - predicted
    rolling_std = (
        pd.Series(residuals)
        .rolling(window=rolling_std_window, min_periods=1)
        .std()
        .fillna(1.0)
        .values
    )
    rolling_std = np.where(rolling_std == 0, 1.0, rolling_std)
    return np.abs(residuals) / rolling_std


def compute_thresholds(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    scores = compute_residual_scores(actual, predicted)
    thresholds = {
        "warning": float(np.percentile(scores, 90)),
        "critical": float(np.percentile(scores, 98)),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }
    logger.info("computed_anomaly_thresholds", **thresholds)
    return thresholds


def save_thresholds(thresholds: dict[str, float]) -> None:
    path = Path(THRESHOLDS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(thresholds, f, indent=2)
    logger.info("saved_anomaly_thresholds", path=str(path))


def load_thresholds() -> dict[str, float]:
    path = Path(THRESHOLDS_PATH)
    if not path.exists():
        settings = get_model_settings()
        return {
            "warning": settings.anomaly.score_threshold_warning,
            "critical": settings.anomaly.score_threshold_critical,
        }
    with open(path) as f:
        return json.load(f)


def score_window_volatility(predictions: np.ndarray, thresholds: dict[str, float]) -> float:
    """Score anomaly based on detrended prediction volatility across a window.

    A healthy engine's predictions should follow a smooth linear trend
    (RUL decreases steadily across the window).  We fit a linear trend to
    the predictions, then measure the std of the residuals around that
    trend.  High residual std = erratic / anomalous behaviour.

    The residual std is normalised by the baseline std learned during
    training so the returned value is comparable to the warning/critical
    thresholds.
    """
    if len(predictions) < 2:
        return 0.0

    preds = np.asarray(predictions, dtype=float)
    n = len(preds)

    # Fit a linear trend: y = a*x + b using least-squares
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = preds.mean()
    slope = np.sum((x - x_mean) * (preds - y_mean)) / max(np.sum((x - x_mean) ** 2), 1e-12)
    intercept = y_mean - slope * x_mean

    # Residuals around the linear trend
    trend = slope * x + intercept
    residuals = preds - trend
    volatility = float(np.std(residuals))

    baseline_std = thresholds.get("std", 1.0)
    if baseline_std == 0:
        baseline_std = 1.0
    return volatility / baseline_std
