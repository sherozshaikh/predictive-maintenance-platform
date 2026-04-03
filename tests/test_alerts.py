import numpy as np

from alerts.anomaly import (
    compute_residual_scores,
    compute_thresholds,
    score_window_volatility,
)
from alerts.engine import AlertResult, compute_alert, compute_anomaly_level

# ---- compute_anomaly_level ----


def test_anomaly_level_low() -> None:
    assert compute_anomaly_level(0.5, warning_threshold=2.0, critical_threshold=3.0) == "low"


def test_anomaly_level_medium() -> None:
    assert compute_anomaly_level(2.5, warning_threshold=2.0, critical_threshold=3.0) == "medium"


def test_anomaly_level_high() -> None:
    assert compute_anomaly_level(3.5, warning_threshold=2.0, critical_threshold=3.0) == "high"


def test_anomaly_level_exact_warning_boundary() -> None:
    assert compute_anomaly_level(2.0, warning_threshold=2.0, critical_threshold=3.0) == "medium"


def test_anomaly_level_exact_critical_boundary() -> None:
    assert compute_anomaly_level(3.0, warning_threshold=2.0, critical_threshold=3.0) == "high"


# ---- compute_alert ----


def test_alert_critical_low_rul_high_failure() -> None:
    """RUL < 8 and failure_prob > 0.7 => CRITICAL."""
    result = compute_alert(
        engine_id=1,
        anomaly_score=0.5,
        anomaly_level="low",
        failure_probability=0.8,
        forecast_rul=5.0,
    )
    assert isinstance(result, AlertResult)
    assert result.alert_level == "CRITICAL"


def test_alert_critical_high_anomaly_high_failure() -> None:
    """High anomaly + failure_prob > 0.7 => CRITICAL regardless of RUL."""
    result = compute_alert(
        engine_id=1,
        anomaly_score=5.0,
        anomaly_level="high",
        failure_probability=0.8,
        forecast_rul=50.0,
    )
    assert result.alert_level == "CRITICAL"


def test_alert_warning_moderate_rul() -> None:
    """RUL < 15 and failure_prob > 0.5 => WARNING."""
    result = compute_alert(
        engine_id=1,
        anomaly_score=0.5,
        anomaly_level="low",
        failure_probability=0.6,
        forecast_rul=10.0,
    )
    assert result.alert_level == "WARNING"


def test_alert_warning_medium_anomaly() -> None:
    """Medium anomaly alone => WARNING."""
    result = compute_alert(
        engine_id=1,
        anomaly_score=2.5,
        anomaly_level="medium",
        failure_probability=0.3,
        forecast_rul=80.0,
    )
    assert result.alert_level == "WARNING"


def test_alert_normal_high_rul_low_failure() -> None:
    """High RUL + low failure_prob + low anomaly => NORMAL."""
    result = compute_alert(
        engine_id=1,
        anomaly_score=0.5,
        anomaly_level="low",
        failure_probability=0.2,
        forecast_rul=90.0,
    )
    assert result.alert_level == "NORMAL"


def test_alert_normal_high_rul_even_with_moderate_failure() -> None:
    """RUL >= 15 with moderate failure_prob and low anomaly => NORMAL."""
    result = compute_alert(
        engine_id=1,
        anomaly_score=0.5,
        anomaly_level="low",
        failure_probability=0.6,
        forecast_rul=20.0,
    )
    assert result.alert_level == "NORMAL"


def test_alert_high_anomaly_low_failure_is_warning_not_critical() -> None:
    """High anomaly with low failure_prob => WARNING, not CRITICAL."""
    result = compute_alert(
        engine_id=1,
        anomaly_score=5.0,
        anomaly_level="high",
        failure_probability=0.3,
        forecast_rul=80.0,
    )
    assert result.alert_level == "WARNING"


def test_alert_boundary_rul_8() -> None:
    """RUL exactly 8 with high failure => WARNING, not CRITICAL (boundary)."""
    result = compute_alert(
        engine_id=1,
        anomaly_score=0.5,
        anomaly_level="low",
        failure_probability=0.8,
        forecast_rul=8.0,
    )
    assert result.alert_level == "WARNING"


def test_alert_boundary_rul_15() -> None:
    """RUL exactly 15 with moderate failure => NORMAL (boundary)."""
    result = compute_alert(
        engine_id=1,
        anomaly_score=0.5,
        anomaly_level="low",
        failure_probability=0.6,
        forecast_rul=15.0,
    )
    assert result.alert_level == "NORMAL"


# ---- score_window_volatility (detrended) ----


def test_volatility_single_prediction_returns_zero() -> None:
    preds = np.array([42.0])
    assert score_window_volatility(preds, {"std": 5.0}) == 0.0


def test_volatility_constant_predictions_returns_zero() -> None:
    preds = np.array([10.0, 10.0, 10.0, 10.0])
    assert score_window_volatility(preds, {"std": 5.0}) == 0.0


def test_volatility_linear_trend_returns_zero() -> None:
    """A perfect linear decrease (normal degradation) should score ~0."""
    preds = np.array([100.0, 90.0, 80.0, 70.0, 60.0, 50.0])
    score = score_window_volatility(preds, {"std": 5.0})
    assert score < 0.01  # essentially zero


def test_volatility_erratic_predictions_scores_high() -> None:
    """Predictions that jump around a trend should score high."""
    # Trend is roughly 100 -> 50, but with wild oscillations
    preds = np.array([100.0, 50.0, 95.0, 45.0, 90.0, 40.0])
    score = score_window_volatility(preds, {"std": 5.0})
    assert score > 1.0


def test_volatility_slight_noise_scores_low() -> None:
    """Small noise around a linear trend should score low."""
    trend = np.linspace(100, 60, 20)
    rng = np.random.RandomState(42)
    preds = trend + rng.normal(0, 0.5, size=20)
    score = score_window_volatility(preds, {"std": 5.0})
    assert score < 0.5


def test_volatility_baseline_std_zero_fallback() -> None:
    preds = np.array([10.0, 20.0, 5.0, 25.0])  # erratic
    score = score_window_volatility(preds, {"std": 0.0})
    assert score > 0  # should not divide by zero


def test_volatility_missing_std_key() -> None:
    preds = np.array([10.0, 20.0, 5.0, 25.0])  # erratic
    score = score_window_volatility(preds, {})
    assert score > 0  # default std=1.0


# ---- compute_residual_scores ----


def test_residual_scores_shape() -> None:
    actual = np.array([100.0, 90.0, 80.0, 70.0, 60.0])
    predicted = np.array([95.0, 85.0, 78.0, 72.0, 55.0])
    scores = compute_residual_scores(actual, predicted, rolling_std_window=3)
    assert scores.shape == actual.shape
    assert np.all(scores >= 0)


def test_residual_scores_perfect_predictions() -> None:
    values = np.array([100.0, 90.0, 80.0, 70.0])
    scores = compute_residual_scores(values, values, rolling_std_window=3)
    assert np.allclose(scores, 0.0)


# ---- compute_thresholds ----


def test_thresholds_keys() -> None:
    actual = np.random.RandomState(42).normal(100, 10, size=200)
    predicted = actual + np.random.RandomState(43).normal(0, 5, size=200)
    thresholds = compute_thresholds(actual, predicted)
    assert "warning" in thresholds
    assert "critical" in thresholds
    assert "mean" in thresholds
    assert "std" in thresholds
    assert thresholds["warning"] <= thresholds["critical"]
