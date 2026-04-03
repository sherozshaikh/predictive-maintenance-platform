"""Edge-case tests for feature engineering helpers."""

import numpy as np
import pandas as pd
import pytest

from features.engineering import (
    _linear_slope,
    _normalize_cycle_index,
    build_features_for_window,
)

# ---- _normalize_cycle_index ----


def test_normalize_cycle_with_engine_id() -> None:
    df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
        }
    )
    result = _normalize_cycle_index(df.copy())
    # engine 1: max_cycle=3, engine 2: max_cycle=2
    assert result.loc[0, "cycle_norm"] == pytest.approx(1 / 3)
    assert result.loc[2, "cycle_norm"] == pytest.approx(1.0)
    assert result.loc[3, "cycle_norm"] == pytest.approx(0.5)
    assert result.loc[4, "cycle_norm"] == pytest.approx(1.0)


def test_normalize_cycle_without_engine_id() -> None:
    df = pd.DataFrame({"cycle": [1, 2, 4]})
    result = _normalize_cycle_index(df.copy())
    assert result["cycle_norm"].max() == pytest.approx(1.0)
    assert result["cycle_norm"].min() == pytest.approx(0.25)


def test_normalize_cycle_single_row() -> None:
    """Single row means max_cycle == cycle, so cycle_norm == 1.0."""
    df = pd.DataFrame({"engine_id": [1], "cycle": [5]})
    result = _normalize_cycle_index(df.copy())
    assert result.loc[0, "cycle_norm"] == pytest.approx(1.0)


def test_normalize_cycle_zero_cycle() -> None:
    """Edge case: cycle=0 should not cause division by zero."""
    df = pd.DataFrame({"cycle": [0]})
    result = _normalize_cycle_index(df.copy())
    # max_cycle is 0, clamped to 1, so cycle_norm = 0/1 = 0.0
    assert result.loc[0, "cycle_norm"] == pytest.approx(0.0)


# ---- _linear_slope ----


def test_linear_slope_constant_series() -> None:
    series = pd.Series([5.0] * 20)
    slopes = _linear_slope(series)
    assert len(slopes) == 20
    assert np.allclose(slopes.values, 0.0)


def test_linear_slope_single_value() -> None:
    series = pd.Series([42.0])
    slopes = _linear_slope(series)
    assert len(slopes) == 1
    assert slopes.iloc[0] == 0.0


def test_linear_slope_two_values() -> None:
    series = pd.Series([0.0, 10.0])
    slopes = _linear_slope(series)
    assert slopes.iloc[1] == pytest.approx(10.0)


def test_linear_slope_perfect_line() -> None:
    """y = 2x: slope should be ~2.0 once the window is full."""
    series = pd.Series([2.0 * i for i in range(20)])
    slopes = _linear_slope(series)
    # After the window of 10 is full, every slope should be 2.0
    assert slopes.iloc[-1] == pytest.approx(2.0)


def test_linear_slope_short_series() -> None:
    """Series shorter than window (10) should still work."""
    series = pd.Series([1.0, 3.0, 5.0])
    slopes = _linear_slope(series)
    assert len(slopes) == 3
    assert slopes.iloc[0] == 0.0  # first element always 0


# ---- build_features_for_window edge cases ----


def test_build_features_minimal_window() -> None:
    """A window with very few rows may produce empty output after dropna."""
    from configs.settings import get_pipeline_settings

    settings = get_pipeline_settings()
    # 2 rows is below the max lag period (10), so all lag columns will have NaN
    # After dropna, we might get 0 rows -- the function should handle this gracefully
    rng = np.random.RandomState(42)
    rows = []
    for cycle in range(1, 3):
        row = {"engine_id": 1, "cycle": cycle, "op1": 0.0, "op2": 0.0, "op3": 100.0}
        for s in settings.data.sensor_columns:
            row[s] = rng.normal(500, 10)
        rows.append(row)
    df = pd.DataFrame(rows)
    featured_df, feature_columns = build_features_for_window(df)
    assert isinstance(featured_df, pd.DataFrame)
    assert isinstance(feature_columns, list)
