import pandas as pd

from configs.settings import get_pipeline_settings
from features.engineering import build_features, build_features_for_window


def test_build_features_creates_lag_columns(sample_train_df: pd.DataFrame) -> None:
    settings = get_pipeline_settings()
    featured_df, feature_columns = build_features(sample_train_df)

    for sensor in settings.data.selected_sensors[:2]:
        for lag in settings.features.lag_periods:
            assert f"{sensor}_lag_{lag}" in feature_columns


def test_build_features_creates_rolling_columns(sample_train_df: pd.DataFrame) -> None:
    settings = get_pipeline_settings()
    featured_df, feature_columns = build_features(sample_train_df)

    for sensor in settings.data.selected_sensors[:2]:
        for window in settings.features.rolling_windows:
            assert f"{sensor}_rmean_{window}" in feature_columns
            assert f"{sensor}_rstd_{window}" in feature_columns


def test_build_features_creates_delta_columns(sample_train_df: pd.DataFrame) -> None:
    settings = get_pipeline_settings()
    featured_df, feature_columns = build_features(sample_train_df)

    for sensor in settings.data.selected_sensors[:2]:
        assert f"{sensor}_delta" in feature_columns


def test_build_features_creates_trend_columns(sample_train_df: pd.DataFrame) -> None:
    settings = get_pipeline_settings()
    featured_df, feature_columns = build_features(sample_train_df)

    for sensor in settings.data.selected_sensors[:2]:
        assert f"{sensor}_trend" in feature_columns


def test_build_features_no_nulls_in_output(sample_train_df: pd.DataFrame) -> None:
    featured_df, feature_columns = build_features(sample_train_df)
    assert featured_df[feature_columns].isnull().sum().sum() == 0


def test_build_features_has_cycle_norm(sample_train_df: pd.DataFrame) -> None:
    featured_df, feature_columns = build_features(sample_train_df)
    assert "cycle_norm" in feature_columns
    assert featured_df["cycle_norm"].max() <= 1.0
    assert featured_df["cycle_norm"].min() >= 0.0


def test_build_features_for_window(sample_window_df: pd.DataFrame) -> None:
    featured_df, feature_columns = build_features_for_window(sample_window_df)
    assert len(featured_df) > 0
    assert "cycle_norm" in feature_columns
