import numpy as np
import pandas as pd
from loguru import logger
from numpy.lib.stride_tricks import sliding_window_view

from configs.settings import get_pipeline_settings


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    settings = get_pipeline_settings()
    sensors = settings.data.selected_sensors
    lag_periods = settings.features.lag_periods
    rolling_windows = settings.features.rolling_windows

    df = df.copy()
    df = _normalize_cycle_index(df)

    feature_columns: list[str] = ["cycle_norm"] + list(sensors)

    grouped = df.groupby("engine_id")

    lag_features = []
    for sensor in sensors:
        for lag in lag_periods:
            col_name = f"{sensor}_lag_{lag}"
            col = grouped[sensor].shift(lag).rename(col_name)
            lag_features.append(col)
            feature_columns.append(col_name)

    if lag_features:
        df = pd.concat([df] + lag_features, axis=1)

    rolling_features = []
    for sensor in sensors:
        series = grouped[sensor]

        for window_size in rolling_windows:
            mean_col = f"{sensor}_rmean_{window_size}"
            std_col = f"{sensor}_rstd_{window_size}"

            mean_series = series.transform(
                lambda x, w=window_size: x.rolling(window=w, min_periods=1).mean()
            ).rename(mean_col)

            std_series = series.transform(
                lambda x, w=window_size: x.rolling(window=w, min_periods=1).std().fillna(0)
            ).rename(std_col)

            rolling_features.extend([mean_series, std_series])
            feature_columns.extend([mean_col, std_col])

    if rolling_features:
        df = pd.concat([df] + rolling_features, axis=1)

    delta_features = []
    for sensor in sensors:
        col_name = f"{sensor}_delta"
        col = grouped[sensor].diff().fillna(0).rename(col_name)
        delta_features.append(col)
        feature_columns.append(col_name)

    if delta_features:
        df = pd.concat([df] + delta_features, axis=1)

    trend_features = []
    for sensor in sensors:
        col_name = f"{sensor}_trend"
        col = grouped[sensor].transform(_linear_slope).rename(col_name)
        trend_features.append(col)
        feature_columns.append(col_name)

    if trend_features:
        df = pd.concat([df] + trend_features, axis=1)

    df = df.copy()
    df = df.dropna(subset=feature_columns).reset_index(drop=True)

    logger.info("built_features", total_features=len(feature_columns), rows=len(df))
    return df, feature_columns


def build_features_for_window(
    window_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    settings = get_pipeline_settings()
    sensors = settings.data.selected_sensors
    lag_periods = settings.features.lag_periods
    rolling_windows = settings.features.rolling_windows

    df = window_df.copy()
    df = _normalize_cycle_index(df)

    feature_columns: list[str] = ["cycle_norm"] + list(sensors)

    lag_features = []
    for sensor in sensors:
        for lag in lag_periods:
            col_name = f"{sensor}_lag_{lag}"
            col = df[sensor].shift(lag).rename(col_name)
            lag_features.append(col)
            feature_columns.append(col_name)

    if lag_features:
        df = pd.concat([df] + lag_features, axis=1)

    rolling_features = []
    for sensor in sensors:
        series = df[sensor]

        for window in rolling_windows:
            mean_col = f"{sensor}_rmean_{window}"
            std_col = f"{sensor}_rstd_{window}"

            mean_series = series.rolling(window=window, min_periods=1).mean().rename(mean_col)
            std_series = (
                series.rolling(window=window, min_periods=1).std().fillna(0).rename(std_col)
            )

            rolling_features.extend([mean_series, std_series])
            feature_columns.extend([mean_col, std_col])

    if rolling_features:
        df = pd.concat([df] + rolling_features, axis=1)

    delta_features = []
    for sensor in sensors:
        col_name = f"{sensor}_delta"
        col = df[sensor].diff().fillna(0).rename(col_name)
        delta_features.append(col)
        feature_columns.append(col_name)

    if delta_features:
        df = pd.concat([df] + delta_features, axis=1)

    trend_features = []
    for sensor in sensors:
        col_name = f"{sensor}_trend"
        col = _linear_slope(df[sensor]).rename(col_name)
        trend_features.append(col)
        feature_columns.append(col_name)

    if trend_features:
        df = pd.concat([df] + trend_features, axis=1)

    df = df.copy()
    df = df.dropna(subset=feature_columns)

    return df, feature_columns


def _normalize_cycle_index(df: pd.DataFrame) -> pd.DataFrame:
    if "engine_id" in df.columns:
        max_cycles = df.groupby("engine_id")["cycle"].transform("max").replace(0, 1)
    else:
        max_cycles = df["cycle"].max()
        if max_cycles == 0:
            max_cycles = 1
    df["cycle_norm"] = df["cycle"] / max_cycles
    return df


def _linear_slope(series: pd.Series) -> pd.Series:
    window = 10
    values = series.values.astype(float)
    n = len(values)
    result = np.zeros(n)

    if n <= 1:
        return pd.Series(result, index=series.index)

    # For indices < window, compute slopes with expanding windows (2..window-1)
    for i in range(1, min(window, n)):
        y = values[: i + 1]
        if np.std(y) == 0:
            result[i] = 0.0
        else:
            x = np.arange(i + 1, dtype=float)
            result[i] = np.polyfit(x, y, 1)[0]

    # For indices >= window, use vectorized rolling OLS slope:
    # slope = (n*sum(x*y) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
    if n >= window:
        x_fixed = np.arange(window, dtype=float)
        sum_x = x_fixed.sum()
        sum_x2 = (x_fixed**2).sum()
        denom = window * sum_x2 - sum_x**2

        windows_y = sliding_window_view(values, window)
        sum_xy = windows_y @ x_fixed
        sum_y = windows_y.sum(axis=1)

        slopes = (window * sum_xy - sum_x * sum_y) / denom

        # Where std is 0, slope should be 0
        std_y = windows_y.std(axis=1)
        slopes[std_y == 0] = 0.0

        result[window - 1 :] = slopes

    return pd.Series(result, index=series.index)
