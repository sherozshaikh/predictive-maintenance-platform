import math

import numpy as np
import pandas as pd

from features.engineering import build_features
from models.failure.lgbm_failure import FailureClassifierModel
from models.forecast.lgbm_forecast import RULForecastModel


def test_forecast_model_trains_and_predicts(sample_train_df: pd.DataFrame) -> None:
    featured_df, feature_columns = build_features(sample_train_df)
    split_idx = int(len(featured_df) * 0.8)
    x_train = featured_df.iloc[:split_idx][feature_columns]
    y_train = featured_df.iloc[:split_idx]["rul"]
    x_val = featured_df.iloc[split_idx:][feature_columns]
    y_val = featured_df.iloc[split_idx:]["rul"]

    model = RULForecastModel()
    metrics = model.train(x_train, y_train, x_val, y_val)

    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert metrics["rmse"] >= 0

    predictions = model.predict(x_val)
    assert len(predictions) == len(x_val)


def test_failure_model_trains_and_predicts(sample_train_df: pd.DataFrame) -> None:
    featured_df, feature_columns = build_features(sample_train_df)
    y_binary = (featured_df["rul"] <= 30).astype(int)

    engine_ids = featured_df["engine_id"].unique()
    train_engines = engine_ids[: len(engine_ids) - 1]
    val_engines = engine_ids[len(engine_ids) - 1 :]

    train_mask = featured_df["engine_id"].isin(train_engines)
    val_mask = featured_df["engine_id"].isin(val_engines)

    x_train = featured_df.loc[train_mask, feature_columns]
    y_train = y_binary.loc[train_mask]
    x_val = featured_df.loc[val_mask, feature_columns]
    y_val = y_binary.loc[val_mask]

    model = FailureClassifierModel()
    metrics = model.train(x_train, y_train, x_val, y_val)

    assert "auc_roc" in metrics
    assert "f1" in metrics
    if not math.isnan(metrics["auc_roc"]):
        assert 0 <= metrics["auc_roc"] <= 1

    probabilities = model.predict_proba(x_val)
    assert len(probabilities) == len(x_val)
    assert all(0 <= p <= 1 for p in probabilities)


def test_forecast_model_save_load(sample_train_df: pd.DataFrame, tmp_path) -> None:
    featured_df, feature_columns = build_features(sample_train_df)
    split_idx = int(len(featured_df) * 0.8)
    x_train = featured_df.iloc[:split_idx][feature_columns]
    y_train = featured_df.iloc[:split_idx]["rul"]
    x_val = featured_df.iloc[split_idx:][feature_columns]
    y_val = featured_df.iloc[split_idx:]["rul"]

    model = RULForecastModel()
    model.train(x_train, y_train, x_val, y_val)

    path = str(tmp_path / "model.pkl")
    model.save(path)

    loaded_model = RULForecastModel()
    loaded_model.load(path)

    original_preds = model.predict(x_val)
    loaded_preds = loaded_model.predict(x_val)
    np.testing.assert_array_almost_equal(original_preds, loaded_preds)
