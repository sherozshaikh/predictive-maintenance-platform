import json
from pathlib import Path

import numpy as np
from loguru import logger

from alerts.anomaly import compute_thresholds, save_thresholds
from configs.settings import get_model_settings, get_pipeline_settings
from features.engineering import build_features
from features.ingestion import load_train_data
from features.validation import validate_raw
from models.failure.lgbm_failure import FailureClassifierModel
from models.forecast.lgbm_forecast import RULForecastModel
from storage.duckdb_store import write_dataframe
from storage.sqlite_store import init_tables

ARTIFACTS_DIR = "data/artifacts"


def run_training(
    run_forecast: bool = True,
    run_failure: bool = True,
    skip_mlflow: bool = False,
) -> dict:
    init_tables()
    settings = get_pipeline_settings()
    model_settings = get_model_settings()

    logger.info("direct_runner_started")

    train_df = load_train_data()
    train_validated = validate_raw(train_df)
    featured_df, feature_columns = build_features(train_validated)
    write_dataframe("train_features", featured_df)

    rng = np.random.RandomState(settings.split.random_state)
    engine_ids = featured_df["engine_id"].unique()
    n_test = max(1, int(len(engine_ids) * settings.split.test_engine_fraction))
    test_engines = rng.choice(engine_ids, size=n_test, replace=False)

    train_mask = ~featured_df["engine_id"].isin(test_engines)
    val_mask = featured_df["engine_id"].isin(test_engines)

    x_train = featured_df.loc[train_mask, feature_columns]
    y_train = featured_df.loc[train_mask, "rul"]
    x_val = featured_df.loc[val_mask, feature_columns]
    y_val = featured_df.loc[val_mask, "rul"]

    forecast_model, forecast_metrics = None, {}
    failure_model, failure_metrics = None, {}

    if run_forecast:
        forecast_model = RULForecastModel()
        forecast_metrics = forecast_model.train(x_train, y_train, x_val, y_val)
        logger.info("forecast_training_complete", **forecast_metrics)

    if run_failure:
        cutoff = model_settings.failure_threshold.rul_cutoff
        y_train_binary = (y_train <= cutoff).astype(int)
        y_val_binary = (y_val <= cutoff).astype(int)
        failure_model = FailureClassifierModel()
        failure_metrics = failure_model.train(x_train, y_train_binary, x_val, y_val_binary)
        logger.info("failure_training_complete", **failure_metrics)

    artifacts_path = Path(ARTIFACTS_DIR)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    features_path = str(artifacts_path / "feature_columns.json")
    with open(features_path, "w") as f:
        json.dump(feature_columns, f)

    if forecast_model is not None:
        forecast_model.save(str(artifacts_path / "forecast_model.pkl"))
    if failure_model is not None:
        failure_model.save(str(artifacts_path / "failure_model.pkl"))

    all_metrics = {}
    if forecast_metrics:
        all_metrics["forecast"] = forecast_metrics
    if failure_metrics:
        all_metrics["failure"] = failure_metrics

    thresholds = {}
    if forecast_model is not None:
        predictions = forecast_model.predict(x_val)
        thresholds = compute_thresholds(y_val.values, predictions)
        save_thresholds(thresholds)

    output_path = artifacts_path / "pipeline_output.json"
    with open(output_path, "w") as f:
        json.dump({"metrics": all_metrics, "thresholds": thresholds}, f, indent=2)

    if not skip_mlflow:
        from pipelines.mlflow_logger import log_to_mlflow

        log_to_mlflow(
            forecast_model,
            forecast_metrics,
            failure_model,
            failure_metrics,
            artifacts_path,
            features_path,
        )

    logger.info("direct_runner_complete", metrics=all_metrics)
    return {"metrics": all_metrics, "thresholds": thresholds}
