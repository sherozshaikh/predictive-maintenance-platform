import json
from pathlib import Path

import pandas as pd
from loguru import logger
from prefect import task

from alerts.anomaly import compute_thresholds, save_thresholds
from models.failure.lgbm_failure import FailureClassifierModel
from models.forecast.lgbm_forecast import RULForecastModel

ARTIFACTS_DIR = "data/artifacts"


@task(name="evaluate_models")
def evaluate_models(
    forecast_metrics: dict[str, float],
    failure_metrics: dict[str, float],
) -> dict[str, dict[str, float]]:
    all_metrics = {}
    if forecast_metrics:
        all_metrics["forecast"] = forecast_metrics
    if failure_metrics:
        all_metrics["failure"] = failure_metrics
    logger.info("model_evaluation_summary", metrics=all_metrics)
    return all_metrics


@task(name="register_models")
def register_models(
    forecast_model: RULForecastModel | None,
    forecast_metrics: dict[str, float],
    failure_model: FailureClassifierModel | None,
    failure_metrics: dict[str, float],
    feature_columns: list[str],
    skip_mlflow: bool = False,
) -> None:
    artifacts_path = Path(ARTIFACTS_DIR)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    features_path = str(artifacts_path / "feature_columns.json")
    with open(features_path, "w") as f:
        json.dump(feature_columns, f)

    if forecast_model is not None:
        forecast_model.save(str(artifacts_path / "forecast_model.pkl"))

    if failure_model is not None:
        failure_model.save(str(artifacts_path / "failure_model.pkl"))

    if skip_mlflow:
        logger.info("skipped_mlflow_registration")
        return

    from pipelines.mlflow_logger import log_to_mlflow

    log_to_mlflow(
        forecast_model,
        forecast_metrics,
        failure_model,
        failure_metrics,
        artifacts_path,
        features_path,
    )


@task(name="compute_residual_thresholds")
def compute_and_save_thresholds(
    forecast_model: RULForecastModel,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, float]:
    predictions = forecast_model.predict(x_val)
    thresholds = compute_thresholds(y_val.values, predictions)
    save_thresholds(thresholds)
    logger.info("thresholds_computed_and_saved", **thresholds)
    return thresholds


@task(name="persist_outputs")
def persist_outputs(
    all_metrics: dict[str, dict[str, float]],
    thresholds: dict[str, float],
) -> None:
    output_path = Path(ARTIFACTS_DIR) / "pipeline_output.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {"metrics": all_metrics, "thresholds": thresholds}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("persisted_pipeline_outputs", path=str(output_path))
