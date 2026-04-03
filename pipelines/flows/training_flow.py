import sys

from loguru import logger
from prefect import flow

from pipelines.tasks.data_tasks import (
    engineer_features,
    ingest,
    split_time_series,
    validate,
)
from pipelines.tasks.evaluation_tasks import (
    compute_and_save_thresholds,
    evaluate_models,
    persist_outputs,
    register_models,
)
from pipelines.tasks.training_tasks import train_failure, train_forecast
from storage.sqlite_store import init_tables

logger.remove()
logger.add(sys.stderr, serialize=True)


@flow(name="predictive_maintenance_training", log_prints=True)
def training_pipeline(
    run_forecast: bool = True,
    run_failure: bool = True,
    skip_mlflow: bool = False,
) -> dict:
    init_tables()

    train_df = ingest()
    train_validated = validate(train_df)
    featured_df, feature_columns = engineer_features(train_validated)
    x_train, y_train, x_val, y_val, feature_cols = split_time_series(featured_df, feature_columns)

    forecast_model, forecast_metrics = None, {}
    failure_model, failure_metrics = None, {}

    if run_forecast:
        forecast_model, forecast_metrics = train_forecast(x_train, y_train, x_val, y_val)

    if run_failure:
        failure_model, failure_metrics = train_failure(x_train, y_train, x_val, y_val)

    all_metrics = evaluate_models(forecast_metrics, failure_metrics)

    register_models(
        forecast_model=forecast_model,
        forecast_metrics=forecast_metrics,
        failure_model=failure_model,
        failure_metrics=failure_metrics,
        feature_columns=feature_cols,
        skip_mlflow=skip_mlflow,
    )

    thresholds = {}
    if forecast_model is not None:
        thresholds = compute_and_save_thresholds(forecast_model, x_val, y_val)

    persist_outputs(all_metrics, thresholds)

    logger.info("training_pipeline_complete")
    return {"metrics": all_metrics, "thresholds": thresholds}
