import os
from pathlib import Path

from loguru import logger

from models.failure.lgbm_failure import FailureClassifierModel
from models.forecast.lgbm_forecast import RULForecastModel


def log_to_mlflow(
    forecast_model: RULForecastModel | None,
    forecast_metrics: dict[str, float],
    failure_model: FailureClassifierModel | None,
    failure_metrics: dict[str, float],
    artifacts_path: Path,
    features_path: str,
) -> None:
    """Log trained models and metrics to MLflow.

    Gracefully handles missing mlflow, unreachable server (falls back
    to local file tracking), and individual run failures.
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow_not_installed_skipping")
        return

    from configs.settings import get_infra_settings, get_pipeline_settings

    settings = get_pipeline_settings()
    infra = get_infra_settings()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", infra.mlflow.tracking_uri)

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(settings.training.mlflow_experiment)
        logger.info("mlflow_connected", tracking_uri=tracking_uri)
    except Exception:
        logger.warning("mlflow_server_unreachable_using_local_tracking")
        mlflow.set_tracking_uri("file:///tmp/mlruns")
        mlflow.set_experiment(settings.training.mlflow_experiment)

    if forecast_model is not None:
        with mlflow.start_run(run_name="forecast_model") as run:
            mlflow.log_params({"model_type": "lightgbm", "task": "regression"})
            mlflow.log_metrics(forecast_metrics)
            mlflow.log_artifact(str(artifacts_path / "forecast_model.pkl"))
            mlflow.log_artifact(features_path)
            mlflow.set_tag("model_type", "forecast")
            logger.info("registered_forecast_model", run_id=run.info.run_id)

    if failure_model is not None:
        with mlflow.start_run(run_name="failure_model") as run:
            mlflow.log_params({"model_type": "lightgbm", "task": "classification"})
            mlflow.log_metrics(failure_metrics)
            mlflow.log_artifact(str(artifacts_path / "failure_model.pkl"))
            mlflow.set_tag("model_type", "failure")
            logger.info("registered_failure_model", run_id=run.info.run_id)
