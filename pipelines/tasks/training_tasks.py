import pandas as pd
from loguru import logger
from prefect import task

from configs.settings import get_model_settings
from models.failure.lgbm_failure import FailureClassifierModel
from models.forecast.lgbm_forecast import RULForecastModel


@task(name="train_forecast_model")
def train_forecast(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[RULForecastModel, dict[str, float]]:
    model = RULForecastModel()
    metrics = model.train(x_train, y_train, x_val, y_val)
    logger.info("forecast_training_complete", **metrics)
    return model, metrics


@task(name="train_failure_model")
def train_failure(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[FailureClassifierModel, dict[str, float]]:
    model_settings = get_model_settings()
    cutoff = model_settings.failure_threshold.rul_cutoff
    y_train_binary = (y_train <= cutoff).astype(int)
    y_val_binary = (y_val <= cutoff).astype(int)
    model = FailureClassifierModel()
    metrics = model.train(x_train, y_train_binary, x_val, y_val_binary)
    logger.info("failure_training_complete", **metrics)
    return model, metrics
