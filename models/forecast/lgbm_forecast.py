import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from configs.settings import get_model_settings


class RULForecastModel:
    def __init__(self) -> None:
        settings = get_model_settings()
        self._params = settings.forecast.params.copy()
        self._name = settings.forecast.name
        self._model: lgb.LGBMRegressor | None = None

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, float]:
        self._model = lgb.LGBMRegressor(**self._params)
        self._model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        predictions = self._model.predict(x_val)
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_val, predictions))),
            "mae": float(mean_absolute_error(y_val, predictions)),
            "r2": float(r2_score(y_val, predictions)),
        }
        logger.info("trained_forecast_model", **metrics)
        return metrics

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Forecast model not trained or loaded.")
        return self._model.predict(x)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info("saved_forecast_model", path=path)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        logger.info("loaded_forecast_model", path=path)

    @property
    def name(self) -> str:
        return self._name

    @property
    def feature_importances(self) -> dict[str, float]:
        if self._model is None:
            return {}
        return dict(
            zip(
                self._model.feature_name_,
                self._model.feature_importances_.tolist(),
                strict=True,
            )
        )
