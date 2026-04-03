import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from configs.settings import get_model_settings


class FailureClassifierModel:
    def __init__(self) -> None:
        settings = get_model_settings()
        self._params = settings.failure.params.copy()
        self._name = settings.failure.name
        self._model: lgb.LGBMClassifier | None = None

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> dict[str, float]:
        self._model = lgb.LGBMClassifier(**self._params)
        self._model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        probabilities = self._model.predict_proba(x_val)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = {
            "auc_roc": float(roc_auc_score(y_val, probabilities)),
            "f1": float(f1_score(y_val, predictions)),
            "precision": float(precision_score(y_val, predictions)),
            "recall": float(recall_score(y_val, predictions)),
        }
        logger.info("trained_failure_model", **metrics)
        return metrics

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Failure model not trained or loaded.")
        return self._model.predict_proba(x)[:, 1]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        logger.info("saved_failure_model", path=path)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            self._model = pickle.load(f)
        logger.info("loaded_failure_model", path=path)

    @property
    def name(self) -> str:
        return self._name
