import json
import threading
from pathlib import Path

import pandas as pd
from loguru import logger

from alerts.anomaly import load_thresholds, score_window_volatility
from alerts.engine import compute_alert, compute_anomaly_level
from apps.api.schemas import ScoreWindowRequest
from configs.settings import get_model_settings
from features.engineering import build_features_for_window
from models.failure.lgbm_failure import FailureClassifierModel
from models.forecast.lgbm_forecast import RULForecastModel

ARTIFACTS_DIR = "data/artifacts"


class ModelNotLoadedError(Exception):
    """Raised when scoring is attempted before models are loaded."""


class ScoringService:
    def __init__(self) -> None:
        self._forecast_model: RULForecastModel | None = None
        self._failure_model: FailureClassifierModel | None = None
        self._feature_columns: list[str] | None = None
        self._thresholds: dict[str, float] | None = None
        self._loaded = False
        self._lock = threading.Lock()

    def load_models(self) -> None:
        with self._lock:
            artifacts = Path(ARTIFACTS_DIR)
            forecast_path = artifacts / "forecast_model.pkl"
            failure_path = artifacts / "failure_model.pkl"
            features_path = artifacts / "feature_columns.json"

            missing = [p for p in [forecast_path, failure_path, features_path] if not p.exists()]
            if missing:
                logger.warning(
                    "model_artifacts_not_found",
                    missing=[str(p) for p in missing],
                )
                return

            forecast = RULForecastModel()
            forecast.load(str(forecast_path))

            failure = FailureClassifierModel()
            failure.load(str(failure_path))

            with open(features_path) as f:
                feature_columns = json.load(f)

            thresholds = load_thresholds()

            # Swap all references atomically while holding the lock
            self._forecast_model = forecast
            self._failure_model = failure
            self._feature_columns = feature_columns
            self._thresholds = thresholds
            self._loaded = True
            logger.info("scoring_models_loaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def score(self, request: ScoreWindowRequest) -> dict:
        if not self._loaded:
            self.load_models()
        if not self._loaded:
            raise ModelNotLoadedError("Models not loaded. Run training pipeline first.")

        settings = get_model_settings()

        # Snapshot references under lock to avoid mid-swap reads
        with self._lock:
            forecast_model = self._forecast_model
            failure_model = self._failure_model
            feature_columns = self._feature_columns
            thresholds = self._thresholds

        rows = [reading.model_dump() for reading in request.window]
        window_df = pd.DataFrame(rows)
        window_df["engine_id"] = request.engine_id

        featured_df, _ = build_features_for_window(window_df)

        if featured_df.empty:
            raise ValueError("Insufficient data after feature engineering.")

        all_features = featured_df[feature_columns]
        all_predictions = forecast_model.predict(all_features)

        last_row = featured_df.iloc[[-1]]
        feature_values = last_row[feature_columns]

        forecast_rul = float(max(0.0, all_predictions[-1]))
        failure_prob = float(failure_model.predict_proba(feature_values)[0])

        anomaly_score = score_window_volatility(
            predictions=all_predictions,
            thresholds=thresholds,
        )

        anomaly_level = compute_anomaly_level(
            anomaly_score,
            thresholds.get("warning", settings.anomaly.score_threshold_warning),
            thresholds.get("critical", settings.anomaly.score_threshold_critical),
        )

        alert_result = compute_alert(
            engine_id=request.engine_id,
            anomaly_score=anomaly_score,
            anomaly_level=anomaly_level,
            failure_probability=failure_prob,
            forecast_rul=forecast_rul,
        )

        return {
            "engine_id": request.engine_id,
            "forecast": round(forecast_rul, 2),
            "anomaly_score": round(anomaly_score, 4),
            "anomaly_level": anomaly_level,
            "failure_probability_next_30_cycles": round(failure_prob, 4),
            "alert_level": alert_result.alert_level,
            "recommended_action": alert_result.recommended_action,
        }
