"""Tests for the ScoringService class and model guard rails."""

import pytest

from apps.api.scoring import ModelNotLoadedError, ScoringService
from models.failure.lgbm_failure import FailureClassifierModel
from models.forecast.lgbm_forecast import RULForecastModel

# ---- Model None guards ----


def test_forecast_predict_without_model_raises() -> None:
    import pandas as pd

    model = RULForecastModel()
    with pytest.raises(RuntimeError, match="not trained or loaded"):
        model.predict(pd.DataFrame({"a": [1]}))


def test_failure_predict_proba_without_model_raises() -> None:
    import pandas as pd

    model = FailureClassifierModel()
    with pytest.raises(RuntimeError, match="not trained or loaded"):
        model.predict_proba(pd.DataFrame({"a": [1]}))


# ---- ScoringService ----


def test_scoring_service_raises_when_no_artifacts() -> None:
    """ScoringService.score() should raise ModelNotLoadedError
    when artifacts don't exist (not return a dict with 'error')."""
    import tempfile

    from apps.api import scoring as scoring_module
    from apps.api.schemas import ScoreWindowRequest, SensorReading

    readings = [
        SensorReading(
            cycle=i,
            op1=0.0,
            op2=0.0,
            op3=100.0,
            s1=500,
            s2=500,
            s3=500,
            s4=500,
            s5=500,
            s6=500,
            s7=500,
            s8=500,
            s9=500,
            s10=500,
            s11=500,
            s12=500,
            s13=500,
            s14=500,
            s15=500,
            s16=500,
            s17=500,
            s18=500,
            s19=500,
            s20=500,
            s21=500,
        )
        for i in range(1, 11)
    ]
    request = ScoreWindowRequest(engine_id=1, window=readings)

    # Point ARTIFACTS_DIR to a temp dir with no model files
    original_dir = scoring_module.ARTIFACTS_DIR
    with tempfile.TemporaryDirectory() as tmpdir:
        scoring_module.ARTIFACTS_DIR = tmpdir
        try:
            service = ScoringService()
            with pytest.raises(ModelNotLoadedError):
                service.score(request)
        finally:
            scoring_module.ARTIFACTS_DIR = original_dir


def test_scoring_service_raises_on_insufficient_data() -> None:
    """A tiny window that produces empty features after dropna should raise ValueError."""
    from apps.api.schemas import ScoreWindowRequest, SensorReading

    service = ScoringService()
    # Only 1 row -- will be dropped by dropna after lag features
    readings = [
        SensorReading(
            cycle=1,
            op1=0.0,
            op2=0.0,
            op3=100.0,
            s1=500,
            s2=500,
            s3=500,
            s4=500,
            s5=500,
            s6=500,
            s7=500,
            s8=500,
            s9=500,
            s10=500,
            s11=500,
            s12=500,
            s13=500,
            s14=500,
            s15=500,
            s16=500,
            s17=500,
            s18=500,
            s19=500,
            s20=500,
            s21=500,
        )
    ]
    request = ScoreWindowRequest(engine_id=1, window=readings)

    # If models are available this should raise ValueError (insufficient data)
    # If models are NOT available this should raise ModelNotLoadedError
    with pytest.raises((ValueError, ModelNotLoadedError)):
        service.score(request)
