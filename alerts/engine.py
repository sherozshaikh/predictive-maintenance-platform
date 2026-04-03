from loguru import logger
from pydantic import BaseModel

from storage.sqlite_store import insert_alert


class AlertResult(BaseModel):
    anomaly_score: float
    anomaly_level: str
    failure_probability: float
    alert_level: str
    recommended_action: str


def compute_anomaly_level(
    anomaly_score: float, warning_threshold: float, critical_threshold: float
) -> str:
    if anomaly_score >= critical_threshold:
        return "high"
    if anomaly_score >= warning_threshold:
        return "medium"
    return "low"


def compute_alert(
    engine_id: int,
    anomaly_score: float,
    anomaly_level: str,
    failure_probability: float,
    forecast_rul: float = 125.0,
) -> AlertResult:
    """Determine alert level using RUL forecast, failure probability, and anomaly.

    Decision matrix (evaluated top-to-bottom, first match wins):
      CRITICAL : RUL < 8   AND  failure_prob > 0.7
      CRITICAL : anomaly == high  AND  failure_prob > 0.7   (sensor anomaly)
      WARNING  : RUL < 15  AND  failure_prob > 0.5
      WARNING  : anomaly == medium or high
      NORMAL   : everything else
    """
    if (forecast_rul < 8 and failure_probability > 0.7) or (
        anomaly_level == "high" and failure_probability > 0.7
    ):
        alert_level = "CRITICAL"
        recommended_action = "Schedule emergency maintenance within 24 hours."
    elif (forecast_rul < 15 and failure_probability > 0.5) or anomaly_level in (
        "medium",
        "high",
    ):
        alert_level = "WARNING"
        recommended_action = "Plan maintenance within next 7 days. Monitor closely."
    else:
        alert_level = "NORMAL"
        recommended_action = "Continue standard monitoring schedule."

    result = AlertResult(
        anomaly_score=anomaly_score,
        anomaly_level=anomaly_level,
        failure_probability=failure_probability,
        alert_level=alert_level,
        recommended_action=recommended_action,
    )

    if alert_level != "NORMAL":
        insert_alert(
            engine_id=engine_id,
            anomaly_score=anomaly_score,
            failure_probability=failure_probability,
            alert_level=alert_level,
            recommended_action=recommended_action,
        )
        logger.info(
            "alert_generated",
            engine_id=engine_id,
            alert_level=alert_level,
            failure_probability=failure_probability,
        )

    return result
