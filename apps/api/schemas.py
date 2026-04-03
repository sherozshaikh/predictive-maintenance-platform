from typing import Literal

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    cycle: int = Field(gt=0)
    op1: float
    op2: float
    op3: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float
    s7: float
    s8: float
    s9: float
    s10: float
    s11: float
    s12: float
    s13: float
    s14: float
    s15: float
    s16: float
    s17: float
    s18: float
    s19: float
    s20: float
    s21: float


class ScoreWindowRequest(BaseModel):
    engine_id: int = Field(gt=0)
    window: list[SensorReading] = Field(min_length=1, max_length=50)


class ScoreWindowResponse(BaseModel):
    engine_id: int
    forecast: float
    anomaly_score: float
    anomaly_level: str
    failure_probability_next_30_cycles: float
    alert_level: str
    recommended_action: str


class TrainRequest(BaseModel):
    models: list[Literal["forecast", "failure"]] = Field(
        default=["forecast", "failure"],
        min_length=1,
    )
    skip_mlflow: bool = True


class TrainResponse(BaseModel):
    status: str
    message: str


class AlertResponse(BaseModel):
    id: int
    engine_id: int
    timestamp: str
    anomaly_score: float
    failure_probability: float
    alert_level: str
    recommended_action: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


class GenerateSampleRequest(BaseModel):
    engine_id: int = Field(default=1, gt=0)
    degradation: str = Field(default="mid", pattern="^(healthy|mid|critical)$")


class GenerateSampleResponse(BaseModel):
    engine_id: int
    degradation: str
    window: list[SensorReading]
