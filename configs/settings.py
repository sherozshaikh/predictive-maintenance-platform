from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

CONFIG_DIR = Path(__file__).parent


class ForecastModelConfig(BaseModel):
    name: str
    type: str
    params: dict


class FailureModelConfig(BaseModel):
    name: str
    type: str
    params: dict


class AnomalyConfig(BaseModel):
    rolling_std_window: int
    score_threshold_warning: float
    score_threshold_critical: float


class FailureThresholdConfig(BaseModel):
    rul_cutoff: int


class ModelSettings(BaseModel):
    forecast: ForecastModelConfig
    failure: FailureModelConfig
    anomaly: AnomalyConfig
    failure_threshold: FailureThresholdConfig


class DataConfig(BaseModel):
    raw_dir: str
    train_file: str
    test_file: str
    rul_file: str
    sensor_columns: list[str]
    operational_columns: list[str]
    selected_sensors: list[str]


class FeaturesConfig(BaseModel):
    lag_periods: list[int]
    rolling_windows: list[int]
    rul_cap: int


class SplitConfig(BaseModel):
    test_engine_fraction: float
    random_state: int


class TrainingConfig(BaseModel):
    seed: int
    mlflow_experiment: str


class PipelineSettings(BaseModel):
    data: DataConfig
    features: FeaturesConfig
    split: SplitConfig
    training: TrainingConfig


class ApiConfig(BaseModel):
    host: str
    port: int
    workers: int


class MlflowConfig(BaseModel):
    tracking_uri: str
    artifact_location: str


class MinioConfig(BaseModel):
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    secure: bool


class DuckdbConfig(BaseModel):
    path: str


class SqliteConfig(BaseModel):
    path: str


class PrometheusConfig(BaseModel):
    port: int


class PrefectConfig(BaseModel):
    api_url: str


class InfraSettings(BaseModel):
    api: ApiConfig
    mlflow: MlflowConfig
    minio: MinioConfig
    duckdb: DuckdbConfig
    sqlite: SqliteConfig
    prometheus: PrometheusConfig
    prefect: PrefectConfig


def _load_yaml(filename: str) -> dict:
    with open(CONFIG_DIR / filename) as f:
        return yaml.safe_load(f)


@lru_cache
def get_model_settings() -> ModelSettings:
    return ModelSettings(**_load_yaml("model.yaml"))


@lru_cache
def get_pipeline_settings() -> PipelineSettings:
    return PipelineSettings(**_load_yaml("pipeline.yaml"))


@lru_cache
def get_infra_settings() -> InfraSettings:
    return InfraSettings(**_load_yaml("infra.yaml"))
