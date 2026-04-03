import pandas as pd
from loguru import logger
from pandera.pandas import Check, Column, DataFrameSchema

from configs.settings import get_pipeline_settings


def get_raw_schema() -> DataFrameSchema:
    settings = get_pipeline_settings()
    columns = {
        "engine_id": Column(int, Check.gt(0)),
        "cycle": Column(int, Check.gt(0)),
    }
    for col in settings.data.operational_columns:
        columns[col] = Column(float, nullable=False)
    for col in settings.data.sensor_columns:
        columns[col] = Column(float, nullable=False)
    return DataFrameSchema(columns, coerce=True)


def get_features_schema(feature_columns: list[str]) -> DataFrameSchema:
    columns = {
        "engine_id": Column(int, Check.gt(0)),
        "cycle": Column(int, Check.gt(0)),
        "rul": Column(float, Check.ge(0)),
    }
    for col in feature_columns:
        columns[col] = Column(float, nullable=True)
    return DataFrameSchema(columns, coerce=True)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    schema = get_raw_schema()
    validated = schema.validate(df)
    logger.info("validated_raw_data", rows=len(validated))
    return validated


def validate_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    schema = get_features_schema(feature_columns)
    validated = schema.validate(df)
    logger.info("validated_feature_data", rows=len(validated))
    return validated
