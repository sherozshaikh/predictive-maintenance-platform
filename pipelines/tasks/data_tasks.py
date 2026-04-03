import numpy as np
import pandas as pd
from loguru import logger
from prefect import task

from configs.settings import get_pipeline_settings
from features.engineering import build_features
from features.ingestion import load_train_data
from features.validation import validate_raw
from storage.duckdb_store import write_dataframe


@task(name="ingest_data")
def ingest() -> pd.DataFrame:
    train_df = load_train_data()
    logger.info("ingestion_complete", train_rows=len(train_df))
    return train_df


@task(name="validate_data")
def validate(train_df: pd.DataFrame) -> pd.DataFrame:
    train_validated = validate_raw(train_df)
    logger.info("validation_complete", rows=len(train_validated))
    return train_validated


@task(name="feature_engineering")
def engineer_features(
    train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    featured_df, feature_columns = build_features(train_df)
    write_dataframe("train_features", featured_df)
    logger.info("feature_engineering_complete", features=len(feature_columns))
    return featured_df, feature_columns


@task(name="split_time_series")
def split_time_series(
    df: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str]]:
    settings = get_pipeline_settings()
    rng = np.random.RandomState(settings.split.random_state)

    engine_ids = df["engine_id"].unique()
    n_test = max(1, int(len(engine_ids) * settings.split.test_engine_fraction))
    test_engines = rng.choice(engine_ids, size=n_test, replace=False)

    train_mask = ~df["engine_id"].isin(test_engines)
    val_mask = df["engine_id"].isin(test_engines)

    x_train = df.loc[train_mask, feature_columns]
    y_train = df.loc[train_mask, "rul"]
    x_val = df.loc[val_mask, feature_columns]
    y_val = df.loc[val_mask, "rul"]

    write_dataframe("train_split", df.loc[train_mask])
    write_dataframe("val_split", df.loc[val_mask])

    logger.info(
        "split_complete",
        train_size=len(x_train),
        val_size=len(x_val),
        train_engines=len(engine_ids) - n_test,
        val_engines=n_test,
    )
    return x_train, y_train, x_val, y_val, feature_columns
