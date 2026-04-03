from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from configs.settings import get_pipeline_settings


def load_train_data() -> pd.DataFrame:
    settings = get_pipeline_settings()
    columns = (
        ["engine_id", "cycle"] + settings.data.operational_columns + settings.data.sensor_columns
    )
    path = Path(settings.data.raw_dir) / settings.data.train_file
    df = pd.read_csv(path, sep=r"\s+", header=None, names=columns)
    df = _add_rul(df, settings.features.rul_cap)
    logger.info("loaded_train_data", rows=len(df), engines=df["engine_id"].nunique())
    return df


def load_test_data() -> tuple[pd.DataFrame, np.ndarray]:
    settings = get_pipeline_settings()
    columns = (
        ["engine_id", "cycle"] + settings.data.operational_columns + settings.data.sensor_columns
    )
    path = Path(settings.data.raw_dir) / settings.data.test_file
    df = pd.read_csv(path, sep=r"\s+", header=None, names=columns)

    rul_path = Path(settings.data.raw_dir) / settings.data.rul_file
    rul_values = pd.read_csv(rul_path, header=None, names=["rul"]).values.flatten()
    logger.info("loaded_test_data", rows=len(df), engines=df["engine_id"].nunique())
    return df, rul_values


def _add_rul(df: pd.DataFrame, rul_cap: int) -> pd.DataFrame:
    max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]
    df = df.merge(max_cycles, on="engine_id")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df["rul"] = df["rul"].clip(upper=rul_cap)
    df = df.drop(columns=["max_cycle"])
    return df
