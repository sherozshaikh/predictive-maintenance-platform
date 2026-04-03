import numpy as np
import pandas as pd
import pytest

from configs.settings import get_pipeline_settings


@pytest.fixture
def sample_train_df() -> pd.DataFrame:
    settings = get_pipeline_settings()
    rng = np.random.RandomState(42)
    n_engines = 5
    cycles_per_engine = 80
    rows = []
    for engine_id in range(1, n_engines + 1):
        for cycle in range(1, cycles_per_engine + 1):
            row = {
                "engine_id": engine_id,
                "cycle": cycle,
                "op1": rng.normal(0, 0.01),
                "op2": rng.normal(0, 0.01),
                "op3": 100.0,
            }
            for s in settings.data.sensor_columns:
                row[s] = rng.normal(500, 10) + cycle * 0.1
            rows.append(row)
    df = pd.DataFrame(rows)
    max_cycles = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycles.columns = ["engine_id", "max_cycle"]
    df = df.merge(max_cycles, on="engine_id")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df["rul"] = df["rul"].clip(upper=settings.features.rul_cap)
    df = df.drop(columns=["max_cycle"])
    return df


@pytest.fixture
def sample_window_df() -> pd.DataFrame:
    settings = get_pipeline_settings()
    rng = np.random.RandomState(42)
    rows = []
    for cycle in range(1, 31):
        row = {
            "engine_id": 1,
            "cycle": cycle,
            "op1": rng.normal(0, 0.01),
            "op2": rng.normal(0, 0.01),
            "op3": 100.0,
        }
        for s in settings.data.sensor_columns:
            row[s] = rng.normal(500, 10) + cycle * 0.1
        rows.append(row)
    return pd.DataFrame(rows)
