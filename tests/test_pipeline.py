import numpy as np

from configs.settings import get_pipeline_settings
from features.engineering import build_features
from features.ingestion import load_train_data
from features.validation import validate_raw


def test_ingestion_loads_data() -> None:
    df = load_train_data()
    assert len(df) > 0
    assert "engine_id" in df.columns
    assert "cycle" in df.columns
    assert "rul" in df.columns


def test_validation_passes_on_valid_data() -> None:
    df = load_train_data()
    validated = validate_raw(df)
    assert len(validated) == len(df)


def test_feature_engineering_produces_valid_output() -> None:
    df = load_train_data()
    featured_df, feature_columns = build_features(df)
    assert len(featured_df) > 0
    assert len(feature_columns) > 10
    assert featured_df[feature_columns].isnull().sum().sum() == 0


def test_split_produces_disjoint_sets() -> None:
    settings = get_pipeline_settings()
    df = load_train_data()
    featured_df, feature_columns = build_features(df)

    rng = np.random.RandomState(settings.split.random_state)
    engine_ids = featured_df["engine_id"].unique()
    n_test = max(1, int(len(engine_ids) * settings.split.test_engine_fraction))
    test_engines = rng.choice(engine_ids, size=n_test, replace=False)
    train_engines = np.setdiff1d(engine_ids, test_engines)

    assert len(np.intersect1d(train_engines, test_engines)) == 0
    assert len(train_engines) + len(test_engines) == len(engine_ids)
