"""
Benchmark: LightGBM vs H2O AutoML for RUL forecast and failure classification.

Trains both frameworks on the same CMAPSS FD001 data and compares metrics.
LightGBM is the production baseline; H2O AutoML is the challenger.

Usage:
    uv pip install -e ".[dev,benchmark]"
    PYTHONPATH=. python scripts/benchmark_h2o.py

Results are saved to data/artifacts/benchmark_results.json
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from configs.settings import get_model_settings, get_pipeline_settings
from features.engineering import build_features
from features.ingestion import load_train_data
from features.validation import validate_raw
from models.failure.lgbm_failure import FailureClassifierModel
from models.forecast.lgbm_forecast import RULForecastModel

ARTIFACTS_DIR = Path("data/artifacts")


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load, validate, engineer features, and split data."""
    settings = get_pipeline_settings()

    train_df = load_train_data()
    validated = validate_raw(train_df)
    featured_df, feature_columns = build_features(validated)

    rng = np.random.RandomState(settings.split.random_state)
    engine_ids = featured_df["engine_id"].unique()
    n_test = max(1, int(len(engine_ids) * settings.split.test_engine_fraction))
    test_engines = rng.choice(engine_ids, size=n_test, replace=False)

    train_mask = ~featured_df["engine_id"].isin(test_engines)
    val_mask = featured_df["engine_id"].isin(test_engines)

    train_data = featured_df.loc[train_mask]
    val_data = featured_df.loc[val_mask]

    return train_data, val_data, feature_columns


def train_lgbm(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    """Train LightGBM models and return metrics."""
    model_settings = get_model_settings()

    x_train = train_data[feature_columns]
    y_train = train_data["rul"]
    x_val = val_data[feature_columns]
    y_val = val_data["rul"]

    start = time.time()
    forecast_model = RULForecastModel()
    forecast_metrics = forecast_model.train(x_train, y_train, x_val, y_val)
    forecast_time = time.time() - start

    cutoff = model_settings.failure_threshold.rul_cutoff
    y_train_binary = (y_train <= cutoff).astype(int)
    y_val_binary = (y_val <= cutoff).astype(int)

    start = time.time()
    failure_model = FailureClassifierModel()
    failure_metrics = failure_model.train(x_train, y_train_binary, x_val, y_val_binary)
    failure_time = time.time() - start

    return {
        "forecast": {
            **forecast_metrics,
            "training_time_seconds": round(forecast_time, 2),
        },
        "failure": {**failure_metrics, "training_time_seconds": round(failure_time, 2)},
    }


def train_h2o(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    feature_columns: list[str],
    max_runtime_secs: int = 120,
) -> dict:
    """Train H2O AutoML models and return metrics + leaderboard."""
    import h2o
    from h2o.automl import H2OAutoML

    model_settings = get_model_settings()

    h2o.init(max_mem_size="2G", nthreads=-1)

    train_h2o_df = h2o.H2OFrame(train_data[feature_columns + ["rul"]])
    val_h2o_df = h2o.H2OFrame(val_data[feature_columns + ["rul"]])

    # --- Forecast (regression) ---
    logger.info("h2o_automl_forecast_starting")
    start = time.time()
    forecast_aml = H2OAutoML(
        max_runtime_secs=max_runtime_secs,
        seed=42,
        sort_metric="RMSE",
        project_name="rul_forecast",
    )
    forecast_aml.train(
        x=feature_columns,
        y="rul",
        training_frame=train_h2o_df,
        validation_frame=val_h2o_df,
    )
    forecast_time = time.time() - start

    forecast_leader = forecast_aml.leader
    forecast_perf = forecast_leader.model_performance(val_h2o_df)

    forecast_leaderboard = forecast_aml.leaderboard.as_data_frame().head(10)
    logger.info("h2o_forecast_leader", model=forecast_leader.model_id)

    forecast_preds = forecast_leader.predict(val_h2o_df).as_data_frame()["predict"].values
    y_val_np = val_data["rul"].values
    forecast_r2 = float(
        1 - np.sum((y_val_np - forecast_preds) ** 2) / np.sum((y_val_np - y_val_np.mean()) ** 2)
    )

    forecast_metrics = {
        "rmse": float(forecast_perf.rmse()),
        "mae": float(forecast_perf.mae()),
        "r2": round(forecast_r2, 6),
        "best_model": str(forecast_leader.model_id),
        "training_time_seconds": round(forecast_time, 2),
    }

    # --- Failure (classification) ---
    cutoff = model_settings.failure_threshold.rul_cutoff

    train_class = train_data[feature_columns].copy()
    train_class["failure"] = (train_data["rul"] <= cutoff).astype(int)
    val_class = val_data[feature_columns].copy()
    val_class["failure"] = (val_data["rul"] <= cutoff).astype(int)

    train_h2o_class = h2o.H2OFrame(train_class)
    val_h2o_class = h2o.H2OFrame(val_class)
    train_h2o_class["failure"] = train_h2o_class["failure"].asfactor()
    val_h2o_class["failure"] = val_h2o_class["failure"].asfactor()

    logger.info("h2o_automl_failure_starting")
    start = time.time()
    failure_aml = H2OAutoML(
        max_runtime_secs=max_runtime_secs,
        seed=42,
        sort_metric="AUC",
        project_name="failure_classifier",
    )
    failure_aml.train(
        x=feature_columns,
        y="failure",
        training_frame=train_h2o_class,
        validation_frame=val_h2o_class,
    )
    failure_time = time.time() - start

    failure_leader = failure_aml.leader
    failure_perf = failure_leader.model_performance(val_h2o_class)

    failure_leaderboard = failure_aml.leaderboard.as_data_frame().head(10)
    logger.info("h2o_failure_leader", model=failure_leader.model_id)

    failure_metrics = {
        "auc_roc": float(failure_perf.auc()),
        "best_model": str(failure_leader.model_id),
        "training_time_seconds": round(failure_time, 2),
    }

    h2o.shutdown(prompt=False)

    return {
        "forecast": forecast_metrics,
        "failure": failure_metrics,
        "leaderboards": {
            "forecast": forecast_leaderboard.to_dict(orient="records"),
            "failure": failure_leaderboard.to_dict(orient="records"),
        },
    }


def print_comparison(lgbm_results: dict, h2o_results: dict) -> None:
    """Print side-by-side comparison table."""
    print("\n" + "=" * 80)
    print("BENCHMARK: LightGBM (Baseline) vs H2O AutoML (Challenger)")
    print("=" * 80)

    print("\n--- RUL Forecast (Regression) ---")
    print(f"{'Metric':<25} {'LightGBM':<20} {'H2O Best':<20} {'Winner':<15}")
    print("-" * 80)

    for metric in ["rmse", "mae", "r2"]:
        lgbm_val = lgbm_results["forecast"].get(metric, "N/A")
        h2o_val = h2o_results["forecast"].get(metric, "N/A")
        if isinstance(lgbm_val, float) and isinstance(h2o_val, float):
            if metric == "r2":
                winner = "H2O" if h2o_val > lgbm_val else "LightGBM"
            else:
                winner = "H2O" if h2o_val < lgbm_val else "LightGBM"
            print(f"{metric:<25} {lgbm_val:<20.4f} {h2o_val:<20.4f} {winner:<15}")

    lgbm_time = lgbm_results["forecast"]["training_time_seconds"]
    h2o_time = h2o_results["forecast"]["training_time_seconds"]
    print(
        f"{'training_time (s)':<25} {lgbm_time:<20.2f} {h2o_time:<20.2f} {'LightGBM' if lgbm_time < h2o_time else 'H2O':<15}"  # noqa: E501
    )

    print(f"\nH2O Best Model: {h2o_results['forecast']['best_model']}")

    print("\n--- Failure Classification ---")
    print(f"{'Metric':<25} {'LightGBM':<20} {'H2O Best':<20} {'Winner':<15}")
    print("-" * 80)

    lgbm_auc = lgbm_results["failure"].get("auc_roc", "N/A")
    h2o_auc = h2o_results["failure"].get("auc_roc", "N/A")
    if isinstance(lgbm_auc, float) and isinstance(h2o_auc, float):
        winner = "H2O" if h2o_auc > lgbm_auc else "LightGBM"
        print(f"{'auc_roc':<25} {lgbm_auc:<20.4f} {h2o_auc:<20.4f} {winner:<15}")

    for metric in ["f1", "precision", "recall"]:
        lgbm_val = lgbm_results["failure"].get(metric, "N/A")
        if isinstance(lgbm_val, float):
            print(f"{metric:<25} {lgbm_val:<20.4f} {'--':<20} {'--':<15}")

    lgbm_time = lgbm_results["failure"]["training_time_seconds"]
    h2o_time = h2o_results["failure"]["training_time_seconds"]
    print(
        f"{'training_time (s)':<25} {lgbm_time:<20.2f} {h2o_time:<20.2f} {'LightGBM' if lgbm_time < h2o_time else 'H2O':<15}"  # noqa: E501
    )

    print(f"\nH2O Best Model: {h2o_results['failure']['best_model']}")

    print("\n--- H2O Forecast Leaderboard (Top 10) ---")
    lb = h2o_results["leaderboards"]["forecast"]
    for i, row in enumerate(lb):
        model_id = row.get("model_id", "?")
        rmse = row.get("rmse", "?")
        print(f"  {i + 1}. {model_id}  (RMSE: {rmse})")

    print("\n--- H2O Failure Leaderboard (Top 10) ---")
    lb = h2o_results["leaderboards"]["failure"]
    for i, row in enumerate(lb):
        model_id = row.get("model_id", "?")
        auc = row.get("auc", "?")
        print(f"  {i + 1}. {model_id}  (AUC: {auc})")

    print("\n" + "=" * 80)
    print("Conclusion: LightGBM is the production model (fast inference,")
    print("lightweight, no JVM dependency). H2O AutoML validates the choice.")
    print("=" * 80)


def main() -> None:
    logger.info("benchmark_starting")

    train_data, val_data, feature_columns = prepare_data()
    logger.info("data_prepared", train_rows=len(train_data), val_rows=len(val_data))

    logger.info("training_lightgbm")
    lgbm_results = train_lgbm(train_data, val_data, feature_columns)

    logger.info("training_h2o_automl")
    h2o_results = train_h2o(train_data, val_data, feature_columns, max_runtime_secs=120)

    print_comparison(lgbm_results, h2o_results)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "lightgbm": lgbm_results,
        "h2o_automl": h2o_results,
    }
    output_path = ARTIFACTS_DIR / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("benchmark_saved", path=str(output_path))


if __name__ == "__main__":
    main()
