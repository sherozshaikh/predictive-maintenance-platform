import argparse
import os
import sys

from loguru import logger

os.environ["PREFECT_API_URL"] = ""
os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")

logger.remove()
logger.add(sys.stderr, serialize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run predictive maintenance training pipeline")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["forecast", "failure", "all"],
        default=["forecast", "failure"],
        help="Models to train (default: forecast failure)",
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Skip MLflow logging (for local runs without MLflow server)",
    )
    args = parser.parse_args()

    model_list = args.models
    if "all" in model_list:
        model_list = ["forecast", "failure"]

    from pipelines.flows.training_flow import training_pipeline

    result = training_pipeline(
        run_forecast="forecast" in model_list,
        run_failure="failure" in model_list,
        skip_mlflow=args.skip_mlflow,
    )

    logger.info("pipeline_result", metrics=result["metrics"], thresholds=result["thresholds"])


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
