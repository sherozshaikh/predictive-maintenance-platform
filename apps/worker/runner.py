import os
import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, serialize=True)

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")


def run_training() -> None:
    api_url = os.environ.get("PREFECT_API_URL", "")

    mlflow_available = bool(os.environ.get("MLFLOW_TRACKING_URI", ""))

    if api_url:
        from pipelines.flows.training_flow import training_pipeline

        logger.info("worker_starting_prefect_flow", api_url=api_url, mlflow=mlflow_available)
        result = training_pipeline(
            run_forecast=True,
            run_failure=True,
            skip_mlflow=not mlflow_available,
        )
    else:
        from pipelines.direct_runner import run_training as _run

        logger.info("worker_starting_direct_runner", mlflow=mlflow_available)
        result = _run(
            run_forecast=True,
            run_failure=True,
            skip_mlflow=not mlflow_available,
        )

    logger.info("worker_training_complete", metrics=result.get("metrics"))


if __name__ == "__main__":
    run_training()
