import threading
import time

from fastapi import APIRouter, HTTPException, Response
from loguru import logger

from apps.api.schemas import (
    AlertResponse,
    GenerateSampleRequest,
    GenerateSampleResponse,
    HealthResponse,
    ScoreWindowRequest,
    ScoreWindowResponse,
    TrainRequest,
    TrainResponse,
)
from apps.api.scoring import ModelNotLoadedError, ScoringService
from apps.api.synthetic import generate_sensor_window
from monitoring.metrics import (
    ALERT_COUNT,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    SCORING_COUNT,
    get_metrics,
    update_rates,
)
from storage.sqlite_store import get_alerts

router = APIRouter(prefix="/v1")
scoring_service = ScoringService()

_training_lock = threading.Lock()
_training_in_progress = False


@router.post("/score-window", response_model=ScoreWindowResponse)
def score_window(request: ScoreWindowRequest) -> dict:
    start_time = time.time()

    try:
        result = scoring_service.score(request)
    except ModelNotLoadedError as exc:
        REQUEST_COUNT.labels(method="POST", endpoint="/v1/score-window", status="503").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        REQUEST_COUNT.labels(method="POST", endpoint="/v1/score-window", status="422").inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    latency = time.time() - start_time

    REQUEST_LATENCY.labels(method="POST", endpoint="/v1/score-window").observe(latency)
    REQUEST_COUNT.labels(method="POST", endpoint="/v1/score-window", status="200").inc()
    SCORING_COUNT.inc()

    ALERT_COUNT.labels(level=result["alert_level"]).inc()
    update_rates(
        is_anomaly=result.get("anomaly_level") == "high",
        is_failure=result.get("failure_probability_next_30_cycles", 0) > 0.5,
    )

    logger.info(
        "score_window_complete",
        engine_id=request.engine_id,
        latency_ms=round(latency * 1000, 2),
    )
    return result


@router.post("/generate-sample", response_model=GenerateSampleResponse)
def generate_sample(request: GenerateSampleRequest) -> dict:
    window = generate_sensor_window(request.engine_id, request.degradation)
    logger.info(
        "generated_sample",
        engine_id=request.engine_id,
        degradation=request.degradation,
    )
    return {
        "engine_id": request.engine_id,
        "degradation": request.degradation,
        "window": window,
    }


@router.post("/train", response_model=TrainResponse)
def trigger_training(request: TrainRequest | None = None) -> dict:
    global _training_in_progress

    if request is None:
        request = TrainRequest()

    with _training_lock:
        if _training_in_progress:
            REQUEST_COUNT.labels(method="POST", endpoint="/v1/train", status="409").inc()
            raise HTTPException(
                status_code=409,
                detail="Training is already in progress.",
            )
        _training_in_progress = True

    def run_pipeline() -> None:
        global _training_in_progress
        try:
            from pipelines.direct_runner import run_training

            run_training(
                run_forecast="forecast" in request.models,
                run_failure="failure" in request.models,
                skip_mlflow=request.skip_mlflow,
            )
            scoring_service.load_models()
            logger.info("training_completed_successfully")
        except Exception:
            logger.exception("training_pipeline_failed")
        finally:
            with _training_lock:
                _training_in_progress = False

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    REQUEST_COUNT.labels(method="POST", endpoint="/v1/train", status="202").inc()
    logger.info("training_triggered", models=request.models)
    return {"status": "accepted", "message": "Training pipeline started in background."}


@router.get("/alerts", response_model=list[AlertResponse])
def list_alerts() -> list[dict]:
    start_time = time.time()
    alerts = get_alerts(limit=100)
    latency = time.time() - start_time

    REQUEST_LATENCY.labels(method="GET", endpoint="/v1/alerts").observe(latency)
    REQUEST_COUNT.labels(method="GET", endpoint="/v1/alerts", status="200").inc()

    return alerts


@router.get("/health", response_model=HealthResponse)
def health_check() -> dict:
    REQUEST_COUNT.labels(method="GET", endpoint="/v1/health", status="200").inc()
    return {
        "status": "healthy",
        "models_loaded": scoring_service.is_loaded,
    }


@router.get("/metrics")
def metrics() -> Response:
    return Response(content=get_metrics(), media_type="text/plain")
