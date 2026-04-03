import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps.api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _make_window(rng: np.random.RandomState, n_cycles: int = 30) -> list[dict]:
    window = []
    for cycle in range(1, n_cycles + 1):
        reading = {
            "cycle": cycle,
            "op1": float(rng.normal(0, 0.01)),
            "op2": float(rng.normal(0, 0.01)),
            "op3": 100.0,
        }
        for i in range(1, 22):
            reading[f"s{i}"] = float(rng.normal(500, 10))
        window.append(reading)
    return window


# ---- Health ----


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data


# ---- Alerts ----


def test_alerts_endpoint(client: TestClient) -> None:
    response = client.get("/v1/alerts")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


# ---- Metrics ----


def test_metrics_endpoint(client: TestClient) -> None:
    response = client.get("/v1/metrics")
    assert response.status_code == 200
    assert b"api_request_total" in response.content


# ---- Score Window ----


def test_score_window_without_models(client: TestClient) -> None:
    rng = np.random.RandomState(42)
    window = _make_window(rng)
    response = client.post(
        "/v1/score-window",
        json={"engine_id": 1, "window": window},
    )
    # 200 if pre-trained models exist, 503 if not
    assert response.status_code in (200, 503)


def test_score_window_validation_negative_engine(client: TestClient) -> None:
    response = client.post(
        "/v1/score-window",
        json={"engine_id": -1, "window": []},
    )
    assert response.status_code == 422


def test_score_window_validation_empty_window(client: TestClient) -> None:
    response = client.post(
        "/v1/score-window",
        json={"engine_id": 1, "window": []},
    )
    assert response.status_code == 422


def test_score_window_validation_missing_sensor(client: TestClient) -> None:
    """A reading missing a required sensor field must be rejected."""
    reading = {"cycle": 1, "op1": 0.0, "op2": 0.0, "op3": 100.0}
    # Only include s1-s20, missing s21
    for i in range(1, 21):
        reading[f"s{i}"] = 500.0
    response = client.post(
        "/v1/score-window",
        json={"engine_id": 1, "window": [reading]},
    )
    assert response.status_code == 422


def test_score_window_validation_too_many_readings(client: TestClient) -> None:
    """Window exceeding max_length=50 must be rejected."""
    rng = np.random.RandomState(99)
    window = _make_window(rng, n_cycles=51)
    response = client.post(
        "/v1/score-window",
        json={"engine_id": 1, "window": window},
    )
    assert response.status_code == 422


# ---- Generate Sample ----


def test_generate_sample_healthy(client: TestClient) -> None:
    response = client.post(
        "/v1/generate-sample",
        json={"engine_id": 1, "degradation": "healthy"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["engine_id"] == 1
    assert data["degradation"] == "healthy"
    assert len(data["window"]) == 30


def test_generate_sample_critical(client: TestClient) -> None:
    response = client.post(
        "/v1/generate-sample",
        json={"engine_id": 5, "degradation": "critical"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["degradation"] == "critical"
    assert all("s1" in r and "s21" in r for r in data["window"])


def test_generate_sample_invalid_degradation(client: TestClient) -> None:
    response = client.post(
        "/v1/generate-sample",
        json={"engine_id": 1, "degradation": "unknown"},
    )
    assert response.status_code == 422


# ---- Train ----


def test_train_invalid_model_name(client: TestClient) -> None:
    response = client.post(
        "/v1/train",
        json={"models": ["nonexistent"]},
    )
    assert response.status_code == 422


def test_train_empty_models_list(client: TestClient) -> None:
    response = client.post(
        "/v1/train",
        json={"models": []},
    )
    assert response.status_code == 422


# ---- Dashboard ----


def test_dashboard_returns_html(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert b"<html" in response.content.lower() or b"<!doctype" in response.content.lower()
