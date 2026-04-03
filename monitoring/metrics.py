import threading

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry()

REQUEST_COUNT = Counter(
    "api_request_total",
    "Total API requests",
    ["method", "endpoint", "status"],
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["method", "endpoint"],
    registry=REGISTRY,
)

ANOMALY_RATE = Gauge(
    "anomaly_rate",
    "Current anomaly rate across scored engines",
    registry=REGISTRY,
)

FAILURE_RATE = Gauge(
    "failure_rate",
    "Current predicted failure rate across scored engines",
    registry=REGISTRY,
)

SCORING_COUNT = Counter(
    "scoring_request_total",
    "Total scoring requests",
    registry=REGISTRY,
)

ALERT_COUNT = Counter(
    "alert_total",
    "Total alerts generated",
    ["level"],
    registry=REGISTRY,
)

# Running counters for computing rates (not exported as metrics)
_rates_lock = threading.Lock()
_scoring_total: int = 0
_anomaly_total: int = 0
_failure_total: int = 0


def update_rates(is_anomaly: bool, is_failure: bool) -> None:
    """Update anomaly/failure rate gauges based on running totals."""
    global _scoring_total, _anomaly_total, _failure_total
    with _rates_lock:
        _scoring_total += 1
        if is_anomaly:
            _anomaly_total += 1
        if is_failure:
            _failure_total += 1
        ANOMALY_RATE.set(_anomaly_total / _scoring_total)
        FAILURE_RATE.set(_failure_total / _scoring_total)


def get_metrics() -> bytes:
    return generate_latest(REGISTRY)
