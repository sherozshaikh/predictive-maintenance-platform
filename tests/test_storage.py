import pandas as pd
import pytest

from storage.duckdb_store import (
    _validate_table_name,
    read_dataframe,
    table_exists,
    write_dataframe,
)
from storage.sqlite_store import get_alerts, init_tables, insert_alert

# ---- DuckDB table name validation ----


def test_valid_table_names() -> None:
    assert _validate_table_name("train_features") == "train_features"
    assert _validate_table_name("MyTable") == "MyTable"
    assert _validate_table_name("_private") == "_private"


def test_invalid_table_names() -> None:
    with pytest.raises(ValueError):
        _validate_table_name("DROP TABLE--")
    with pytest.raises(ValueError):
        _validate_table_name("table; DELETE")
    with pytest.raises(ValueError):
        _validate_table_name("")
    with pytest.raises(ValueError):
        _validate_table_name("123start")


# ---- DuckDB write/read round-trip ----


def test_duckdb_write_and_read() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    write_dataframe("test_roundtrip", df)
    result = read_dataframe("test_roundtrip")
    assert len(result) == 3
    assert list(result.columns) == ["a", "b"]


def test_duckdb_table_exists() -> None:
    df = pd.DataFrame({"x": [1]})
    write_dataframe("test_exists_check", df)
    assert table_exists("test_exists_check") is True
    assert table_exists("nonexistent_table_xyz") is False


# ---- SQLite alerts ----


def test_sqlite_init_tables() -> None:
    init_tables()  # should not raise


def test_sqlite_insert_and_get_alerts() -> None:
    init_tables()
    insert_alert(
        engine_id=42,
        anomaly_score=3.5,
        failure_probability=0.85,
        alert_level="CRITICAL",
        recommended_action="Test action",
    )
    alerts = get_alerts(limit=10)
    assert len(alerts) > 0
    latest = alerts[0]
    assert latest["engine_id"] == 42
    assert latest["alert_level"] == "CRITICAL"
    assert "timestamp" in latest
