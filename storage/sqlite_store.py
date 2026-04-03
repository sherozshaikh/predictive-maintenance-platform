import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from configs.settings import get_infra_settings

_CREATE_ALERTS = """
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        engine_id INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        anomaly_score REAL NOT NULL,
        failure_probability REAL NOT NULL,
        alert_level TEXT NOT NULL,
        recommended_action TEXT NOT NULL
    )
"""


def _get_connection() -> sqlite3.Connection:
    settings = get_infra_settings()
    db_path = Path(settings.sqlite.path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_ALERTS)
    conn.commit()
    return conn


def init_tables() -> None:
    conn = _get_connection()
    conn.close()
    logger.info("initialized_sqlite_tables")


def insert_alert(
    engine_id: int,
    anomaly_score: float,
    failure_probability: float,
    alert_level: str,
    recommended_action: str,
) -> None:
    conn = _get_connection()
    conn.execute(
        """
        INSERT INTO alerts
        (engine_id, timestamp, anomaly_score, failure_probability, alert_level, recommended_action)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            engine_id,
            datetime.now(UTC).isoformat(),
            anomaly_score,
            failure_probability,
            alert_level,
            recommended_action,
        ),
    )
    conn.commit()
    conn.close()
    logger.info("inserted_alert", engine_id=engine_id, alert_level=alert_level)


def get_alerts(limit: int = 100) -> list[dict]:
    conn = _get_connection()
    rows = conn.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]
