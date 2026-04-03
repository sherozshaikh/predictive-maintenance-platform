import re
from pathlib import Path

import duckdb
import pandas as pd
from loguru import logger

from configs.settings import get_infra_settings

_SAFE_TABLE_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_table_name(table_name: str) -> str:
    if not _SAFE_TABLE_NAME.match(table_name):
        raise ValueError(f"Invalid table name: {table_name!r}")
    return table_name


def _get_connection() -> duckdb.DuckDBPyConnection:
    settings = get_infra_settings()
    db_path = Path(settings.duckdb.path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def write_dataframe(table_name: str, df: pd.DataFrame) -> None:
    safe_name = _validate_table_name(table_name)
    conn = _get_connection()
    conn.execute(f"DROP TABLE IF EXISTS {safe_name}")
    conn.execute(f"CREATE TABLE {safe_name} AS SELECT * FROM df")
    conn.close()
    logger.info("wrote_duckdb_table", table=safe_name, rows=len(df))


def read_dataframe(table_name: str) -> pd.DataFrame:
    safe_name = _validate_table_name(table_name)
    conn = _get_connection()
    df = conn.execute(f"SELECT * FROM {safe_name}").fetchdf()
    conn.close()
    logger.info("read_duckdb_table", table=safe_name, rows=len(df))
    return df


def table_exists(table_name: str) -> bool:
    conn = _get_connection()
    result = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table_name],
    ).fetchone()
    conn.close()
    return result[0] > 0
