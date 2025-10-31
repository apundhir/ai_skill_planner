"""Database initialization and connection helpers."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse

from api.core.config import get_config
from api.core.logging import get_logger


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


logger = get_logger(__name__)


def init_database(database_url: Optional[str] = None, *, skip_if_exists: bool = True) -> sqlite3.Connection:
    """Initialise the database schema using the configured database URL."""

    config = get_config()
    target_url = _normalize_database_url(database_url, config.database_url)
    backend, location = _parse_database_url(target_url)

    if backend != "sqlite":
        raise RuntimeError("Only SQLite databases are supported by the default initialiser.")

    conn = sqlite3.connect(location)
    conn.execute("PRAGMA foreign_keys = ON")

    if skip_if_exists and _database_has_tables(conn):
        return conn

    schema_sql = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
    for statement in _split_statements(schema_sql):
        conn.execute(statement)

    conn.commit()
    return conn


def get_db_connection(database_url: Optional[str] = None) -> sqlite3.Connection:
    """Return a SQLite connection respecting configuration defaults."""

    config = get_config()
    target_url = _normalize_database_url(database_url, config.database_url)
    backend, location = _parse_database_url(target_url)

    if backend != "sqlite":
        raise RuntimeError("Only SQLite connections are available in this deployment.")

    conn = sqlite3.connect(location)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def get_db_path() -> Path:
    """Return the SQLite database path resolved from configuration."""

    config = get_config()
    target_url = _normalize_database_url(None, config.database_url)
    backend, location = _parse_database_url(target_url)

    if backend != "sqlite" or location == ":memory:":
        raise RuntimeError("Configured database does not use a filesystem path.")

    return Path(location)


def _database_has_tables(conn: sqlite3.Connection) -> bool:
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
    has_tables = cursor.fetchone() is not None
    cursor.close()
    return has_tables


def _split_statements(schema_sql: str) -> Iterable[str]:
    return [statement.strip() for statement in schema_sql.split(";") if statement.strip()]


def _normalize_database_url(candidate: Optional[str], default: str) -> str:
    if not candidate:
        return default
    if "://" in candidate:
        return candidate

    db_path = Path(candidate)
    if not db_path.is_absolute():
        db_path = (_PROJECT_ROOT / db_path).resolve()

    return f"sqlite:///{db_path.as_posix()}"


def _parse_database_url(database_url: str) -> tuple[str, str]:
    parsed = urlparse(database_url)
    backend = parsed.scheme or "sqlite"

    if backend != "sqlite":
        location = f"{parsed.netloc}{parsed.path}".lstrip("/")
        return backend, location

    if parsed.path in ("", "/") and parsed.netloc:
        path = parsed.netloc
    else:
        path = parsed.path

    if path in (":memory:", "/:memory:"):
        return "sqlite", ":memory:"

    if parsed.netloc and not parsed.path:
        path = parsed.netloc

    if not parsed.netloc:
        if path.startswith("//"):
            path = path[1:]
        elif path.startswith("/"):
            path = path[1:]

    db_path = Path(path)
    if not db_path.is_absolute():
        db_path = (_PROJECT_ROOT / db_path).resolve()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite", str(db_path)


__all__ = ["init_database", "get_db_connection", "get_db_path"]


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    connection = init_database(skip_if_exists=False)
    try:
        cursor = connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = ", ".join(sorted(row[0] for row in cursor.fetchall()))
        logger.info("database_initialised", tables=tables)
    finally:
        connection.close()
