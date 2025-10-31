"""Regression tests for database initialisation behaviour."""
from __future__ import annotations

import pytest

from database.init_db import get_db_connection, init_database
from api.core.config import get_config


@pytest.fixture(autouse=True)
def _clear_config_cache(monkeypatch):
    """Ensure configuration is reloaded for each test."""

    get_config.cache_clear()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("AI_SKILL_PLANNER_DB_PATH", raising=False)
    yield
    get_config.cache_clear()


def test_init_database_preserves_existing_data(tmp_path, monkeypatch):
    db_path = tmp_path / "existing.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", db_url)

    conn = init_database(db_url, skip_if_exists=False)
    conn.close()

    with get_db_connection(db_url) as conn:
        conn.execute("INSERT INTO skills (id, name, category) VALUES (?, ?, ?)",
                     ("skill-1", "Skill One", "category"))
        conn.commit()

    init_database(db_url, skip_if_exists=True)

    with get_db_connection(db_url) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM skills WHERE id = ?", ("skill-1",))
        assert cursor.fetchone()[0] == 1


def test_init_database_supports_memory_sqlite(monkeypatch):
    db_url = "sqlite:///:memory:"
    monkeypatch.setenv("DATABASE_URL", db_url)

    conn = init_database(db_url, skip_if_exists=False)
    try:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert "skills" in tables
    finally:
        conn.close()
