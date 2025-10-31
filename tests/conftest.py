"""Shared pytest fixtures for API integration tests."""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Generator

import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from api.core.config import get_config
from database.init_db import init_database, get_db_connection
from security.auth import get_authentication_system


def _seed_sample_data(conn: sqlite3.Connection) -> None:
    """Populate the test database with a minimal but coherent dataset."""

    cursor = conn.cursor()

    # Core reference data
    cursor.executemany(
        """
        INSERT INTO skills (id, name, category, decay_rate)
        VALUES (?, ?, ?, ?)
        """,
        [
            ("skill-python", "Python", "engineering", 0.1),
            ("skill-analytics", "Analytics", "data-science", 0.15),
        ],
    )

    cursor.executemany(
        """
        INSERT INTO people (id, name, location, timezone, fte, cost_hourly)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            ("person-alice", "Alice Analyst", "Remote", "UTC", 1.0, 120.0),
            ("person-bob", "Bob Builder", "Onsite", "UTC+1", 0.8, 95.0),
        ],
    )

    cursor.executemany(
        """
        INSERT INTO person_skills (
            person_id, skill_id, base_level, effective_level,
            confidence_low, confidence_high, last_used
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("person-alice", "skill-python", 4.5, 4.2, 3.8, 4.6, "2024-01-10"),
            ("person-alice", "skill-analytics", 4.0, 3.9, 3.5, 4.3, "2023-12-04"),
            ("person-bob", "skill-python", 3.0, 2.8, 2.5, 3.3, "2023-11-22"),
        ],
    )

    cursor.execute(
        """
        INSERT INTO projects (
            id, name, complexity, regulatory_intensity,
            start_date, end_date, cost_of_delay_weekly, risk_tolerance
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "project-aurora",
            "Project Aurora",
            "high",
            "moderate",
            "2024-01-01",
            "2024-06-30",
            50000.0,
            "medium",
        ),
    )

    cursor.executemany(
        """
        INSERT INTO phases (project_id, phase_name, start_date, end_date, gate_threshold)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            ("project-aurora", "discovery", "2024-01-01", "2024-02-15", 0.7),
            ("project-aurora", "deployment", "2024-04-01", "2024-06-30", 0.8),
        ],
    )

    cursor.executemany(
        """
        INSERT INTO project_requirements (
            project_id, phase_name, skill_id, required_level,
            min_level, fte_weeks, criticality
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("project-aurora", "discovery", "skill-analytics", 4.0, 3.5, 6.0, 0.9),
            ("project-aurora", "deployment", "skill-python", 4.2, 3.8, 8.0, 0.8),
        ],
    )

    cursor.executemany(
        """
        INSERT INTO assignments (
            project_id, person_id, phase_name, availability, start_date, end_date
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "project-aurora",
                "person-alice",
                "discovery",
                0.5,
                "2024-01-01",
                "2024-02-15",
            ),
            (
                "project-aurora",
                "person-bob",
                "deployment",
                0.6,
                "2024-04-01",
                "2024-06-30",
            ),
        ],
    )

    cursor.execute(
        """
        INSERT INTO evidence (
            person_id, skill_id, evidence_type, description, date_achieved, verified_by
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "person-alice",
            "skill-python",
            "certification",
            "Completed advanced Python certification",
            "2023-10-10",
            "admin",
        ),
    )

    conn.commit()


@pytest.fixture(scope="session")
def api_client(tmp_path_factory: pytest.TempPathFactory) -> Generator[TestClient, None, None]:
    """Provide a configured TestClient backed by an isolated SQLite database."""

    db_dir = tmp_path_factory.mktemp("db")
    db_path = Path(db_dir) / "test.db"
    db_url = f"sqlite:///{db_path}"

    tracked_env_vars: Dict[str, str | None] = {
        "APP_ENV": os.environ.get("APP_ENV"),
        "JWT_SECRET_KEY": os.environ.get("JWT_SECRET_KEY"),
        "CORS_ALLOWED_ORIGINS": os.environ.get("CORS_ALLOWED_ORIGINS"),
        "ACCESS_TOKEN_EXPIRE_MINUTES": os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES"),
        "DATABASE_URL": os.environ.get("DATABASE_URL"),
    }

    os.environ.setdefault("APP_ENV", "test")
    os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key")
    os.environ.setdefault("CORS_ALLOWED_ORIGINS", "http://testserver")
    os.environ.setdefault("ACCESS_TOKEN_EXPIRE_MINUTES", "15")
    os.environ["DATABASE_URL"] = db_url

    get_config.cache_clear()

    conn = init_database(db_url, skip_if_exists=False)
    _seed_sample_data(conn)
    conn.close()

    # Ensure authentication tables and default users exist
    get_authentication_system()

    # Avoid spawning long-running background threads during tests
    import importlib

    import api.metrics_recalculation as metrics_recalculation
    import api.websocket_manager as websocket_manager

    metrics_recalculation.start_recalculation_monitor = lambda: None  # type: ignore[assignment]
    websocket_manager.start_background_tasks = lambda: None  # type: ignore[assignment]

    main_module = importlib.reload(importlib.import_module("api.main"))
    client = TestClient(main_module.app)

    yield client

    client.close()
    get_config.cache_clear()

    for key, original in tracked_env_vars.items():
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original

    # Clean up TestClient state to avoid holding the database file open
    try:
        Path(db_path).unlink(missing_ok=True)
    except OSError:
        pass


@pytest.fixture()
def db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Provide a direct database connection for tests that need to inspect data."""

    conn = get_db_connection(os.environ.get("DATABASE_URL"))
    try:
        yield conn
    finally:
        conn.close()
