"""Integration tests for the authentication flow and admin access control."""
from __future__ import annotations

import os
import tempfile
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Configure test environment before importing the application
_TEST_DB_DIR = tempfile.mkdtemp(prefix="ai_skill_planner_test_")
_TEST_DB_PATH = os.path.join(_TEST_DB_DIR, "test.db")
_TEST_DB_URL = f"sqlite:///{_TEST_DB_PATH}"

sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key")
os.environ.setdefault("CORS_ALLOWED_ORIGINS", "http://testserver")
os.environ["DATABASE_URL"] = _TEST_DB_URL
os.environ.setdefault("ACCESS_TOKEN_EXPIRE_MINUTES", "30")

from database.init_db import init_database  # noqa: E402
from api.core.config import get_config  # noqa: E402

# Initialize database schema before importing application components
get_config.cache_clear()
_connection = init_database(_TEST_DB_URL, skip_if_exists=False)
_connection.close()

from security.auth import get_authentication_system  # noqa: E402
from api.main import app  # noqa: E402

get_authentication_system()  # Ensures default users exist

client = TestClient(app)


def _login(username: str, password: str):
    return client.post("/auth/login", json={"username": username, "password": password})


def test_login_success_returns_signed_token():
    response = _login("admin", "admin123")
    assert response.status_code == 200
    payload = response.json()
    assert payload["token_type"] == "bearer"
    assert payload["user"]["role"] == "ADMIN"
    assert isinstance(payload["expires_in"], int)
    assert payload["expires_in"] > 0
    assert payload["access_token"]


def test_login_rejects_invalid_credentials():
    response = _login("admin", "wrong-password")
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid username or password"


def test_me_endpoint_requires_valid_token():
    login_response = _login("admin", "admin123")
    token = login_response.json()["access_token"]

    valid_response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert valid_response.status_code == 200
    assert valid_response.json()["user"]["username"] == "admin"

    invalid_response = client.get(
        "/auth/me",
        headers={"Authorization": "Bearer invalid-token"},
    )
    assert invalid_response.status_code == 401


def test_admin_routes_require_admin_role():
    exec_login = _login("executive", "exec123")
    exec_token = exec_login.json()["access_token"]

    forbidden_response = client.get(
        "/admin/system/status",
        headers={"Authorization": f"Bearer {exec_token}"},
    )
    assert forbidden_response.status_code == 403

    admin_login = _login("admin", "admin123")
    admin_token = admin_login.json()["access_token"]

    allowed_response = client.get(
        "/admin/system/status",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert allowed_response.status_code == 200
    assert "system_health" in allowed_response.json()
