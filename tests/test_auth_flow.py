"""Integration tests for the authentication flow and admin access control."""
from __future__ import annotations

import pytest


@pytest.fixture()
def login(api_client):
    def _login(username: str, password: str):
        return api_client.post("/auth/login", json={"username": username, "password": password})

    return _login


def test_login_success_returns_signed_token(login):
    response = login("admin", "admin123")
    assert response.status_code == 200
    payload = response.json()
    assert payload["token_type"] == "bearer"
    assert payload["user"]["role"] == "ADMIN"
    assert isinstance(payload["expires_in"], int)
    assert payload["expires_in"] > 0
    assert payload["access_token"]


def test_login_rejects_invalid_credentials(login):
    response = login("admin", "wrong-password")
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid username or password"


def test_me_endpoint_requires_valid_token(api_client, login):
    login_response = login("admin", "admin123")
    token = login_response.json()["access_token"]

    valid_response = api_client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert valid_response.status_code == 200
    assert valid_response.json()["user"]["username"] == "admin"

    invalid_response = api_client.get(
        "/auth/me",
        headers={"Authorization": "Bearer invalid-token"},
    )
    assert invalid_response.status_code == 401


def test_admin_routes_require_admin_role(api_client, login):
    exec_login = login("executive", "exec123")
    exec_token = exec_login.json()["access_token"]

    forbidden_response = api_client.get(
        "/admin/system/status",
        headers={"Authorization": f"Bearer {exec_token}"},
    )
    assert forbidden_response.status_code == 403

    admin_login = login("admin", "admin123")
    admin_token = admin_login.json()["access_token"]

    allowed_response = api_client.get(
        "/admin/system/status",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert allowed_response.status_code == 200
    assert "system_health" in allowed_response.json()
