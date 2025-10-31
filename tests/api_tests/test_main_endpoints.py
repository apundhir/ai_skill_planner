"""High-level API tests covering key endpoints and error handling."""
from __future__ import annotations

from typing import Iterable

import pytest


@pytest.mark.parametrize(
    "path", ["/health", "/skills", "/people", "/projects", "/analytics/team-summary"]
)
def test_endpoints_return_success(api_client, path: str) -> None:
    response = api_client.get(path)
    assert response.status_code == 200


def test_skills_category_filtering(api_client) -> None:
    all_skills = api_client.get("/skills").json()
    assert {skill["category"] for skill in all_skills} == {"engineering", "data-science"}

    engineering_skills = api_client.get("/skills", params={"category": "engineering"}).json()
    assert engineering_skills
    assert all(skill["category"] == "engineering" for skill in engineering_skills)


def test_skill_categories_endpoint_reports_counts(api_client) -> None:
    response = api_client.get("/skills/categories")
    payload = response.json()
    categories = {item["category"]: item["skill_count"] for item in payload["categories"]}
    assert categories["engineering"] == 1
    assert categories["data-science"] == 1


def test_people_include_skills_by_default(api_client) -> None:
    payload = api_client.get("/people").json()
    assert len(payload) == 2
    assert payload[0]["skills"]
    assert any(skill["skill_name"] == "Python" for skill in payload[0]["skills"])


def test_people_can_exclude_skills(api_client) -> None:
    payload = api_client.get("/people", params={"include_skills": "false"}).json()
    assert len(payload) == 2
    assert all(not person["skills"] for person in payload)


def test_get_person_returns_404_for_unknown_person(api_client) -> None:
    response = api_client.get("/people/unknown-person")
    assert response.status_code == 404
    assert response.json()["detail"] == "Person not found"


def _flatten(iterable: Iterable[Iterable]) -> list:
    return [item for sequence in iterable for item in sequence]


def test_projects_include_phases_and_requirements(api_client) -> None:
    payload = api_client.get("/projects").json()
    assert len(payload) == 1
    project = payload[0]
    assert project["phases"]
    requirements = _flatten(phase["requirements"] for phase in project["phases"])
    assert any(req["skill_name"] == "Analytics" for req in requirements)


def test_projects_can_exclude_phases(api_client) -> None:
    payload = api_client.get("/projects", params={"include_phases": "false"}).json()
    assert payload
    assert all(not project["phases"] for project in payload)


def test_get_project_returns_404_for_unknown_project(api_client) -> None:
    response = api_client.get("/projects/nonexistent")
    assert response.status_code == 404
    assert response.json()["detail"] == "Project not found"


def test_assignments_filtering(api_client) -> None:
    response = api_client.get("/assignments", params={"person_id": "person-alice"})
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["person_name"] == "Alice Analyst"
    assert payload[0]["phase_name"] == "discovery"


def test_evidence_filters_by_skill(api_client) -> None:
    response = api_client.get(
        "/evidence", params={"person_id": "person-alice", "skill_id": "skill-python"}
    )
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["evidence_type"] == "certification"


def test_analytics_team_summary_contains_expected_sections(api_client) -> None:
    payload = api_client.get("/analytics/team-summary").json()
    assert set(payload.keys()) == {"team", "projects", "evidence", "generated_at"}
    assert payload["team"]["total_people"] >= 2
    assert payload["projects"]["total_projects"] >= 1


def test_health_endpoint_payload(api_client) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "X-Request-ID" in response.headers
