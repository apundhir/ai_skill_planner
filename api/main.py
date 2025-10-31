#!/usr/bin/env python3
"""
AI Skill Planner API - Basic endpoints for data access
FastAPI application providing REST endpoints for the skill gap analysis system
"""

import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sqlite3

# Add parent directory to path to import database module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.init_db import get_db_connection
from api.core.config import get_config
from api.gap_endpoints import gap_router
from api.executive_endpoints import exec_router
from api.validation_endpoints import validation_router
from api.admin_endpoints import admin_router
from api.metrics_recalculation import metrics_router, start_recalculation_monitor
from api.user_project_endpoints import user_project_router
from api.websocket_manager import websocket_router, start_background_tasks
from api.routes.auth import router as auth_router

# Initialize FastAPI app
app = FastAPI(
    title="AI Team Skill Gap Planner API",
    description="REST API for analyzing team skill gaps and capacity planning",
    version="1.0.0"
)

# Load runtime configuration
settings = get_config()

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(gap_router)
app.include_router(exec_router)
app.include_router(validation_router)
app.include_router(admin_router)
app.include_router(user_project_router)
app.include_router(metrics_router)
app.include_router(websocket_router)
app.include_router(auth_router)

# Start background services
start_recalculation_monitor()
start_background_tasks()

# Mount static files for heat map visualization
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    pass  # Static files not available in all environments

# Pydantic models for API responses
class Skill(BaseModel):
    id: str
    name: str
    category: str
    decay_rate: float

class PersonSkill(BaseModel):
    person_id: str
    skill_id: str
    skill_name: str
    base_level: float
    effective_level: Optional[float] = None
    confidence_low: Optional[float] = None
    confidence_high: Optional[float] = None
    last_used: Optional[date] = None

class Person(BaseModel):
    id: str
    name: str
    location: str
    timezone: str
    fte: float
    cost_hourly: float
    skills: List[PersonSkill] = []

class ProjectRequirement(BaseModel):
    skill_id: str
    skill_name: str
    required_level: float
    min_level: float
    fte_weeks: float
    criticality: float

class ProjectPhase(BaseModel):
    phase_name: str
    start_date: date
    end_date: date
    gate_threshold: float
    requirements: List[ProjectRequirement] = []

class Project(BaseModel):
    id: str
    name: str
    complexity: str
    regulatory_intensity: str
    start_date: date
    end_date: date
    cost_of_delay_weekly: float
    risk_tolerance: str
    phases: List[ProjectPhase] = []

class Assignment(BaseModel):
    project_id: str
    project_name: str
    person_id: str
    person_name: str
    phase_name: str
    availability: float
    start_date: date
    end_date: date

class Evidence(BaseModel):
    id: int
    person_id: str
    skill_id: str
    skill_name: str
    evidence_type: str
    description: str
    date_achieved: date
    verified_by: Optional[str] = None

# Dependency to get database connection
def get_db():
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()

# === SKILLS ENDPOINTS ===

@app.get("/skills", response_model=List[Skill])
def get_skills(category: Optional[str] = None, db: sqlite3.Connection = Depends(get_db)):
    """Get all skills, optionally filtered by category"""
    cursor = db.cursor()

    if category:
        cursor.execute("""
            SELECT id, name, category, decay_rate
            FROM skills
            WHERE category = ?
            ORDER BY name
        """, (category,))
    else:
        cursor.execute("""
            SELECT id, name, category, decay_rate
            FROM skills
            ORDER BY category, name
        """)

    skills = [Skill(**dict(row)) for row in cursor.fetchall()]
    return skills

@app.get("/skills/categories")
def get_skill_categories(db: sqlite3.Connection = Depends(get_db)):
    """Get all skill categories with counts"""
    cursor = db.cursor()
    cursor.execute("""
        SELECT category, COUNT(*) as skill_count
        FROM skills
        GROUP BY category
        ORDER BY skill_count DESC
    """)

    categories = [{"category": row["category"], "skill_count": row["skill_count"]}
                 for row in cursor.fetchall()]
    return {"categories": categories}

# === PEOPLE ENDPOINTS ===

@app.get("/people", response_model=List[Person])
def get_people(include_skills: bool = True, db: sqlite3.Connection = Depends(get_db)):
    """Get all people with optional skill details"""
    cursor = db.cursor()

    # Get people
    cursor.execute("""
        SELECT id, name, location, timezone, fte, cost_hourly
        FROM people
        ORDER BY name
    """)

    people = []
    for row in cursor.fetchall():
        person = Person(**dict(row))

        if include_skills:
            # Get skills for this person
            cursor.execute("""
                SELECT ps.person_id, ps.skill_id, s.name as skill_name,
                       ps.base_level, ps.effective_level, ps.confidence_low,
                       ps.confidence_high, ps.last_used
                FROM person_skills ps
                JOIN skills s ON ps.skill_id = s.id
                WHERE ps.person_id = ?
                ORDER BY ps.base_level DESC
            """, (person.id,))

            person.skills = [PersonSkill(**dict(skill_row))
                           for skill_row in cursor.fetchall()]

        people.append(person)

    return people

@app.get("/people/{person_id}", response_model=Person)
def get_person(person_id: str, db: sqlite3.Connection = Depends(get_db)):
    """Get a specific person with their skills"""
    cursor = db.cursor()

    # Get person
    cursor.execute("""
        SELECT id, name, location, timezone, fte, cost_hourly
        FROM people
        WHERE id = ?
    """, (person_id,))

    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Person not found")

    person = Person(**dict(row))

    # Get skills
    cursor.execute("""
        SELECT ps.person_id, ps.skill_id, s.name as skill_name,
               ps.base_level, ps.effective_level, ps.confidence_low,
               ps.confidence_high, ps.last_used
        FROM person_skills ps
        JOIN skills s ON ps.skill_id = s.id
        WHERE ps.person_id = ?
        ORDER BY ps.base_level DESC
    """, (person_id,))

    person.skills = [PersonSkill(**dict(skill_row))
                   for skill_row in cursor.fetchall()]

    return person

# === PROJECTS ENDPOINTS ===

@app.get("/projects", response_model=List[Project])
def get_projects(
    include_phases: bool = True,
    user_id: Optional[str] = None,
    role: Optional[str] = None,
    db: sqlite3.Connection = Depends(get_db)
):
    """Get all projects with optional phase and requirement details, filtered by user access"""
    cursor = db.cursor()

    # Apply role-based filtering
    if user_id and role:
        try:
            from api.user_project_endpoints import get_user_accessible_projects
            accessible_project_ids = get_user_accessible_projects(user_id, role)

            if not accessible_project_ids:
                return []

            # Get projects that user can access
            placeholders = ",".join(["?" for _ in accessible_project_ids])
            cursor.execute(f"""
                SELECT id, name, complexity, regulatory_intensity, start_date, end_date,
                       cost_of_delay_weekly, risk_tolerance
                FROM projects
                WHERE id IN ({placeholders})
                ORDER BY cost_of_delay_weekly DESC
            """, accessible_project_ids)
        except ImportError:
            # Fallback if user_project_endpoints not available
            cursor.execute("""
                SELECT id, name, complexity, regulatory_intensity, start_date, end_date,
                       cost_of_delay_weekly, risk_tolerance
                FROM projects
                ORDER BY cost_of_delay_weekly DESC
            """)
    else:
        # No filtering - return all projects
        cursor.execute("""
            SELECT id, name, complexity, regulatory_intensity, start_date, end_date,
                   cost_of_delay_weekly, risk_tolerance
            FROM projects
            ORDER BY cost_of_delay_weekly DESC
        """)

    projects = []
    for row in cursor.fetchall():
        project = Project(**dict(row))

        if include_phases:
            # Get phases for this project
            cursor.execute("""
                SELECT phase_name, start_date, end_date, gate_threshold
                FROM phases
                WHERE project_id = ?
                ORDER BY start_date
            """, (project.id,))

            project.phases = []
            for phase_row in cursor.fetchall():
                phase = ProjectPhase(**dict(phase_row))

                # Get requirements for this phase
                cursor.execute("""
                    SELECT pr.skill_id, s.name as skill_name, pr.required_level,
                           pr.min_level, pr.fte_weeks, pr.criticality
                    FROM project_requirements pr
                    JOIN skills s ON pr.skill_id = s.id
                    WHERE pr.project_id = ? AND pr.phase_name = ?
                    ORDER BY pr.criticality DESC
                """, (project.id, phase.phase_name))

                phase.requirements = [ProjectRequirement(**dict(req_row))
                                    for req_row in cursor.fetchall()]

                project.phases.append(phase)

        projects.append(project)

    return projects

@app.get("/projects/{project_id}", response_model=Project)
def get_project(project_id: str, db: sqlite3.Connection = Depends(get_db)):
    """Get a specific project with full details"""
    cursor = db.cursor()

    # Get project
    cursor.execute("""
        SELECT id, name, complexity, regulatory_intensity, start_date, end_date,
               cost_of_delay_weekly, risk_tolerance
        FROM projects
        WHERE id = ?
    """, (project_id,))

    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Project not found")

    project = Project(**dict(row))

    # Get phases (same logic as above)
    cursor.execute("""
        SELECT phase_name, start_date, end_date, gate_threshold
        FROM phases
        WHERE project_id = ?
        ORDER BY start_date
    """, (project_id,))

    project.phases = []
    for phase_row in cursor.fetchall():
        phase = ProjectPhase(**dict(phase_row))

        cursor.execute("""
            SELECT pr.skill_id, s.name as skill_name, pr.required_level,
                   pr.min_level, pr.fte_weeks, pr.criticality
            FROM project_requirements pr
            JOIN skills s ON pr.skill_id = s.id
            WHERE pr.project_id = ? AND pr.phase_name = ?
            ORDER BY pr.criticality DESC
        """, (project_id, phase.phase_name))

        phase.requirements = [ProjectRequirement(**dict(req_row))
                            for req_row in cursor.fetchall()]

        project.phases.append(phase)

    return project

# === ASSIGNMENTS ENDPOINTS ===

@app.get("/assignments", response_model=List[Assignment])
def get_assignments(project_id: Optional[str] = None,
                   person_id: Optional[str] = None,
                   db: sqlite3.Connection = Depends(get_db)):
    """Get assignments, optionally filtered by project or person"""
    cursor = db.cursor()

    query = """
        SELECT a.project_id, pr.name as project_name, a.person_id, p.name as person_name,
               a.phase_name, a.availability, a.start_date, a.end_date
        FROM assignments a
        JOIN projects pr ON a.project_id = pr.id
        JOIN people p ON a.person_id = p.id
    """

    params = []
    if project_id:
        query += " WHERE a.project_id = ?"
        params.append(project_id)
    elif person_id:
        query += " WHERE a.person_id = ?"
        params.append(person_id)

    query += " ORDER BY pr.name, p.name, a.phase_name"

    cursor.execute(query, params)

    assignments = [Assignment(**dict(row)) for row in cursor.fetchall()]
    return assignments

# === EVIDENCE ENDPOINTS ===

@app.get("/evidence", response_model=List[Evidence])
def get_evidence(person_id: Optional[str] = None,
                skill_id: Optional[str] = None,
                evidence_type: Optional[str] = None,
                db: sqlite3.Connection = Depends(get_db)):
    """Get evidence, optionally filtered by person, skill, or type"""
    cursor = db.cursor()

    query = """
        SELECT e.id, e.person_id, e.skill_id, s.name as skill_name,
               e.evidence_type, e.description, e.date_achieved, e.verified_by
        FROM evidence e
        JOIN skills s ON e.skill_id = s.id
    """

    params = []
    conditions = []

    if person_id:
        conditions.append("e.person_id = ?")
        params.append(person_id)
    if skill_id:
        conditions.append("e.skill_id = ?")
        params.append(skill_id)
    if evidence_type:
        conditions.append("e.evidence_type = ?")
        params.append(evidence_type)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY e.date_achieved DESC"

    cursor.execute(query, params)

    evidence_list = [Evidence(**dict(row)) for row in cursor.fetchall()]
    return evidence_list

# === ANALYTICS ENDPOINTS ===

@app.get("/analytics/team-summary")
def get_team_summary(db: sqlite3.Connection = Depends(get_db)):
    """Get high-level team analytics"""
    cursor = db.cursor()

    # Team composition
    cursor.execute("""
        SELECT
            COUNT(*) as total_people,
            ROUND(AVG(cost_hourly), 0) as avg_hourly_cost,
            ROUND(SUM(fte), 1) as total_fte
        FROM people
    """)
    team_stats = dict(cursor.fetchone())

    # Project summary
    cursor.execute("""
        SELECT
            COUNT(*) as total_projects,
            ROUND(AVG(cost_of_delay_weekly), 0) as avg_cost_of_delay,
            COUNT(DISTINCT a.person_id) as people_assigned
        FROM projects p
        LEFT JOIN assignments a ON p.id = a.project_id
    """)
    project_stats = dict(cursor.fetchone())

    # Evidence summary
    cursor.execute("""
        SELECT
            COUNT(*) as total_evidence,
            COUNT(DISTINCT person_id) as people_with_evidence,
            COUNT(*) * 1.0 / COUNT(DISTINCT person_id) as avg_evidence_per_person
        FROM evidence
    """)
    evidence_stats = dict(cursor.fetchone())

    return {
        "team": team_stats,
        "projects": project_stats,
        "evidence": evidence_stats,
        "generated_at": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Serve the main executive dashboard"""
    return FileResponse("static/executive_dashboard.html")

@app.get("/heatmap")
async def heatmap():
    """Serve the skill gap heatmap"""
    return FileResponse("static/heatmap.html")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)