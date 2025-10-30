#!/usr/bin/env python3
"""
Admin-only endpoints for data management and system administration
Handles Excel file uploads, project onboarding, and system configuration
"""

from __future__ import annotations

import sys
import os
import io
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from datetime import datetime, date
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel
import sqlite3
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.init_db import get_db_connection
from engines.gap_analysis import GapAnalysisEngine
from engines.proficiency import ProficiencyCalculator
from api.dependencies.auth import verify_admin_role

if TYPE_CHECKING:
    import pandas as pd
else:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import multipart  # type: ignore

    MULTIPART_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    MULTIPART_AVAILABLE = False

# Create router for admin endpoints
admin_router = APIRouter(prefix="/admin", tags=["Administration"])

# Initialize engines
gap_engine = GapAnalysisEngine()
proficiency_calc = ProficiencyCalculator()

# Pydantic models
class UploadResult(BaseModel):
    success: bool
    message: str
    details: Dict[str, Any]
    errors: List[str] = []

class ValidationResult(BaseModel):
    valid: bool
    errors: List[str]
    warnings: List[str]
    records_count: int

class ProjectOnboardingResult(BaseModel):
    project_id: str
    project_name: str
    phases_created: int
    requirements_created: int
    metrics_updated: bool


class ProjectOnboardingPayload(BaseModel):
    project: Dict[str, Any]
    phases: List[Dict[str, Any]]
    requirements: List[Dict[str, Any]]


def _require_pandas():
    if pd is None:
        raise HTTPException(
            status_code=500,
            detail="Pandas is required for this operation but is not installed",
        )
    return pd


def validate_project_excel(df: pd.DataFrame) -> ValidationResult:
    """Validate project Excel file structure and data"""
    pd = _require_pandas()
    errors = []
    warnings = []

    # Required columns for project data
    required_cols = ['project_name', 'complexity', 'start_date', 'end_date',
                    'cost_of_delay_weekly', 'regulatory_intensity', 'risk_tolerance']

    # Check required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")

    # Check data types and values
    if 'cost_of_delay_weekly' in df.columns:
        non_numeric = df[~pd.to_numeric(df['cost_of_delay_weekly'], errors='coerce').notna()]
        if len(non_numeric) > 0:
            errors.append(f"Non-numeric cost_of_delay_weekly values in rows: {list(non_numeric.index)}")

    # Check date formats
    date_cols = ['start_date', 'end_date']
    for col in date_cols:
        if col in df.columns:
            try:
                pd.to_datetime(df[col])
            except:
                errors.append(f"Invalid date format in column {col}")

    # Check for empty project names
    if 'project_name' in df.columns:
        empty_names = df[df['project_name'].isna() | (df['project_name'] == '')]
        if len(empty_names) > 0:
            errors.append(f"Empty project names in rows: {list(empty_names.index)}")

    # Warnings for optional issues
    if len(df) > 10:
        warnings.append(f"Large dataset ({len(df)} rows) - processing may take time")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        records_count=len(df)
    )

def validate_team_excel(df: pd.DataFrame) -> ValidationResult:
    """Validate team Excel file structure and data"""
    pd = _require_pandas()
    errors = []
    warnings = []

    # Required columns for team data
    required_cols = ['person_name', 'location', 'timezone', 'fte', 'cost_hourly']

    # Check required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")

    # Check numeric fields
    numeric_fields = ['fte', 'cost_hourly']
    for field in numeric_fields:
        if field in df.columns:
            non_numeric = df[~pd.to_numeric(df[field], errors='coerce').notna()]
            if len(non_numeric) > 0:
                errors.append(f"Non-numeric {field} values in rows: {list(non_numeric.index)}")

    # Check FTE range
    if 'fte' in df.columns:
        try:
            fte_values = pd.to_numeric(df['fte'], errors='coerce')
            invalid_fte = fte_values[(fte_values < 0) | (fte_values > 1)]
            if len(invalid_fte) > 0:
                warnings.append(f"FTE values outside 0-1 range detected")
        except:
            pass

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        records_count=len(df)
    )

def validate_skills_excel(df: pd.DataFrame) -> ValidationResult:
    """Validate skills Excel file structure and data"""
    pd = _require_pandas()
    errors = []
    warnings = []

    # Required columns for skills data
    required_cols = ['person_name', 'skill_name', 'skill_category', 'base_level']

    # Check required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")

    # Check skill level range
    if 'base_level' in df.columns:
        try:
            skill_levels = pd.to_numeric(df['base_level'], errors='coerce')
            invalid_levels = skill_levels[(skill_levels < 0) | (skill_levels > 10)]
            if len(invalid_levels) > 0:
                errors.append(f"Skill levels outside 0-10 range in rows: {list(invalid_levels.dropna().index)}")
        except:
            errors.append("Invalid skill level format")

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        records_count=len(df)
    )

if MULTIPART_AVAILABLE:

    @admin_router.post("/upload/project", response_model=UploadResult)
    async def upload_project_data(
        file: UploadFile = File(...),
        _: bool = Depends(verify_admin_role)
    ):
        """Upload project data from Excel file"""

        if not file.filename.endswith((".xlsx", ".xls")):
            raise HTTPException(status_code=400, detail="File must be Excel format (.xlsx or .xls)")

        try:
            pandas_module = _require_pandas()
            contents = await file.read()
            df = pandas_module.read_excel(io.BytesIO(contents))

            validation = validate_project_excel(df)
            if not validation.valid:
                return UploadResult(
                    success=False,
                    message="Validation failed",
                    details={"validation": validation.dict()},
                    errors=validation.errors,
                )

            conn = get_db_connection()
            cursor = conn.cursor()

            projects_created = 0
            errors: List[str] = []

            for idx, row in df.iterrows():
                try:
                    project_id = f"proj_{datetime.now().strftime('%Y%m%d')}_{idx}"

                    cursor.execute(
                        """
                        INSERT INTO projects
                        (id, name, complexity, regulatory_intensity, start_date, end_date,
                         cost_of_delay_weekly, risk_tolerance)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            project_id,
                            row["project_name"],
                            row["complexity"],
                            row["regulatory_intensity"],
                            pandas_module.to_datetime(row["start_date"]).date(),
                            pandas_module.to_datetime(row["end_date"]).date(),
                            float(row["cost_of_delay_weekly"]),
                            row["risk_tolerance"],
                        ),
                    )

                    projects_created += 1

                except Exception as exc:  # pragma: no cover - defensive
                    errors.append(f"Row {idx}: {exc}")

            conn.commit()
            conn.close()

            return UploadResult(
                success=True,
                message=f"Successfully created {projects_created} projects",
                details={
                    "projects_created": projects_created,
                    "validation": validation.dict(),
                },
                errors=errors,
            )

        except Exception as exc:  # pragma: no cover - defensive
            return UploadResult(
                success=False,
                message=f"Upload failed: {exc}",
                details={},
                errors=[str(exc)],
            )

    @admin_router.post("/upload/team", response_model=UploadResult)
    async def upload_team_data(
        file: UploadFile = File(...),
        _: bool = Depends(verify_admin_role)
    ):
        """Upload team member data from Excel file"""

        if not file.filename.endswith((".xlsx", ".xls")):
            raise HTTPException(status_code=400, detail="File must be Excel format (.xlsx or .xls)")

        try:
            pandas_module = _require_pandas()
            contents = await file.read()
            df = pandas_module.read_excel(io.BytesIO(contents))

            validation = validate_team_excel(df)
            if not validation.valid:
                return UploadResult(
                    success=False,
                    message="Validation failed",
                    details={"validation": validation.dict()},
                    errors=validation.errors,
                )

            conn = get_db_connection()
            cursor = conn.cursor()

            people_created = 0
            errors: List[str] = []

            for idx, row in df.iterrows():
                try:
                    person_id = f"person_{datetime.now().strftime('%Y%m%d')}_{idx}"

                    cursor.execute(
                        """
                        INSERT INTO people
                        (id, name, location, timezone, fte, cost_hourly)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            person_id,
                            row["person_name"],
                            row["location"],
                            row["timezone"],
                            float(row["fte"]),
                            float(row["cost_hourly"]),
                        ),
                    )

                    people_created += 1

                except Exception as exc:  # pragma: no cover - defensive
                    errors.append(f"Row {idx}: {exc}")

            conn.commit()
            conn.close()

            return UploadResult(
                success=True,
                message=f"Successfully created {people_created} team members",
                details={
                    "people_created": people_created,
                    "validation": validation.dict(),
                },
                errors=errors,
            )

        except Exception as exc:  # pragma: no cover - defensive
            return UploadResult(
                success=False,
                message=f"Upload failed: {exc}",
                details={},
                errors=[str(exc)],
            )

    @admin_router.post("/upload/skills", response_model=UploadResult)
    async def upload_skills_data(
        file: UploadFile = File(...),
        _: bool = Depends(verify_admin_role)
    ):
        """Upload team skills data from Excel file"""

        if not file.filename.endswith((".xlsx", ".xls")):
            raise HTTPException(status_code=400, detail="File must be Excel format (.xlsx or .xls)")

        try:
            pandas_module = _require_pandas()
            contents = await file.read()
            df = pandas_module.read_excel(io.BytesIO(contents))

            validation = validate_skills_excel(df)
            if not validation.valid:
                return UploadResult(
                    success=False,
                    message="Validation failed",
                    details={"validation": validation.dict()},
                    errors=validation.errors,
                )

            conn = get_db_connection()
            cursor = conn.cursor()

            skills_created = 0
            errors: List[str] = []

            for idx, row in df.iterrows():
                try:
                    cursor.execute("SELECT id FROM skills WHERE name = ?", (row["skill_name"],))
                    skill_result = cursor.fetchone()

                    if not skill_result:
                        skill_id = f"skill_{row['skill_name'].lower().replace(' ', '_')}"
                        cursor.execute(
                            """
                            INSERT INTO skills (id, name, category, decay_rate)
                            VALUES (?, ?, ?, ?)
                            """,
                            (skill_id, row["skill_name"], row["skill_category"], 0.1),
                        )
                    else:
                        skill_id = skill_result["id"]

                    cursor.execute("SELECT id FROM people WHERE name = ?", (row["person_name"],))
                    person_result = cursor.fetchone()

                    if person_result:
                        person_id = person_result["id"]

                        cursor.execute(
                            """
                            INSERT OR REPLACE INTO person_skills
                            (person_id, skill_id, base_level, last_used)
                            VALUES (?, ?, ?, ?)
                            """,
                            (
                                person_id,
                                skill_id,
                                float(row["base_level"]),
                                datetime.now().date(),
                            ),
                        )

                        skills_created += 1
                    else:
                        errors.append(f"Row {idx}: Person '{row['person_name']}' not found")

                except Exception as exc:  # pragma: no cover - defensive
                    errors.append(f"Row {idx}: {exc}")

            conn.commit()
            conn.close()

            try:
                proficiency_calc.update_all_skill_levels()
            except Exception:  # pragma: no cover - defensive
                pass

            return UploadResult(
                success=True,
                message=f"Successfully created {skills_created} skill assignments",
                details={
                    "skills_created": skills_created,
                    "validation": validation.dict(),
                },
                errors=errors,
            )

        except Exception as exc:  # pragma: no cover - defensive
            return UploadResult(
                success=False,
                message=f"Upload failed: {exc}",
                details={},
                errors=[str(exc)],
            )

else:

    @admin_router.post("/upload/project", response_model=UploadResult)
    async def upload_project_data(_: bool = Depends(verify_admin_role)):
        """Inform clients that file upload support is unavailable."""

        raise HTTPException(
            status_code=503,
            detail="File upload endpoints require the python-multipart package",
        )

    @admin_router.post("/upload/team", response_model=UploadResult)
    async def upload_team_data(_: bool = Depends(verify_admin_role)):
        raise HTTPException(
            status_code=503,
            detail="File upload endpoints require the python-multipart package",
        )

    @admin_router.post("/upload/skills", response_model=UploadResult)
    async def upload_skills_data(_: bool = Depends(verify_admin_role)):
        raise HTTPException(
            status_code=503,
            detail="File upload endpoints require the python-multipart package",
        )


@admin_router.post("/onboard/project", response_model=ProjectOnboardingResult)
async def onboard_new_project(
    payload: ProjectOnboardingPayload,
    _: bool = Depends(verify_admin_role)
):
    """Complete project onboarding with phases and requirements"""

    try:
        project = payload.project
        phases = payload.phases
        requirements = payload.requirements

        conn = get_db_connection()
        cursor = conn.cursor()

        # Create project
        project_id = f"proj_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        cursor.execute("""
            INSERT INTO projects
            (id, name, complexity, regulatory_intensity, start_date, end_date,
             cost_of_delay_weekly, risk_tolerance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            project_id,
            project['name'],
            project['complexity'],
            project['regulatory_intensity'],
            project['start_date'],
            project['end_date'],
            project['cost_of_delay_weekly'],
            project['risk_tolerance']
        ))

        # Create phases
        phases_created = 0
        for phase in phases:
            cursor.execute("""
                INSERT INTO phases
                (project_id, phase_name, start_date, end_date, gate_threshold)
                VALUES (?, ?, ?, ?, ?)
            """, (
                project_id,
                phase['phase_name'],
                phase['start_date'],
                phase['end_date'],
                phase.get('gate_threshold', 0.8)
            ))
            phases_created += 1

        # Create requirements
        requirements_created = 0
        for req in requirements:
            # Ensure skill exists
            cursor.execute("SELECT id FROM skills WHERE id = ?", (req['skill_id'],))
            if cursor.fetchone():
                cursor.execute("""
                    INSERT INTO project_requirements
                    (project_id, phase_name, skill_id, required_level, min_level, fte_weeks, criticality)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    project_id,
                    req['phase_name'],
                    req['skill_id'],
                    req['required_level'],
                    req['min_level'],
                    req['fte_weeks'],
                    req['criticality']
                ))
                requirements_created += 1

        conn.commit()
        conn.close()

        # Trigger metrics recalculation
        metrics_updated = False
        try:
            proficiency_calc.update_all_skill_levels()
            metrics_updated = True
        except Exception as e:
            print(f"Metrics update failed: {e}")

        return ProjectOnboardingResult(
            project_id=project_id,
            project_name=project['name'],
            phases_created=phases_created,
            requirements_created=requirements_created,
            metrics_updated=metrics_updated
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Project onboarding failed: {str(e)}")

@admin_router.post("/recalculate-metrics")
async def recalculate_all_metrics(_: bool = Depends(verify_admin_role)):
    """Trigger recalculation of all system metrics"""

    try:
        # Update proficiency calculations
        proficiency_stats = proficiency_calc.update_all_skill_levels()

        # Get updated organization overview
        org_overview = gap_engine.get_organization_gap_overview()

        return {
            "success": True,
            "message": "Metrics recalculated successfully",
            "details": {
                "proficiency_stats": proficiency_stats,
                "total_projects": len(org_overview.get('project_summaries', [])),
                "recalculated_at": datetime.now().isoformat()
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics recalculation failed: {str(e)}")

@admin_router.get("/system/status")
async def get_system_status(_: bool = Depends(verify_admin_role)):
    """Get comprehensive system status for admin dashboard"""

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get counts
        cursor.execute("SELECT COUNT(*) as count FROM people")
        people_count = cursor.fetchone()['count']

        cursor.execute("SELECT COUNT(*) as count FROM projects")
        projects_count = cursor.fetchone()['count']

        cursor.execute("SELECT COUNT(*) as count FROM skills")
        skills_count = cursor.fetchone()['count']

        cursor.execute("SELECT COUNT(*) as count FROM person_skills")
        skill_assignments_count = cursor.fetchone()['count']

        # Get recent activity
        cursor.execute("""
            SELECT 'project' as type, name as description,
                   date(start_date) as activity_date
            FROM projects
            ORDER BY start_date DESC
            LIMIT 5
        """)
        recent_activity = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return {
            "system_health": "healthy",
            "database": {
                "people": people_count,
                "projects": projects_count,
                "skills": skills_count,
                "skill_assignments": skill_assignments_count
            },
            "recent_activity": recent_activity,
            "last_updated": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System status check failed: {str(e)}")

# Export the router
__all__ = ['admin_router']