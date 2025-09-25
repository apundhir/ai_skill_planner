#!/usr/bin/env python3
"""
Gap Analysis API Endpoints
Extends the main API with gap analysis, proficiency, and capacity endpoints
"""

import sys
import os
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import sqlite3

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.init_db import get_db_connection
from engines.gap_analysis import GapAnalysisEngine
from engines.proficiency import ProficiencyCalculator
from engines.capacity import CapacityModel

# Create router for gap analysis endpoints
gap_router = APIRouter(prefix="/gap-analysis", tags=["Gap Analysis"])

# Initialize engines
gap_engine = GapAnalysisEngine()
proficiency_calc = ProficiencyCalculator()
capacity_model = CapacityModel()

# Pydantic models for responses
class SkillGap(BaseModel):
    project_id: str
    project_name: str
    phase: str
    skill_id: str
    skill_name: str
    skill_category: str
    gap_severity: str
    expected_gap_fte: float
    conservative_gap_fte: float
    optimistic_gap_fte: float
    coverage_ratio: float
    bus_factor: float
    cost_impact_weekly: float
    priority: str

class ProjectGapSummary(BaseModel):
    project_id: str
    project_name: str
    recommendation: str
    project_risk_score: float
    total_gaps: int
    critical_gaps: int
    high_gaps: int
    medium_gaps: int
    total_cost_impact: float

class HeatMapData(BaseModel):
    projects: List[Dict[str, Any]]
    skills: List[Dict[str, Any]]
    gap_matrix: List[List[Dict[str, Any]]]

@gap_router.get("/project/{project_id}/gaps", response_model=Dict[str, Any])
def get_project_gaps(project_id: str, phase: Optional[str] = None):
    """Get comprehensive gap analysis for a project"""
    try:
        analysis = gap_engine.analyze_project_gaps(project_id, phase)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gap analysis failed: {str(e)}")

@gap_router.get("/project/{project_id}/phase/{phase}/skill/{skill_id}")
def get_specific_skill_gap(project_id: str, phase: str, skill_id: str):
    """Get detailed gap analysis for a specific skill"""
    try:
        gap = gap_engine.detect_skill_gap(project_id, phase, skill_id)
        return gap
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Skill gap analysis failed: {str(e)}")

@gap_router.get("/organization/overview")
def get_organization_gaps():
    """Get organization-wide gap overview"""
    try:
        overview = gap_engine.get_organization_gap_overview()
        return overview
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Organization gap analysis failed: {str(e)}")

@gap_router.get("/heatmap", response_model=HeatMapData)
def get_gap_heatmap(project_ids: Optional[str] = Query(None, description="Comma-separated project IDs")):
    """
    Generate heat map data showing skill gaps across projects
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get projects to analyze
        if project_ids:
            project_list = project_ids.split(",")
            placeholders = ",".join(["?" for _ in project_list])
            cursor.execute(f"""
                SELECT id, name, cost_of_delay_weekly
                FROM projects
                WHERE id IN ({placeholders})
                ORDER BY cost_of_delay_weekly DESC
            """, project_list)
        else:
            cursor.execute("""
                SELECT id, name, cost_of_delay_weekly
                FROM projects
                ORDER BY cost_of_delay_weekly DESC
            """)

        projects = [dict(row) for row in cursor.fetchall()]

        # Get all skills that have requirements
        cursor.execute("""
            SELECT DISTINCT s.id, s.name, s.category
            FROM skills s
            JOIN project_requirements pr ON s.id = pr.skill_id
            ORDER BY s.category, s.name
        """)

        skills = [dict(row) for row in cursor.fetchall()]

        # Generate simplified gap matrix based on requirements vs available skills
        gap_matrix = []
        for project in projects:
            project_row = []

            # Get project requirements
            cursor.execute("""
                SELECT pr.skill_id, pr.required_level, pr.min_level, pr.fte_weeks, pr.criticality,
                       s.name as skill_name
                FROM project_requirements pr
                JOIN skills s ON pr.skill_id = s.id
                WHERE pr.project_id = ?
            """, (project['id'],))

            requirements = {row['skill_id']: dict(row) for row in cursor.fetchall()}

            # Get available team capacity for this project's skills
            for skill in skills:
                skill_id = skill['id']

                if skill_id in requirements:
                    req = requirements[skill_id]

                    # Get team members with this skill
                    cursor.execute("""
                        SELECT ps.base_level, ps.effective_level, p.fte, p.name as person_name
                        FROM person_skills ps
                        JOIN people p ON ps.person_id = p.id
                        WHERE ps.skill_id = ?
                        ORDER BY COALESCE(ps.effective_level, ps.base_level) DESC
                    """, (skill_id,))

                    team_skills = cursor.fetchall()

                    # Calculate basic gap metrics
                    if team_skills:
                        # Calculate available capacity
                        available_capacity = sum(
                            (row['effective_level'] or row['base_level']) * row['fte']
                            for row in team_skills
                        )
                        required_capacity = req['required_level'] * (req['fte_weeks'] / 52.0)  # Convert to annual FTE

                        coverage_ratio = min(1.0, available_capacity / required_capacity) if required_capacity > 0 else 1.0
                        gap_fte = max(0, required_capacity - available_capacity)

                        # Determine severity based on gap and criticality
                        if coverage_ratio >= 0.9:
                            severity = 'low'
                        elif coverage_ratio >= 0.7:
                            severity = 'medium' if req['criticality'] > 7 else 'low'
                        elif coverage_ratio >= 0.5:
                            severity = 'high'
                        else:
                            severity = 'critical'

                        cost_impact = project['cost_of_delay_weekly'] * (1 - coverage_ratio) * req['criticality'] / 10
                    else:
                        # No team members have this skill
                        coverage_ratio = 0.0
                        gap_fte = req['required_level'] * (req['fte_weeks'] / 52.0)
                        severity = 'critical' if req['criticality'] > 7 else 'high'
                        cost_impact = project['cost_of_delay_weekly'] * req['criticality'] / 10

                    project_row.append({
                        'severity': severity,
                        'coverage_ratio': coverage_ratio,
                        'cost_impact_weekly': cost_impact,
                        'gap_fte': gap_fte,
                        'phase': 'all'  # Simplified
                    })
                else:
                    # Skill not required for this project
                    project_row.append({
                        'severity': 'none',
                        'coverage_ratio': 1.0,
                        'cost_impact_weekly': 0.0,
                        'gap_fte': 0.0,
                        'phase': None
                    })

            gap_matrix.append(project_row)

        conn.close()

        return HeatMapData(
            projects=projects,
            skills=skills,
            gap_matrix=gap_matrix
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Heat map generation failed: {str(e)}")

@gap_router.post("/update-proficiency")
def update_skill_proficiency(person_id: Optional[str] = None):
    """Update skill proficiency calculations"""
    try:
        if person_id:
            updated = proficiency_calc.update_person_skill_levels(person_id)
            return {
                "message": f"Updated {len(updated)} skills for {person_id}",
                "updated_skills": updated
            }
        else:
            stats = proficiency_calc.update_all_skill_levels()
            return {
                "message": "Updated all skill proficiency levels",
                "statistics": stats
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proficiency update failed: {str(e)}")

@gap_router.get("/capacity/{project_id}/phase/{phase}")
def get_phase_capacity(project_id: str, phase: str):
    """Get capacity analysis for a project phase"""
    try:
        analysis = capacity_model.analyze_project_phase(project_id, phase)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Capacity analysis failed: {str(e)}")

@gap_router.get("/proficiency/summary")
def get_proficiency_summary():
    """Get organization skill proficiency summary"""
    try:
        summary = proficiency_calc.get_skill_distribution_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proficiency summary failed: {str(e)}")

@gap_router.get("/top-gaps")
def get_top_gaps(limit: int = Query(10, ge=1, le=50)):
    """Get top skill gaps across all projects"""
    try:
        overview = gap_engine.get_organization_gap_overview()

        # Extract and sort all gaps
        all_gaps = []
        for project_summary in overview['project_summaries']:
            project_id = project_summary['project_id']
            project_analysis = gap_engine.analyze_project_gaps(project_id)

            for gap in project_analysis['top_gaps']:
                if gap['requirement']:
                    all_gaps.append({
                        'project_name': project_analysis['project_info']['name'],
                        'skill_name': gap['skill_name'],
                        'phase': gap['phase'],
                        'severity': gap['gap_analysis']['gap_severity'],
                        'gap_fte': gap['gap_analysis']['expected_gap_fte'],
                        'cost_impact_weekly': gap['business_impact']['cost_impact_weekly'],
                        'coverage_ratio': gap['gap_analysis']['coverage_ratio'],
                        'bus_factor': gap['gap_analysis']['bus_factor']
                    })

        # Sort by cost impact and limit
        top_gaps = sorted(all_gaps, key=lambda x: x['cost_impact_weekly'], reverse=True)[:limit]

        return {
            'top_gaps': top_gaps,
            'total_gaps_analyzed': len(all_gaps),
            'generated_at': overview['generated_at']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Top gaps analysis failed: {str(e)}")

# Export the router
__all__ = ['gap_router']