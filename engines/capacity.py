#!/usr/bin/env python3
"""
Capacity Model for Supply/Demand Analysis
Implements capacity-aware coverage calculations that consider:
- Actual team availability with assignment constraints
- Skill level matching to requirements
- Bus factor calculations for risk assessment
- Supply vs demand analysis with realistic FTE calculations
"""

import sys
import os
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Tuple
import sqlite3

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.init_db import get_db_connection


class CapacityModel:
    """
    Models actual capacity considering availability constraints and skill matching
    Provides supply/demand analysis with bus factor calculations
    """

    def __init__(self, db_path: str = "ai_skill_planner.db"):
        """Initialize the capacity model"""
        self.db_path = db_path

    def get_requirement(self, project_id: str, phase: str, skill_id: str,
                       conn: Optional[sqlite3.Connection] = None) -> Optional[Dict[str, Any]]:
        """Get skill requirement for a specific project phase"""
        should_close_conn = conn is None
        if conn is None:
            conn = get_db_connection(self.db_path)

        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT required_level, min_level, fte_weeks, criticality
                FROM project_requirements
                WHERE project_id = ? AND phase_name = ? AND skill_id = ?
            """, (project_id, phase, skill_id))

            result = cursor.fetchone()
            return dict(result) if result else None

        finally:
            if should_close_conn:
                conn.close()

    def get_assignments(self, project_id: str, phase: str,
                       conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
        """Get all assignments for a project phase"""
        should_close_conn = conn is None
        if conn is None:
            conn = get_db_connection(self.db_path)

        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT a.person_id, a.availability, p.fte,
                       p.name, p.cost_hourly
                FROM assignments a
                JOIN people p ON a.person_id = p.id
                WHERE a.project_id = ? AND a.phase_name = ?
            """, (project_id, phase))

            return [dict(row) for row in cursor.fetchall()]

        finally:
            if should_close_conn:
                conn.close()

    def calculate_coverage(self, project_id: str, phase: str, skill_id: str,
                          conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        """
        Models actual capacity considering availability and assignments
        Returns comprehensive coverage analysis
        """
        should_close_conn = conn is None
        if conn is None:
            conn = get_db_connection(self.db_path)

        try:
            cursor = conn.cursor()

            # Get requirement for this skill
            requirement = self.get_requirement(project_id, phase, skill_id, conn)
            if not requirement:
                return {
                    'coverage_ratio': 0.0,
                    'bus_factor': 0.0,
                    'total_supply_fte': 0.0,
                    'total_demand_fte': 0.0,
                    'contributors': [],
                    'gap_fte': 0.0,
                    'requirement': None
                }

            # Get assignments for this phase
            assignments = self.get_assignments(project_id, phase, conn)

            # Calculate each person's contribution
            contributions = []
            for assignment in assignments:
                person_id = assignment['person_id']

                # Get person's skill level for this skill
                cursor.execute("""
                    SELECT base_level, effective_level, confidence_low, confidence_high
                    FROM person_skills
                    WHERE person_id = ? AND skill_id = ?
                """, (person_id, skill_id))

                skill_result = cursor.fetchone()
                if not skill_result:
                    continue  # Person doesn't have this skill

                skill_level = dict(skill_result)

                # Use effective level if available, otherwise base level
                person_level = skill_level.get('effective_level') or skill_level['base_level']

                # Only count if person meets minimum level requirement
                if person_level >= requirement['min_level']:
                    # Calculate contribution considering availability and skill match
                    skill_match_ratio = min(person_level / requirement['required_level'], 1.0)

                    # Effective FTE contribution
                    contribution_fte = (
                        skill_match_ratio *  # How well skill level matches requirement
                        assignment['availability'] *  # Assignment availability (0-1)
                        assignment['fte']  # Person's overall FTE
                    )

                    # Determine if this person is critical (meets full requirement)
                    is_critical = person_level >= requirement['required_level']

                    contributions.append({
                        'person_id': person_id,
                        'person_name': assignment['name'],
                        'person_level': person_level,
                        'skill_match_ratio': skill_match_ratio,
                        'availability': assignment['availability'],
                        'fte': assignment['fte'],
                        'contribution_fte': contribution_fte,
                        'is_critical': is_critical,
                        'cost_hourly': assignment['cost_hourly']
                    })

            # Calculate metrics
            total_supply_fte = sum(c['contribution_fte'] for c in contributions)
            total_demand_fte = requirement['fte_weeks']
            coverage_ratio = total_supply_fte / total_demand_fte if total_demand_fte > 0 else 0

            # Bus factor calculation (risk if critical people leave)
            critical_people = [c for c in contributions if c['is_critical']]
            bus_factor = 1.0 / max(len(critical_people), 1)  # Higher = more risky

            # Gap analysis
            gap_fte = max(0, total_demand_fte - total_supply_fte)

            return {
                'coverage_ratio': round(coverage_ratio, 2),
                'bus_factor': round(bus_factor, 2),
                'total_supply_fte': round(total_supply_fte, 2),
                'total_demand_fte': total_demand_fte,
                'contributors': contributions,
                'gap_fte': round(gap_fte, 2),
                'requirement': requirement,
                'critical_people_count': len(critical_people),
                'total_people_assigned': len(contributions)
            }

        finally:
            if should_close_conn:
                conn.close()

    def analyze_project_phase(self, project_id: str, phase: str,
                             conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of all skills for a project phase
        """
        should_close_conn = conn is None
        if conn is None:
            conn = get_db_connection(self.db_path)

        try:
            cursor = conn.cursor()

            # Get project information
            cursor.execute("""
                SELECT name, cost_of_delay_weekly, risk_tolerance
                FROM projects
                WHERE id = ?
            """, (project_id,))

            project_info = dict(cursor.fetchone())

            # Get phase information
            cursor.execute("""
                SELECT start_date, end_date, gate_threshold
                FROM phases
                WHERE project_id = ? AND phase_name = ?
            """, (project_id, phase))

            phase_info = dict(cursor.fetchone())

            # Get all skill requirements for this phase
            cursor.execute("""
                SELECT skill_id, s.name as skill_name, s.category,
                       pr.required_level, pr.min_level, pr.fte_weeks, pr.criticality
                FROM project_requirements pr
                JOIN skills s ON pr.skill_id = s.id
                WHERE pr.project_id = ? AND pr.phase_name = ?
                ORDER BY pr.criticality DESC, pr.fte_weeks DESC
            """, (project_id, phase))

            skill_requirements = [dict(row) for row in cursor.fetchall()]

            # Analyze coverage for each skill
            skill_analyses = []
            total_demand_fte = 0
            total_supply_fte = 0
            total_gap_fte = 0
            high_risk_skills = 0
            under_coverage_skills = 0

            for req in skill_requirements:
                skill_id = req['skill_id']
                coverage = self.calculate_coverage(project_id, phase, skill_id, conn)

                # Add requirement details to coverage
                coverage['skill_name'] = req['skill_name']
                coverage['skill_category'] = req['category']
                coverage['criticality'] = req['criticality']

                # Classify risk level
                risk_level = 'low'
                if coverage['coverage_ratio'] < 0.5 or coverage['bus_factor'] > 0.5:
                    risk_level = 'high'
                    high_risk_skills += 1
                elif coverage['coverage_ratio'] < 0.8 or coverage['bus_factor'] > 0.33:
                    risk_level = 'medium'

                if coverage['coverage_ratio'] < 1.0:
                    under_coverage_skills += 1

                coverage['risk_level'] = risk_level

                skill_analyses.append(coverage)

                # Accumulate totals
                total_demand_fte += coverage['total_demand_fte']
                total_supply_fte += coverage['total_supply_fte']
                total_gap_fte += coverage['gap_fte']

            # Calculate phase-level metrics
            phase_coverage_ratio = total_supply_fte / total_demand_fte if total_demand_fte > 0 else 0

            # Determine gate status based on coverage and risk
            gate_status = 'GO'
            if high_risk_skills > 0 or phase_coverage_ratio < phase_info['gate_threshold']:
                if phase_coverage_ratio < 0.5 or high_risk_skills >= 3:
                    gate_status = 'NO-GO'
                else:
                    gate_status = 'CONDITIONAL'

            return {
                'project_info': project_info,
                'phase_info': phase_info,
                'skill_analyses': skill_analyses,
                'phase_summary': {
                    'total_skills': len(skill_requirements),
                    'total_demand_fte': round(total_demand_fte, 2),
                    'total_supply_fte': round(total_supply_fte, 2),
                    'total_gap_fte': round(total_gap_fte, 2),
                    'phase_coverage_ratio': round(phase_coverage_ratio, 2),
                    'high_risk_skills': high_risk_skills,
                    'under_coverage_skills': under_coverage_skills,
                    'gate_status': gate_status,
                    'gate_threshold': phase_info['gate_threshold']
                },
                'generated_at': datetime.now().isoformat()
            }

        finally:
            if should_close_conn:
                conn.close()

    def analyze_full_project(self, project_id: str,
                           conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        """
        Complete project analysis across all phases
        """
        should_close_conn = conn is None
        if conn is None:
            conn = get_db_connection(self.db_path)

        try:
            cursor = conn.cursor()

            # Get all phases for this project
            cursor.execute("""
                SELECT phase_name
                FROM phases
                WHERE project_id = ?
                ORDER BY start_date
            """, (project_id,))

            phases = [row['phase_name'] for row in cursor.fetchall()]

            # Analyze each phase
            phase_analyses = {}
            overall_risk_score = 0
            total_no_go_phases = 0

            for phase in phases:
                analysis = self.analyze_project_phase(project_id, phase, conn)
                phase_analyses[phase] = analysis

                # Contribute to overall risk score
                phase_summary = analysis['phase_summary']
                phase_risk = (
                    1.0 - phase_summary['phase_coverage_ratio'] +
                    phase_summary['high_risk_skills'] * 0.2 +
                    (1 if phase_summary['gate_status'] == 'NO-GO' else 0) * 0.5
                )
                overall_risk_score += phase_risk

                if phase_summary['gate_status'] == 'NO-GO':
                    total_no_go_phases += 1

            # Calculate project-level metrics
            overall_risk_score = overall_risk_score / len(phases) if phases else 0
            project_status = 'GO'
            if total_no_go_phases > 0:
                project_status = 'NO-GO'
            elif overall_risk_score > 0.6:
                project_status = 'CONDITIONAL'

            return {
                'project_id': project_id,
                'phase_analyses': phase_analyses,
                'project_summary': {
                    'total_phases': len(phases),
                    'no_go_phases': total_no_go_phases,
                    'overall_risk_score': round(overall_risk_score, 2),
                    'project_status': project_status
                },
                'generated_at': datetime.now().isoformat()
            }

        finally:
            if should_close_conn:
                conn.close()

    def get_organization_capacity_summary(self,
                                        conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        """
        Organization-wide capacity analysis
        """
        should_close_conn = conn is None
        if conn is None:
            conn = get_db_connection(self.db_path)

        try:
            cursor = conn.cursor()

            # Get all active projects
            cursor.execute("SELECT id, name FROM projects ORDER BY cost_of_delay_weekly DESC")
            projects = [dict(row) for row in cursor.fetchall()]

            # Analyze each project
            project_summaries = []
            overall_risk_projects = 0

            for project in projects:
                project_analysis = self.analyze_full_project(project['id'], conn)
                project_summary = project_analysis['project_summary']
                project_summary['project_name'] = project['name']

                if project_summary['project_status'] in ['NO-GO', 'CONDITIONAL']:
                    overall_risk_projects += 1

                project_summaries.append(project_summary)

            # Overall capacity utilization
            cursor.execute("""
                SELECT COUNT(DISTINCT person_id) as people_assigned,
                       COUNT(DISTINCT project_id) as projects_with_assignments,
                       ROUND(AVG(availability), 2) as avg_availability
                FROM assignments
            """)

            utilization_stats = dict(cursor.fetchone())

            cursor.execute("SELECT COUNT(*) FROM people")
            total_people = cursor.fetchone()[0]

            utilization_stats['utilization_rate'] = round(
                utilization_stats['people_assigned'] / total_people * 100, 1
            ) if total_people > 0 else 0

            return {
                'project_summaries': project_summaries,
                'organization_stats': {
                    'total_projects': len(projects),
                    'risk_projects': overall_risk_projects,
                    'healthy_projects': len(projects) - overall_risk_projects,
                    **utilization_stats
                },
                'generated_at': datetime.now().isoformat()
            }

        finally:
            if should_close_conn:
                conn.close()


if __name__ == "__main__":
    # Test the capacity model
    capacity = CapacityModel()

    print("🏗️  Testing Capacity Model...")

    # Test single skill coverage
    print("\n🔍 Testing individual skill coverage...")
    coverage = capacity.calculate_coverage(
        'personalization_engine', 'modeling', 'recommendation_systems'
    )

    print(f"Recommendation Systems Coverage for Personalization Engine (Modeling):")
    print(f"  Coverage Ratio: {coverage['coverage_ratio']} ({coverage['total_supply_fte']:.1f} FTE supply vs {coverage['total_demand_fte']:.1f} demand)")
    print(f"  Bus Factor: {coverage['bus_factor']} ({coverage['critical_people_count']} critical people)")
    print(f"  Gap: {coverage['gap_fte']:.1f} FTE")
    print(f"  Contributors: {coverage['total_people_assigned']} people assigned")

    # Test phase analysis
    print(f"\n📊 Testing phase analysis...")
    phase_analysis = capacity.analyze_project_phase('personalization_engine', 'modeling')

    summary = phase_analysis['phase_summary']
    print(f"Personalization Engine - Modeling Phase:")
    print(f"  Gate Status: {summary['gate_status']} (threshold: {summary['gate_threshold']})")
    print(f"  Phase Coverage: {summary['phase_coverage_ratio']} ({summary['total_supply_fte']} vs {summary['total_demand_fte']} FTE)")
    print(f"  High Risk Skills: {summary['high_risk_skills']}/{summary['total_skills']}")
    print(f"  Under-covered Skills: {summary['under_coverage_skills']}/{summary['total_skills']}")

    # Show top 3 riskiest skills
    risky_skills = [s for s in phase_analysis['skill_analyses'] if s['risk_level'] == 'high']
    if risky_skills:
        print(f"\n⚠️  High Risk Skills:")
        for skill in risky_skills[:3]:
            print(f"    {skill['skill_name']}: {skill['coverage_ratio']} coverage, {skill['bus_factor']} bus factor")

    # Test full project analysis
    print(f"\n🎯 Testing full project analysis...")
    project_analysis = capacity.analyze_full_project('personalization_engine')

    proj_summary = project_analysis['project_summary']
    print(f"Personalization Engine - Full Project:")
    print(f"  Project Status: {proj_summary['project_status']}")
    print(f"  Risk Score: {proj_summary['overall_risk_score']}")
    print(f"  NO-GO Phases: {proj_summary['no_go_phases']}/{proj_summary['total_phases']}")

    # Organization summary
    print(f"\n🏢 Testing organization capacity summary...")
    org_summary = capacity.get_organization_capacity_summary()

    org_stats = org_summary['organization_stats']
    print(f"Organization Capacity Summary:")
    print(f"  Total Projects: {org_stats['total_projects']}")
    print(f"  Risk Projects: {org_stats['risk_projects']}")
    print(f"  People Utilization: {org_stats['utilization_rate']}% ({org_stats['people_assigned']} assigned)")
    print(f"  Average Availability: {org_stats['avg_availability']}")

    print(f"\n✅ Capacity analysis complete!")