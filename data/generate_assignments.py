#!/usr/bin/env python3
"""
Generate realistic project assignments with availability constraints
Matches team member skills to project requirements while considering:
- Timeline overlaps and conflicts
- Senior people spread across multiple projects
- Junior people focused on fewer projects
- Realistic availability percentages
"""

import sqlite3
import sys
import os
import random
from datetime import datetime, date
from typing import List, Dict, Any, Set, Tuple

# Add parent directory to path to import database module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.init_db import get_db_connection

def get_person_skills(person_id: str, db_path: str = "ai_skill_planner.db") -> Dict[str, float]:
    """Get all skills for a person with their base levels"""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT skill_id, base_level
        FROM person_skills
        WHERE person_id = ?
    """, (person_id,))

    skills = {row['skill_id']: row['base_level'] for row in cursor.fetchall()}
    conn.close()
    return skills

def get_project_requirements(project_id: str, db_path: str = "ai_skill_planner.db") -> Dict[str, List[Dict]]:
    """Get project requirements organized by phase"""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT phase_name, skill_id, required_level, min_level, fte_weeks, criticality
        FROM project_requirements
        WHERE project_id = ?
        ORDER BY phase_name, criticality DESC
    """, (project_id,))

    requirements = {}
    for row in cursor.fetchall():
        phase = row['phase_name']
        if phase not in requirements:
            requirements[phase] = []
        requirements[phase].append({
            'skill_id': row['skill_id'],
            'required_level': row['required_level'],
            'min_level': row['min_level'],
            'fte_weeks': row['fte_weeks'],
            'criticality': row['criticality']
        })

    conn.close()
    return requirements

def calculate_person_project_fit(person_skills: Dict[str, float],
                               project_requirements: Dict[str, List[Dict]]) -> float:
    """Calculate how well a person fits a project (0.0 to 1.0)"""
    total_score = 0.0
    total_weight = 0.0

    for phase, requirements in project_requirements.items():
        for req in requirements:
            skill_id = req['skill_id']
            required_level = req['required_level']
            min_level = req['min_level']
            criticality = req['criticality']

            if skill_id in person_skills:
                person_level = person_skills[skill_id]
                # Score based on how much above minimum level
                if person_level >= min_level:
                    # Scale score: meeting min = 0.5, meeting required = 1.0
                    if person_level >= required_level:
                        skill_score = 1.0
                    else:
                        skill_score = 0.5 + 0.5 * (person_level - min_level) / (required_level - min_level)
                else:
                    skill_score = 0.0  # Below minimum
            else:
                skill_score = 0.0  # No skill

            weight = criticality * req['fte_weeks']
            total_score += skill_score * weight
            total_weight += weight

    return total_score / total_weight if total_weight > 0 else 0.0

def get_timeline_overlap(project1: Tuple[date, date], project2: Tuple[date, date]) -> float:
    """Calculate overlap percentage between two project timelines"""
    start1, end1 = project1
    start2, end2 = project2

    # Find overlap period
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    if overlap_start > overlap_end:
        return 0.0  # No overlap

    # Calculate overlap as percentage of total period
    overlap_days = (overlap_end - overlap_start).days + 1
    total_days = max((end1 - start1).days + 1, (end2 - start2).days + 1)

    return min(overlap_days / total_days, 1.0)

def generate_project_assignments(db_path: str = "ai_skill_planner.db") -> List[Dict[str, Any]]:
    """Generate realistic project assignments for all team members"""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Get all people with their cost (proxy for seniority)
    cursor.execute("""
        SELECT id, name, cost_hourly, fte
        FROM people
        ORDER BY cost_hourly DESC
    """)
    people = [dict(row) for row in cursor.fetchall()]

    # Get all projects with timelines
    cursor.execute("""
        SELECT id, name, start_date, end_date, complexity, risk_tolerance
        FROM projects
        ORDER BY cost_of_delay_weekly DESC
    """)
    projects = [dict(row) for row in cursor.fetchall()]

    conn.close()

    assignments = []
    person_workload = {}  # Track total allocation per person

    # Initialize workload tracking
    for person in people:
        person_workload[person['id']] = {}
        for project in projects:
            person_workload[person['id']][project['id']] = 0.0

    # Priority assignment: critical projects first, senior people first
    for project in projects:
        print(f"\nAssigning people to {project['name']}...")

        # Get project requirements and timeline
        project_requirements = get_project_requirements(project['id'], db_path)
        project_timeline = (
            datetime.strptime(project['start_date'], '%Y-%m-%d').date(),
            datetime.strptime(project['end_date'], '%Y-%m-%d').date()
        )

        # Calculate person-project fit scores
        person_fits = []
        for person in people:
            person_skills = get_person_skills(person['id'], db_path)
            fit_score = calculate_person_project_fit(person_skills, project_requirements)

            # Adjust fit based on seniority and project complexity
            seniority_bonus = 0.0
            if project['complexity'] == 'high' and person['cost_hourly'] >= 150:
                seniority_bonus = 0.1  # Senior people better for complex projects
            elif project['complexity'] == 'low' and person['cost_hourly'] < 100:
                seniority_bonus = 0.05  # Junior people can handle simple projects

            person_fits.append({
                'person': person,
                'fit_score': fit_score + seniority_bonus,
                'skills': person_skills
            })

        # Sort by fit score (best fits first)
        person_fits.sort(key=lambda x: x['fit_score'], reverse=True)

        # Assign top candidates with realistic availability
        assigned_count = 0
        target_assignments = random.randint(4, 8)  # Realistic team size

        for person_fit in person_fits:
            if assigned_count >= target_assignments:
                break

            person = person_fit['person']
            fit_score = person_fit['fit_score']

            # Skip if person has no relevant skills
            if fit_score < 0.2:
                continue

            # Calculate realistic availability based on:
            # 1. Person's base FTE
            # 2. Existing commitments
            # 3. Seniority (senior people spread thinner)
            # 4. Project timeline overlaps

            base_availability = person['fte']

            # Check timeline conflicts with existing assignments
            timeline_conflicts = 0.0
            for other_project_id, allocation in person_workload[person['id']].items():
                if allocation > 0 and other_project_id != project['id']:
                    # Get other project timeline
                    other_project = next(p for p in projects if p['id'] == other_project_id)
                    other_timeline = (
                        datetime.strptime(other_project['start_date'], '%Y-%m-%d').date(),
                        datetime.strptime(other_project['end_date'], '%Y-%m-%d').date()
                    )
                    overlap = get_timeline_overlap(project_timeline, other_timeline)
                    timeline_conflicts += overlap * allocation

            available_capacity = base_availability - timeline_conflicts

            if available_capacity <= 0.1:  # Less than 10% available
                continue

            # Determine allocation based on fit, seniority, and project needs
            if person['cost_hourly'] >= 160:  # Senior/Principal
                # Senior people: 20-60% on projects (spread across multiple)
                allocation = min(random.uniform(0.2, 0.6), available_capacity)
            elif person['cost_hourly'] >= 120:  # Mid-level
                # Mid-level: 40-80% on projects
                allocation = min(random.uniform(0.4, 0.8), available_capacity)
            else:  # Junior
                # Junior: 60-100% on fewer projects
                allocation = min(random.uniform(0.6, 1.0), available_capacity)

            # Boost allocation for high fit scores
            if fit_score > 0.7:
                allocation = min(allocation * 1.2, available_capacity)

            # Final allocation (minimum 10% to be meaningful)
            final_allocation = max(0.1, min(allocation, available_capacity))

            # Record assignment
            person_workload[person['id']][project['id']] = final_allocation

            # Create assignment for each project phase
            # (Simplified: same availability across all phases)
            phases = ['discovery', 'data_prep', 'modeling', 'deployment', 'monitoring']
            for phase in phases:
                if phase in project_requirements:  # Only assign to phases that exist
                    assignments.append({
                        'project_id': project['id'],
                        'person_id': person['id'],
                        'phase_name': phase,
                        'availability': final_allocation,
                        'start_date': project['start_date'],
                        'end_date': project['end_date'],
                        'fit_score': fit_score
                    })

            assigned_count += 1
            print(f"  {person['name']}: {final_allocation:.1%} allocation (fit: {fit_score:.2f})")

    return assignments

def populate_assignments(db_path: str = "ai_skill_planner.db") -> None:
    """Populate assignments table with realistic data"""
    assignments = generate_project_assignments(db_path)

    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Clear existing assignments
    cursor.execute("DELETE FROM assignments")

    # Insert new assignments
    for assignment in assignments:
        cursor.execute("""
            INSERT INTO assignments (project_id, person_id, phase_name, availability, start_date, end_date)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (assignment['project_id'], assignment['person_id'], assignment['phase_name'],
              assignment['availability'], assignment['start_date'], assignment['end_date']))

    conn.commit()

    # Show summary statistics
    cursor.execute("SELECT COUNT(*) FROM assignments")
    assignment_count = cursor.fetchone()[0]

    print(f"\n📊 Assignment Summary:")
    print(f"Total assignments: {assignment_count}")

    # Show workload distribution
    cursor.execute("""
        SELECT p.name, p.cost_hourly,
               COUNT(DISTINCT a.project_id) as project_count,
               ROUND(SUM(a.availability), 2) as total_allocation
        FROM people p
        LEFT JOIN assignments a ON p.id = a.person_id
        GROUP BY p.id
        HAVING total_allocation > 0
        ORDER BY p.cost_hourly DESC
    """)

    print(f"\nWorkload by person:")
    for row in cursor.fetchall():
        print(f"  {row['name']} (${row['cost_hourly']}/hr): {row['project_count']} projects, {row['total_allocation']:.1f} total allocation")

    # Show project staffing
    cursor.execute("""
        SELECT pr.name, COUNT(DISTINCT a.person_id) as team_size,
               ROUND(AVG(a.availability), 2) as avg_availability
        FROM projects pr
        LEFT JOIN assignments a ON pr.id = a.project_id
        GROUP BY pr.id
        ORDER BY team_size DESC
    """)

    print(f"\nProject staffing:")
    for row in cursor.fetchall():
        print(f"  {row['name']}: {row['team_size']} people (avg {row['avg_availability']:.1%} availability)")

    conn.close()

if __name__ == "__main__":
    # Generate assignments when run as script
    populate_assignments()

    # Test query: Show sample assignments
    conn = get_db_connection()
    cursor = conn.cursor()

    print(f"\n=== Sample Assignments ===")
    cursor.execute("""
        SELECT p.name, pr.name as project_name, a.phase_name, a.availability
        FROM assignments a
        JOIN people p ON a.person_id = p.id
        JOIN projects pr ON a.project_id = pr.id
        WHERE pr.id = 'personalization_engine' AND a.phase_name = 'modeling'
        ORDER BY a.availability DESC
        LIMIT 5
    """)

    for row in cursor.fetchall():
        print(f"  {row['name']}: {row['availability']:.1%} on {row['project_name']} ({row['phase_name']})")

    conn.close()