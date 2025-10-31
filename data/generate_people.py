#!/usr/bin/env python3
"""
Generate realistic team members for AI Skill Planner
Creates 20 diverse team members with varying skills and experience levels
"""

import sqlite3
import sys
import os
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Add parent directory to path to import database module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.init_db import get_db_connection

# Realistic team member profiles for a 20-person AI company
TEAM_MEMBERS = [
    # SENIOR LEADERSHIP & PRINCIPAL ENGINEERS
    {
        "id": "sarah_chen",
        "name": "Sarah Chen",
        "location": "San Francisco, CA",
        "timezone": "America/Los_Angeles",
        "fte": 1.0,
        "cost_hourly": 180.0,
        "role": "VP Engineering",
        "experience_years": 12,
        "skills": {
            "technical_strategy": 4.8, "technical_mentoring": 4.6, "hiring_interviewing": 4.5,
            "system_design": 4.2, "project_management": 4.4, "stakeholder_communication": 4.7,
            "ml_supervised": 3.8, "python_advanced": 4.0, "business_metrics": 4.2
        }
    },
    {
        "id": "alex_rodriguez",
        "name": "Alex Rodriguez",
        "location": "New York, NY",
        "timezone": "America/New_York",
        "fte": 1.0,
        "cost_hourly": 165.0,
        "role": "Principal ML Engineer",
        "experience_years": 9,
        "skills": {
            "deep_learning": 4.7, "nlp": 4.5, "llm_fine_tuning": 4.3, "research_methodology": 4.2,
            "paper_implementation": 4.4, "python_advanced": 4.6, "model_deployment": 4.0,
            "technical_mentoring": 4.1, "experimentation": 4.3
        }
    },

    # SENIOR ENGINEERS
    {
        "id": "priya_sharma",
        "name": "Priya Sharma",
        "location": "London, UK",
        "timezone": "Europe/London",
        "fte": 1.0,
        "cost_hourly": 145.0,
        "role": "Senior ML Engineer",
        "experience_years": 7,
        "skills": {
            "computer_vision": 4.3, "deep_learning": 4.1, "model_deployment": 4.2,
            "containerization": 4.0, "python_advanced": 4.2, "ml_pipelines": 3.9,
            "api_development": 3.8, "testing_ml": 3.7
        }
    },
    {
        "id": "michael_wong",
        "name": "Michael Wong",
        "location": "Toronto, ON",
        "timezone": "America/Toronto",
        "fte": 1.0,
        "cost_hourly": 140.0,
        "role": "Senior MLOps Engineer",
        "experience_years": 6,
        "skills": {
            "ml_pipelines": 4.4, "model_monitoring": 4.3, "containerization": 4.5,
            "cloud_ml": 4.2, "model_versioning": 4.1, "system_design": 3.9,
            "microservices": 4.0, "database_design": 3.8
        }
    },
    {
        "id": "elena_petrov",
        "name": "Elena Petrov",
        "location": "Berlin, Germany",
        "timezone": "Europe/Berlin",
        "fte": 1.0,
        "cost_hourly": 135.0,
        "role": "Senior Data Engineer",
        "experience_years": 8,
        "skills": {
            "data_pipelines": 4.5, "sql_advanced": 4.6, "spark": 4.3, "etl_design": 4.4,
            "streaming_data": 4.0, "data_warehousing": 4.2, "python_advanced": 4.0,
            "system_design": 3.9
        }
    },

    # MID-LEVEL ENGINEERS
    {
        "id": "david_kim",
        "name": "David Kim",
        "location": "Austin, TX",
        "timezone": "America/Chicago",
        "fte": 1.0,
        "cost_hourly": 125.0,
        "role": "ML Engineer",
        "experience_years": 4,
        "skills": {
            "ml_supervised": 3.8, "ml_unsupervised": 3.6, "time_series": 4.0,
            "python_advanced": 3.7, "model_deployment": 3.5, "api_development": 3.4,
            "experimentation": 3.6, "sql_advanced": 3.5
        }
    },
    {
        "id": "aisha_okafor",
        "name": "Aisha Okafor",
        "location": "Remote",
        "timezone": "America/New_York",
        "fte": 1.0,
        "cost_hourly": 120.0,
        "role": "NLP Engineer",
        "experience_years": 5,
        "skills": {
            "nlp": 4.2, "llm_fine_tuning": 3.9, "deep_learning": 3.7, "python_advanced": 3.8,
            "model_deployment": 3.6, "ab_testing": 3.4, "research_methodology": 3.5,
            "api_development": 3.3
        }
    },
    {
        "id": "carlos_mendoza",
        "name": "Carlos Mendoza",
        "location": "Mexico City, Mexico",
        "timezone": "America/Mexico_City",
        "fte": 1.0,
        "cost_hourly": 95.0,
        "role": "MLOps Engineer",
        "experience_years": 4,
        "skills": {
            "containerization": 3.9, "cloud_ml": 3.7, "model_monitoring": 3.8,
            "ml_pipelines": 3.6, "api_development": 3.5, "database_design": 3.4,
            "testing_ml": 3.7, "model_versioning": 3.8
        }
    },
    {
        "id": "jennifer_liu",
        "name": "Jennifer Liu",
        "location": "Seattle, WA",
        "timezone": "America/Los_Angeles",
        "fte": 1.0,
        "cost_hourly": 130.0,
        "role": "Data Engineer",
        "experience_years": 5,
        "skills": {
            "data_pipelines": 3.9, "sql_advanced": 4.1, "etl_design": 3.8,
            "python_advanced": 3.6, "spark": 3.7, "streaming_data": 3.5,
            "database_design": 3.8, "api_development": 3.4
        }
    },
    {
        "id": "raj_patel",
        "name": "Raj Patel",
        "location": "Bangalore, India",
        "timezone": "Asia/Kolkata",
        "fte": 1.0,
        "cost_hourly": 85.0,
        "role": "Computer Vision Engineer",
        "experience_years": 4,
        "skills": {
            "computer_vision": 4.1, "deep_learning": 3.8, "python_advanced": 3.7,
            "model_deployment": 3.5, "containerization": 3.3, "api_development": 3.4,
            "testing_ml": 3.6, "experimentation": 3.5
        }
    },

    # JUNIOR/GROWING ENGINEERS
    {
        "id": "sophie_dubois",
        "name": "Sophie Dubois",
        "location": "Paris, France",
        "timezone": "Europe/Paris",
        "fte": 1.0,
        "cost_hourly": 105.0,
        "role": "Junior ML Engineer",
        "experience_years": 2,
        "skills": {
            "ml_supervised": 3.2, "python_advanced": 3.4, "sql_advanced": 3.0,
            "experimentation": 3.1, "api_development": 2.8, "model_deployment": 2.9,
            "testing_ml": 3.0, "containerization": 2.7
        }
    },
    {
        "id": "tommy_nguyen",
        "name": "Tommy Nguyen",
        "location": "Ho Chi Minh City, Vietnam",
        "timezone": "Asia/Ho_Chi_Minh",
        "fte": 1.0,
        "cost_hourly": 70.0,
        "role": "Junior Data Engineer",
        "experience_years": 2,
        "skills": {
            "sql_advanced": 3.3, "data_pipelines": 3.0, "python_advanced": 3.1,
            "etl_design": 2.9, "database_design": 3.2, "api_development": 2.8,
            "containerization": 2.6, "spark": 2.8
        }
    },

    # SPECIALISTS & RESEARCHERS
    {
        "id": "maria_gonzalez",
        "name": "Maria Gonzalez",
        "location": "Barcelona, Spain",
        "timezone": "Europe/Madrid",
        "fte": 1.0,
        "cost_hourly": 125.0,
        "role": "Research Scientist",
        "experience_years": 6,
        "skills": {
            "research_methodology": 4.3, "paper_implementation": 4.2, "reinforcement_learning": 4.0,
            "deep_learning": 3.9, "experimentation": 4.1, "python_advanced": 3.8,
            "technical_mentoring": 3.6, "stakeholder_communication": 3.4
        }
    },
    {
        "id": "james_taylor",
        "name": "James Taylor",
        "location": "London, UK",
        "timezone": "Europe/London",
        "fte": 1.0,
        "cost_hourly": 150.0,
        "role": "ML Product Manager",
        "experience_years": 7,
        "skills": {
            "product_sense": 4.4, "business_metrics": 4.2, "roi_analysis": 4.1,
            "stakeholder_communication": 4.5, "project_management": 4.3, "ab_testing": 3.8,
            "ml_supervised": 3.2, "sql_advanced": 3.5
        }
    },
    {
        "id": "yuki_tanaka",
        "name": "Yuki Tanaka",
        "location": "Tokyo, Japan",
        "timezone": "Asia/Tokyo",
        "fte": 1.0,
        "cost_hourly": 115.0,
        "role": "Recommendation Systems Engineer",
        "experience_years": 5,
        "skills": {
            "recommendation_systems": 4.2, "personalization": 4.0, "ml_supervised": 3.8,
            "deep_learning": 3.6, "python_advanced": 3.7, "ab_testing": 3.9,
            "sql_advanced": 3.8, "api_development": 3.5
        }
    },

    # CONTRACT/PART-TIME SPECIALISTS
    {
        "id": "robert_miller",
        "name": "Robert Miller",
        "location": "Chicago, IL",
        "timezone": "America/Chicago",
        "fte": 0.6,  # Part-time
        "cost_hourly": 175.0,
        "role": "ML Consultant (Risk Modeling)",
        "experience_years": 11,
        "skills": {
            "risk_modeling": 4.6, "fraud_detection": 4.3, "time_series": 4.2,
            "ml_supervised": 4.1, "business_metrics": 4.0, "stakeholder_communication": 4.2,
            "python_advanced": 3.9, "sql_advanced": 4.0
        }
    },
    {
        "id": "nina_kowalski",
        "name": "Nina Kowalski",
        "location": "Warsaw, Poland",
        "timezone": "Europe/Warsaw",
        "fte": 0.8,  # 4 days/week
        "cost_hourly": 110.0,
        "role": "MLOps Specialist",
        "experience_years": 4,
        "skills": {
            "model_monitoring": 3.9, "containerization": 4.0, "cloud_ml": 3.8,
            "ml_pipelines": 3.7, "model_versioning": 3.8, "testing_ml": 3.6,
            "api_development": 3.5, "microservices": 3.4
        }
    },

    # FRESH GRADUATES & INTERNS
    {
        "id": "zara_ahmed",
        "name": "Zara Ahmed",
        "location": "Dubai, UAE",
        "timezone": "Asia/Dubai",
        "fte": 1.0,
        "cost_hourly": 75.0,
        "role": "Graduate ML Engineer",
        "experience_years": 1,
        "skills": {
            "ml_supervised": 2.8, "deep_learning": 2.9, "python_advanced": 3.0,
            "sql_advanced": 2.6, "experimentation": 2.7, "api_development": 2.5,
            "containerization": 2.3, "testing_ml": 2.6
        }
    },
    {
        "id": "lucas_silva",
        "name": "Lucas Silva",
        "location": "São Paulo, Brazil",
        "timezone": "America/Sao_Paulo",
        "fte": 1.0,
        "cost_hourly": 65.0,
        "role": "Junior Data Scientist",
        "experience_years": 1,
        "skills": {
            "ml_supervised": 2.9, "ml_unsupervised": 2.7, "python_advanced": 2.8,
            "sql_advanced": 2.9, "experimentation": 2.8, "research_methodology": 2.6,
            "time_series": 2.5, "business_metrics": 2.4
        }
    },
    {
        "id": "kevin_zhang",
        "name": "Kevin Zhang",
        "location": "Vancouver, BC",
        "timezone": "America/Vancouver",
        "fte": 1.0,
        "cost_hourly": 90.0,
        "role": "Software Engineer (ML Focus)",
        "experience_years": 3,
        "skills": {
            "python_advanced": 3.6, "api_development": 3.7, "microservices": 3.5,
            "database_design": 3.4, "testing_ml": 3.6, "containerization": 3.3,
            "ml_supervised": 3.0, "system_design": 3.2
        }
    }
]

def generate_person_skills(person: Dict[str, Any], database: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Generate person_skills records with realistic last_used dates and rating info

    Args:
        person: Person dictionary with skills
        database: Optional database URL or path override

    Returns:
        List of person_skills records
    """
    person_skills = []

    # Get all available skills from database
    conn = get_db_connection(database)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM skills")
    all_skills = [row['id'] for row in cursor.fetchall()]
    conn.close()

    for skill_id, base_level in person['skills'].items():
        if skill_id in all_skills:
            # Generate realistic last_used date (more recent for higher skill levels)
            days_ago = random.randint(
                max(1, int(60 - base_level * 10)),  # Higher skills used more recently
                min(365, int(120 - base_level * 5))
            )
            last_used = datetime.now() - timedelta(days=days_ago)

            # Generate rating metadata
            rating_date = last_used + timedelta(days=random.randint(0, 30))
            rater_options = ["sarah_chen", "alex_rodriguez", "self_assessment", "peer_review"]
            rater_id = random.choice(rater_options)

            person_skills.append({
                'person_id': person['id'],
                'skill_id': skill_id,
                'base_level': base_level,
                'effective_level': None,  # Will be calculated by ProficiencyCalculator
                'confidence_low': None,   # Will be calculated
                'confidence_high': None,  # Will be calculated
                'last_used': last_used.date(),
                'rater_id': rater_id,
                'rating_date': rating_date.date()
            })

    return person_skills

def populate_people_and_skills(database: Optional[str] = None) -> None:
    """
    Populate people and person_skills tables

    Args:
        database: Optional database URL or path override
    """
    conn = get_db_connection(database)
    cursor = conn.cursor()

    # Clear existing data
    cursor.execute("DELETE FROM person_skills")
    cursor.execute("DELETE FROM people")

    # Insert people
    for person in TEAM_MEMBERS:
        cursor.execute("""
            INSERT INTO people (id, name, location, timezone, fte, cost_hourly)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (person['id'], person['name'], person['location'],
              person['timezone'], person['fte'], person['cost_hourly']))

        # Generate and insert person skills
        skills = generate_person_skills(person, database)
        for skill in skills:
            cursor.execute("""
                INSERT INTO person_skills
                (person_id, skill_id, base_level, effective_level, confidence_low,
                 confidence_high, last_used, rater_id, rating_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (skill['person_id'], skill['skill_id'], skill['base_level'],
                  skill['effective_level'], skill['confidence_low'], skill['confidence_high'],
                  skill['last_used'], skill['rater_id'], skill['rating_date']))

    conn.commit()

    # Verify insertion
    cursor.execute("SELECT COUNT(*) FROM people")
    people_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM person_skills")
    skills_count = cursor.fetchone()[0]

    print(f"Inserted {people_count} people with {skills_count} skill assessments")

    # Show team composition
    cursor.execute("""
        SELECT
            CASE
                WHEN cost_hourly >= 160 THEN 'Senior/Principal'
                WHEN cost_hourly >= 120 THEN 'Mid-Level'
                WHEN cost_hourly >= 80 THEN 'Junior'
                ELSE 'Intern/Contract'
            END as level,
            COUNT(*) as count,
            ROUND(AVG(cost_hourly), 0) as avg_hourly_cost
        FROM people
        GROUP BY
            CASE
                WHEN cost_hourly >= 160 THEN 'Senior/Principal'
                WHEN cost_hourly >= 120 THEN 'Mid-Level'
                WHEN cost_hourly >= 80 THEN 'Junior'
                ELSE 'Intern/Contract'
            END
        ORDER BY avg_hourly_cost DESC
    """)

    print("\nTeam composition by level:")
    for row in cursor.fetchall():
        print(f"  {row['level']}: {row['count']} people (avg ${row['avg_hourly_cost']}/hr)")

    # Show geographic distribution
    cursor.execute("""
        SELECT
            CASE
                WHEN location LIKE '%CA%' OR location LIKE '%NY%' OR location LIKE '%TX%'
                     OR location LIKE '%WA%' OR location LIKE '%IL%' THEN 'North America'
                WHEN location LIKE '%UK%' OR location LIKE '%Germany%' OR location LIKE '%France%'
                     OR location LIKE '%Spain%' OR location LIKE '%Poland%' THEN 'Europe'
                WHEN location LIKE '%India%' OR location LIKE '%Japan%' OR location LIKE '%Vietnam%'
                     OR location LIKE '%UAE%' THEN 'Asia'
                ELSE 'Other'
            END as region,
            COUNT(*) as count
        FROM people
        GROUP BY
            CASE
                WHEN location LIKE '%CA%' OR location LIKE '%NY%' OR location LIKE '%TX%'
                     OR location LIKE '%WA%' OR location LIKE '%IL%' THEN 'North America'
                WHEN location LIKE '%UK%' OR location LIKE '%Germany%' OR location LIKE '%France%'
                     OR location LIKE '%Spain%' OR location LIKE '%Poland%' THEN 'Europe'
                WHEN location LIKE '%India%' OR location LIKE '%Japan%' OR location LIKE '%Vietnam%'
                     OR location LIKE '%UAE%' THEN 'Asia'
                ELSE 'Other'
            END
    """)

    print("\nGeographic distribution:")
    for row in cursor.fetchall():
        print(f"  {row['region']}: {row['count']} people")

    conn.close()

if __name__ == "__main__":
    # Generate people and skills when run as script
    populate_people_and_skills()

    print("\nSample team members:")
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name, location, cost_hourly,
               COUNT(ps.skill_id) as skill_count
        FROM people p
        LEFT JOIN person_skills ps ON p.id = ps.person_id
        GROUP BY p.id
        ORDER BY cost_hourly DESC
        LIMIT 5
    """)

    for row in cursor.fetchall():
        print(f"  {row['name']} ({row['location']}) - ${row['cost_hourly']}/hr, {row['skill_count']} skills")

    conn.close()