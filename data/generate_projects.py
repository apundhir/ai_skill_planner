#!/usr/bin/env python3
"""
Generate realistic AI projects with phases and skill requirements
Creates 4 diverse projects typical of a growing AI company
"""

import sqlite3
import sys
import os
import random
from datetime import datetime, timedelta, date
from typing import List, Dict, Any

# Add parent directory to path to import database module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.init_db import get_db_connection

# Realistic AI projects for the team
AI_PROJECTS = [
    {
        "id": "personalization_engine",
        "name": "E-commerce Personalization Engine",
        "complexity": "high",
        "regulatory_intensity": "moderate",  # GDPR/CCPA considerations
        "start_date": date(2024, 2, 1),
        "end_date": date(2024, 8, 31),
        "cost_of_delay_weekly": 75000.0,  # High revenue impact
        "risk_tolerance": "medium",
        "description": "Build ML system to personalize product recommendations and search results",
        "business_value": "$2M annual revenue impact from improved conversion rates"
    },
    {
        "id": "fraud_detection_v2",
        "name": "Real-time Fraud Detection System V2",
        "complexity": "high",
        "regulatory_intensity": "high",  # Financial regulations
        "start_date": date(2024, 1, 15),
        "end_date": date(2024, 7, 15),
        "cost_of_delay_weekly": 125000.0,  # Very high - fraud costs money daily
        "risk_tolerance": "low",  # Can't afford mistakes in fraud detection
        "description": "Upgrade fraud detection with real-time ML models and better accuracy",
        "business_value": "$5M annual savings from reduced fraud losses"
    },
    {
        "id": "customer_support_ai",
        "name": "AI-Powered Customer Support Platform",
        "complexity": "medium",
        "regulatory_intensity": "moderate",
        "start_date": date(2024, 3, 1),
        "end_date": date(2024, 9, 30),
        "cost_of_delay_weekly": 35000.0,  # Operational efficiency impact
        "risk_tolerance": "medium",
        "description": "LLM-based chatbot and ticket routing system to improve support efficiency",
        "business_value": "$1.5M annual cost savings from reduced support headcount"
    },
    {
        "id": "quality_vision_system",
        "name": "Manufacturing Quality Control Vision System",
        "complexity": "medium",
        "regulatory_intensity": "high",  # Manufacturing/safety standards
        "start_date": date(2024, 4, 1),
        "end_date": date(2024, 10, 31),
        "cost_of_delay_weekly": 45000.0,  # Quality issues are expensive
        "risk_tolerance": "low",  # Can't ship defective products
        "description": "Computer vision system for automated quality inspection in manufacturing",
        "business_value": "$3M annual savings from reduced defects and manual inspection"
    }
]

# Project phases with realistic timelines and skill requirements
PROJECT_PHASES = {
    "personalization_engine": [
        {
            "phase_name": "discovery",
            "start_date": date(2024, 2, 1),
            "end_date": date(2024, 2, 28),
            "gate_threshold": 0.7,
            "requirements": {
                "product_sense": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 2.0, "criticality": 0.9},
                "business_metrics": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 1.5, "criticality": 0.8},
                "stakeholder_communication": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 1.5, "criticality": 0.7},
                "recommendation_systems": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 3.0, "criticality": 0.9},
                "sql_advanced": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 2.0, "criticality": 0.6}
            }
        },
        {
            "phase_name": "data_prep",
            "start_date": date(2024, 3, 1),
            "end_date": date(2024, 4, 15),
            "gate_threshold": 0.75,
            "requirements": {
                "data_pipelines": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 4.0, "criticality": 0.9},
                "sql_advanced": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 3.0, "criticality": 0.8},
                "etl_design": {"required_level": 3.8, "min_level": 3.0, "fte_weeks": 3.5, "criticality": 0.8},
                "spark": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 2.0, "criticality": 0.7},
                "python_advanced": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 2.5, "criticality": 0.6}
            }
        },
        {
            "phase_name": "modeling",
            "start_date": date(2024, 4, 16),
            "end_date": date(2024, 6, 30),
            "gate_threshold": 0.8,
            "requirements": {
                "recommendation_systems": {"required_level": 4.2, "min_level": 3.8, "fte_weeks": 6.0, "criticality": 1.0},
                "personalization": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 5.0, "criticality": 0.9},
                "ml_supervised": {"required_level": 3.8, "min_level": 3.5, "fte_weeks": 4.0, "criticality": 0.8},
                "deep_learning": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 3.0, "criticality": 0.7},
                "ab_testing": {"required_level": 3.8, "min_level": 3.0, "fte_weeks": 2.5, "criticality": 0.8},
                "experimentation": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 3.0, "criticality": 0.7}
            }
        },
        {
            "phase_name": "deployment",
            "start_date": date(2024, 7, 1),
            "end_date": date(2024, 8, 15),
            "gate_threshold": 0.85,
            "requirements": {
                "model_deployment": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 4.0, "criticality": 1.0},
                "api_development": {"required_level": 3.8, "min_level": 3.0, "fte_weeks": 3.5, "criticality": 0.9},
                "containerization": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 2.5, "criticality": 0.8},
                "cloud_ml": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 2.0, "criticality": 0.7},
                "microservices": {"required_level": 3.0, "min_level": 2.5, "fte_weeks": 2.0, "criticality": 0.6}
            }
        },
        {
            "phase_name": "monitoring",
            "start_date": date(2024, 8, 16),
            "end_date": date(2024, 8, 31),
            "gate_threshold": 0.8,
            "requirements": {
                "model_monitoring": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 2.0, "criticality": 1.0},
                "ab_testing": {"required_level": 3.8, "min_level": 3.0, "fte_weeks": 1.5, "criticality": 0.9},
                "business_metrics": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 1.0, "criticality": 0.8},
                "system_design": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 1.0, "criticality": 0.7}
            }
        }
    ],

    "fraud_detection_v2": [
        {
            "phase_name": "discovery",
            "start_date": date(2024, 1, 15),
            "end_date": date(2024, 2, 15),
            "gate_threshold": 0.8,  # Higher threshold for fraud detection
            "requirements": {
                "risk_modeling": {"required_level": 4.5, "min_level": 4.0, "fte_weeks": 3.0, "criticality": 1.0},
                "fraud_detection": {"required_level": 4.2, "min_level": 3.8, "fte_weeks": 2.5, "criticality": 1.0},
                "business_metrics": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 1.5, "criticality": 0.8},
                "stakeholder_communication": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 1.0, "criticality": 0.7}
            }
        },
        {
            "phase_name": "data_prep",
            "start_date": date(2024, 2, 16),
            "end_date": date(2024, 3, 31),
            "gate_threshold": 0.85,
            "requirements": {
                "streaming_data": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 4.0, "criticality": 1.0},
                "data_pipelines": {"required_level": 4.2, "min_level": 3.8, "fte_weeks": 5.0, "criticality": 0.9},
                "sql_advanced": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 3.0, "criticality": 0.8},
                "data_warehousing": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 2.0, "criticality": 0.7}
            }
        },
        {
            "phase_name": "modeling",
            "start_date": date(2024, 4, 1),
            "end_date": date(2024, 5, 31),
            "gate_threshold": 0.9,  # Very high threshold for fraud models
            "requirements": {
                "fraud_detection": {"required_level": 4.5, "min_level": 4.0, "fte_weeks": 6.0, "criticality": 1.0},
                "risk_modeling": {"required_level": 4.2, "min_level": 3.8, "fte_weeks": 5.0, "criticality": 1.0},
                "ml_supervised": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 4.0, "criticality": 0.9},
                "time_series": {"required_level": 3.8, "min_level": 3.0, "fte_weeks": 3.0, "criticality": 0.8},
                "experimentation": {"required_level": 3.8, "min_level": 3.0, "fte_weeks": 2.5, "criticality": 0.7}
            }
        },
        {
            "phase_name": "deployment",
            "start_date": date(2024, 6, 1),
            "end_date": date(2024, 7, 1),
            "gate_threshold": 0.9,
            "requirements": {
                "streaming_data": {"required_level": 4.2, "min_level": 3.8, "fte_weeks": 3.0, "criticality": 1.0},
                "model_deployment": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 3.5, "criticality": 1.0},
                "api_development": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 3.0, "criticality": 0.9},
                "microservices": {"required_level": 3.8, "min_level": 3.0, "fte_weeks": 2.5, "criticality": 0.8},
                "database_design": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 2.0, "criticality": 0.7}
            }
        },
        {
            "phase_name": "monitoring",
            "start_date": date(2024, 7, 2),
            "end_date": date(2024, 7, 15),
            "gate_threshold": 0.9,
            "requirements": {
                "model_monitoring": {"required_level": 4.2, "min_level": 3.8, "fte_weeks": 2.0, "criticality": 1.0},
                "fraud_detection": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 1.5, "criticality": 0.9},
                "business_metrics": {"required_level": 3.8, "min_level": 3.0, "fte_weeks": 1.0, "criticality": 0.8}
            }
        }
    ],

    "customer_support_ai": [
        {
            "phase_name": "discovery",
            "start_date": date(2024, 3, 1),
            "end_date": date(2024, 3, 31),
            "gate_threshold": 0.7,
            "requirements": {
                "nlp": {"required_level": 3.8, "min_level": 3.0, "fte_weeks": 2.5, "criticality": 0.9},
                "product_sense": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 2.0, "criticality": 0.8},
                "stakeholder_communication": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 1.5, "criticality": 0.7},
                "business_metrics": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 1.0, "criticality": 0.6}
            }
        },
        {
            "phase_name": "data_prep",
            "start_date": date(2024, 4, 1),
            "end_date": date(2024, 5, 15),
            "gate_threshold": 0.75,
            "requirements": {
                "data_pipelines": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 3.0, "criticality": 0.8},
                "sql_advanced": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 2.5, "criticality": 0.7},
                "nlp": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 4.0, "criticality": 0.9},
                "python_advanced": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 2.0, "criticality": 0.6}
            }
        },
        {
            "phase_name": "modeling",
            "start_date": date(2024, 5, 16),
            "end_date": date(2024, 7, 31),
            "gate_threshold": 0.8,
            "requirements": {
                "llm_fine_tuning": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 5.0, "criticality": 1.0},
                "nlp": {"required_level": 4.2, "min_level": 3.8, "fte_weeks": 6.0, "criticality": 1.0},
                "deep_learning": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 3.0, "criticality": 0.8},
                "experimentation": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 2.5, "criticality": 0.7}
            }
        },
        {
            "phase_name": "deployment",
            "start_date": date(2024, 8, 1),
            "end_date": date(2024, 9, 15),
            "gate_threshold": 0.8,
            "requirements": {
                "api_development": {"required_level": 3.8, "min_level": 3.0, "fte_weeks": 3.5, "criticality": 0.9},
                "model_deployment": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 3.0, "criticality": 0.8},
                "containerization": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 2.0, "criticality": 0.7},
                "cloud_ml": {"required_level": 3.0, "min_level": 2.5, "fte_weeks": 1.5, "criticality": 0.6}
            }
        },
        {
            "phase_name": "monitoring",
            "start_date": date(2024, 9, 16),
            "end_date": date(2024, 9, 30),
            "gate_threshold": 0.75,
            "requirements": {
                "model_monitoring": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 1.5, "criticality": 0.8},
                "business_metrics": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 1.0, "criticality": 0.7},
                "ab_testing": {"required_level": 3.0, "min_level": 2.5, "fte_weeks": 1.0, "criticality": 0.6}
            }
        }
    ],

    "quality_vision_system": [
        {
            "phase_name": "discovery",
            "start_date": date(2024, 4, 1),
            "end_date": date(2024, 4, 30),
            "gate_threshold": 0.8,  # High threshold for manufacturing
            "requirements": {
                "computer_vision": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 2.5, "criticality": 1.0},
                "product_sense": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 1.5, "criticality": 0.8},
                "stakeholder_communication": {"required_level": 3.8, "min_level": 3.0, "fte_weeks": 1.5, "criticality": 0.7},
                "business_metrics": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 1.0, "criticality": 0.6}
            }
        },
        {
            "phase_name": "data_prep",
            "start_date": date(2024, 5, 1),
            "end_date": date(2024, 6, 15),
            "gate_threshold": 0.8,
            "requirements": {
                "data_pipelines": {"required_level": 3.8, "min_level": 3.0, "fte_weeks": 3.5, "criticality": 0.8},
                "computer_vision": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 4.0, "criticality": 1.0},
                "python_advanced": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 2.5, "criticality": 0.7},
                "sql_advanced": {"required_level": 3.0, "min_level": 2.5, "fte_weeks": 1.5, "criticality": 0.6}
            }
        },
        {
            "phase_name": "modeling",
            "start_date": date(2024, 6, 16),
            "end_date": date(2024, 8, 31),
            "gate_threshold": 0.85,
            "requirements": {
                "computer_vision": {"required_level": 4.3, "min_level": 4.0, "fte_weeks": 6.0, "criticality": 1.0},
                "deep_learning": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 5.0, "criticality": 0.9},
                "python_advanced": {"required_level": 3.8, "min_level": 3.0, "fte_weeks": 3.0, "criticality": 0.8},
                "experimentation": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 2.5, "criticality": 0.7},
                "testing_ml": {"required_level": 3.8, "min_level": 3.0, "fte_weeks": 2.0, "criticality": 0.8}
            }
        },
        {
            "phase_name": "deployment",
            "start_date": date(2024, 9, 1),
            "end_date": date(2024, 10, 15),
            "gate_threshold": 0.9,  # Very high for manufacturing
            "requirements": {
                "model_deployment": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 3.5, "criticality": 1.0},
                "system_design": {"required_level": 3.8, "min_level": 3.0, "fte_weeks": 3.0, "criticality": 0.9},
                "api_development": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 2.5, "criticality": 0.8},
                "containerization": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 2.0, "criticality": 0.7},
                "testing_ml": {"required_level": 3.8, "min_level": 3.0, "fte_weeks": 2.5, "criticality": 0.8}
            }
        },
        {
            "phase_name": "monitoring",
            "start_date": date(2024, 10, 16),
            "end_date": date(2024, 10, 31),
            "gate_threshold": 0.9,
            "requirements": {
                "model_monitoring": {"required_level": 4.0, "min_level": 3.5, "fte_weeks": 1.5, "criticality": 1.0},
                "computer_vision": {"required_level": 3.8, "min_level": 3.0, "fte_weeks": 1.0, "criticality": 0.8},
                "business_metrics": {"required_level": 3.5, "min_level": 3.0, "fte_weeks": 1.0, "criticality": 0.7}
            }
        }
    ]
}

def populate_projects_and_phases(db_path: str = "ai_skill_planner.db") -> None:
    """
    Populate projects, phases, and project_requirements tables

    Args:
        db_path: Path to SQLite database
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Clear existing data
    cursor.execute("DELETE FROM project_requirements")
    cursor.execute("DELETE FROM phases")
    cursor.execute("DELETE FROM projects")

    # Insert projects
    for project in AI_PROJECTS:
        cursor.execute("""
            INSERT INTO projects (id, name, complexity, regulatory_intensity, start_date,
                                end_date, cost_of_delay_weekly, risk_tolerance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (project['id'], project['name'], project['complexity'],
              project['regulatory_intensity'], project['start_date'],
              project['end_date'], project['cost_of_delay_weekly'],
              project['risk_tolerance']))

        print(f"Inserted project: {project['name']}")
        print(f"  Cost of Delay: ${project['cost_of_delay_weekly']:,}/week")
        print(f"  Business Value: {project['business_value']}")

    # Insert phases and requirements
    for project_id, phases in PROJECT_PHASES.items():
        for phase in phases:
            # Insert phase
            cursor.execute("""
                INSERT INTO phases (project_id, phase_name, start_date, end_date, gate_threshold)
                VALUES (?, ?, ?, ?, ?)
            """, (project_id, phase['phase_name'], phase['start_date'],
                  phase['end_date'], phase['gate_threshold']))

            # Insert requirements for this phase
            for skill_id, req in phase['requirements'].items():
                cursor.execute("""
                    INSERT INTO project_requirements
                    (project_id, phase_name, skill_id, required_level, min_level, fte_weeks, criticality)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (project_id, phase['phase_name'], skill_id,
                      req['required_level'], req['min_level'],
                      req['fte_weeks'], req['criticality']))

    conn.commit()

    # Show summary statistics
    cursor.execute("SELECT COUNT(*) FROM projects")
    project_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM phases")
    phase_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM project_requirements")
    req_count = cursor.fetchone()[0]

    print(f"\nSummary:")
    print(f"  Projects: {project_count}")
    print(f"  Phases: {phase_count}")
    print(f"  Skill requirements: {req_count}")

    # Show project complexity breakdown
    cursor.execute("""
        SELECT complexity, COUNT(*) as count,
               ROUND(AVG(cost_of_delay_weekly), 0) as avg_cod
        FROM projects
        GROUP BY complexity
        ORDER BY avg_cod DESC
    """)

    print(f"\nProjects by complexity:")
    for row in cursor.fetchall():
        print(f"  {row['complexity']}: {row['count']} projects (avg CoD: ${row['avg_cod']:,}/week)")

    # Show most in-demand skills
    cursor.execute("""
        SELECT skill_id, COUNT(*) as demand_count,
               ROUND(AVG(required_level), 1) as avg_level_needed
        FROM project_requirements
        GROUP BY skill_id
        HAVING COUNT(*) >= 3
        ORDER BY demand_count DESC, avg_level_needed DESC
        LIMIT 10
    """)

    print(f"\nMost in-demand skills:")
    for row in cursor.fetchall():
        print(f"  {row['skill_id']}: {row['demand_count']} phases need it (avg level {row['avg_level_needed']})")

    conn.close()

if __name__ == "__main__":
    # Generate projects when run as script
    populate_projects_and_phases()

    # Show a sample project timeline
    conn = get_db_connection()
    cursor = conn.cursor()

    print(f"\n=== Sample Project: Personalization Engine ===")
    cursor.execute("""
        SELECT p.phase_name, p.start_date, p.end_date, p.gate_threshold,
               COUNT(pr.skill_id) as skill_requirements
        FROM phases p
        LEFT JOIN project_requirements pr ON p.project_id = pr.project_id
                                           AND p.phase_name = pr.phase_name
        WHERE p.project_id = 'personalization_engine'
        GROUP BY p.phase_name, p.start_date, p.end_date, p.gate_threshold
        ORDER BY p.start_date
    """)

    for row in cursor.fetchall():
        print(f"  {row['phase_name']}: {row['start_date']} to {row['end_date']}")
        print(f"    Gate threshold: {row['gate_threshold']}, Skills needed: {row['skill_requirements']}")

    conn.close()