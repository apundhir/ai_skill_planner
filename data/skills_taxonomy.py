#!/usr/bin/env python3
"""
AI Skills Taxonomy - Comprehensive skill definitions for AI companies
Based on real-world AI engineering roles and requirements
"""

import sqlite3
import sys
import os
from typing import List, Dict, Any

# Add parent directory to path to import database module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.init_db import get_db_connection

# Comprehensive AI Skills Taxonomy
AI_SKILLS_TAXONOMY = [
    # MACHINE LEARNING CORE
    {
        "id": "ml_supervised",
        "name": "Supervised Learning",
        "category": "machine_learning",
        "decay_rate": 0.15  # Skills decay moderately if not used
    },
    {
        "id": "ml_unsupervised",
        "name": "Unsupervised Learning",
        "category": "machine_learning",
        "decay_rate": 0.15
    },
    {
        "id": "deep_learning",
        "name": "Deep Learning",
        "category": "machine_learning",
        "decay_rate": 0.20  # Fast-moving field, higher decay
    },
    {
        "id": "nlp",
        "name": "Natural Language Processing",
        "category": "machine_learning",
        "decay_rate": 0.25  # Very fast-moving, LLMs evolving rapidly
    },
    {
        "id": "computer_vision",
        "name": "Computer Vision",
        "category": "machine_learning",
        "decay_rate": 0.20
    },
    {
        "id": "time_series",
        "name": "Time Series Analysis",
        "category": "machine_learning",
        "decay_rate": 0.10  # More stable field
    },
    {
        "id": "reinforcement_learning",
        "name": "Reinforcement Learning",
        "category": "machine_learning",
        "decay_rate": 0.25
    },
    {
        "id": "llm_fine_tuning",
        "name": "LLM Fine-tuning",
        "category": "machine_learning",
        "decay_rate": 0.30  # Very fast evolving
    },

    # MLOPS & INFRASTRUCTURE
    {
        "id": "model_deployment",
        "name": "Model Deployment",
        "category": "mlops",
        "decay_rate": 0.15
    },
    {
        "id": "model_monitoring",
        "name": "Model Monitoring & Observability",
        "category": "mlops",
        "decay_rate": 0.15
    },
    {
        "id": "ml_pipelines",
        "name": "ML Pipeline Development",
        "category": "mlops",
        "decay_rate": 0.15
    },
    {
        "id": "containerization",
        "name": "Docker/Kubernetes",
        "category": "mlops",
        "decay_rate": 0.12
    },
    {
        "id": "cloud_ml",
        "name": "Cloud ML Platforms",
        "category": "mlops",
        "decay_rate": 0.18
    },
    {
        "id": "model_versioning",
        "name": "Model Versioning (MLflow, DVC)",
        "category": "mlops",
        "decay_rate": 0.12
    },
    {
        "id": "ab_testing",
        "name": "A/B Testing for ML",
        "category": "mlops",
        "decay_rate": 0.10
    },

    # DATA ENGINEERING
    {
        "id": "data_pipelines",
        "name": "Data Pipeline Development",
        "category": "data_engineering",
        "decay_rate": 0.12
    },
    {
        "id": "sql_advanced",
        "name": "Advanced SQL",
        "category": "data_engineering",
        "decay_rate": 0.08  # Stable skill
    },
    {
        "id": "spark",
        "name": "Apache Spark",
        "category": "data_engineering",
        "decay_rate": 0.12
    },
    {
        "id": "streaming_data",
        "name": "Real-time Data Processing",
        "category": "data_engineering",
        "decay_rate": 0.15
    },
    {
        "id": "data_warehousing",
        "name": "Data Warehousing",
        "category": "data_engineering",
        "decay_rate": 0.10
    },
    {
        "id": "etl_design",
        "name": "ETL/ELT Design",
        "category": "data_engineering",
        "decay_rate": 0.10
    },

    # SOFTWARE ENGINEERING
    {
        "id": "python_advanced",
        "name": "Advanced Python",
        "category": "software_engineering",
        "decay_rate": 0.10
    },
    {
        "id": "system_design",
        "name": "System Design",
        "category": "software_engineering",
        "decay_rate": 0.08
    },
    {
        "id": "api_development",
        "name": "API Development",
        "category": "software_engineering",
        "decay_rate": 0.10
    },
    {
        "id": "microservices",
        "name": "Microservices Architecture",
        "category": "software_engineering",
        "decay_rate": 0.12
    },
    {
        "id": "database_design",
        "name": "Database Design",
        "category": "software_engineering",
        "decay_rate": 0.08
    },
    {
        "id": "testing_ml",
        "name": "ML Testing Strategies",
        "category": "software_engineering",
        "decay_rate": 0.15
    },

    # SPECIALIZED DOMAINS
    {
        "id": "risk_modeling",
        "name": "Risk Modeling",
        "category": "domain_expertise",
        "decay_rate": 0.12
    },
    {
        "id": "recommendation_systems",
        "name": "Recommendation Systems",
        "category": "domain_expertise",
        "decay_rate": 0.15
    },
    {
        "id": "fraud_detection",
        "name": "Fraud Detection",
        "category": "domain_expertise",
        "decay_rate": 0.12
    },
    {
        "id": "personalization",
        "name": "Personalization Systems",
        "category": "domain_expertise",
        "decay_rate": 0.15
    },

    # RESEARCH & INNOVATION
    {
        "id": "research_methodology",
        "name": "Research Methodology",
        "category": "research",
        "decay_rate": 0.05  # Stable methodological skill
    },
    {
        "id": "paper_implementation",
        "name": "Research Paper Implementation",
        "category": "research",
        "decay_rate": 0.20
    },
    {
        "id": "experimentation",
        "name": "ML Experimentation Design",
        "category": "research",
        "decay_rate": 0.10
    },

    # LEADERSHIP & SOFT SKILLS
    {
        "id": "technical_mentoring",
        "name": "Technical Mentoring",
        "category": "leadership",
        "decay_rate": 0.05
    },
    {
        "id": "project_management",
        "name": "Technical Project Management",
        "category": "leadership",
        "decay_rate": 0.08
    },
    {
        "id": "stakeholder_communication",
        "name": "Stakeholder Communication",
        "category": "leadership",
        "decay_rate": 0.05
    },
    {
        "id": "technical_strategy",
        "name": "Technical Strategy",
        "category": "leadership",
        "decay_rate": 0.08
    },
    {
        "id": "hiring_interviewing",
        "name": "Technical Hiring & Interviewing",
        "category": "leadership",
        "decay_rate": 0.08
    },

    # BUSINESS & PRODUCT
    {
        "id": "product_sense",
        "name": "ML Product Sense",
        "category": "business",
        "decay_rate": 0.10
    },
    {
        "id": "business_metrics",
        "name": "Business Metrics & KPIs",
        "category": "business",
        "decay_rate": 0.08
    },
    {
        "id": "roi_analysis",
        "name": "ML ROI Analysis",
        "category": "business",
        "decay_rate": 0.10
    }
]

def populate_skills_taxonomy(db_path: str = "ai_skill_planner.db") -> None:
    """
    Populate the skills table with comprehensive AI skills taxonomy

    Args:
        db_path: Path to SQLite database
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Clear existing skills
    cursor.execute("DELETE FROM skills")

    # Insert all skills from taxonomy
    for skill in AI_SKILLS_TAXONOMY:
        cursor.execute("""
            INSERT INTO skills (id, name, category, decay_rate)
            VALUES (?, ?, ?, ?)
        """, (skill["id"], skill["name"], skill["category"], skill["decay_rate"]))

    conn.commit()

    # Verify insertion
    cursor.execute("SELECT COUNT(*) FROM skills")
    count = cursor.fetchone()[0]
    print(f"Inserted {count} skills into taxonomy")

    # Show category breakdown
    cursor.execute("""
        SELECT category, COUNT(*) as skill_count
        FROM skills
        GROUP BY category
        ORDER BY skill_count DESC
    """)

    print("\nSkills by category:")
    for row in cursor.fetchall():
        print(f"  {row['category']}: {row['skill_count']} skills")

    conn.close()

def get_skills_by_category(category: str, db_path: str = "ai_skill_planner.db") -> List[Dict[str, Any]]:
    """
    Get all skills in a specific category

    Args:
        category: Skill category to filter by
        db_path: Path to SQLite database

    Returns:
        List of skill dictionaries
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, name, category, decay_rate
        FROM skills
        WHERE category = ?
        ORDER BY name
    """, (category,))

    skills = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return skills

if __name__ == "__main__":
    # Populate skills when run as script
    populate_skills_taxonomy()

    # Test retrieval
    print(f"\nMachine Learning skills:")
    ml_skills = get_skills_by_category("machine_learning")
    for skill in ml_skills:
        print(f"  {skill['name']} (decay: {skill['decay_rate']})")

    print(f"\nMLOps skills:")
    mlops_skills = get_skills_by_category("mlops")
    for skill in mlops_skills:
        print(f"  {skill['name']} (decay: {skill['decay_rate']})")