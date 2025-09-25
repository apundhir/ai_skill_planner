#!/usr/bin/env python3
"""
Generate realistic evidence data for skill proficiency calculations
Creates various types of evidence (certifications, deployments, incidents, etc.)
that will be used by ProficiencyCalculator to adjust base skill levels
"""

import sqlite3
import sys
import os
import random
from datetime import datetime, date, timedelta
from typing import List, Dict, Any

# Add parent directory to path to import database module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.init_db import get_db_connection

# Evidence templates with realistic data
EVIDENCE_TEMPLATES = {
    "certifications": [
        # Cloud & Infrastructure
        {"name": "AWS Certified Solutions Architect", "skills": ["cloud_ml", "system_design", "containerization"], "weight": 1.0},
        {"name": "Google Cloud Professional ML Engineer", "skills": ["cloud_ml", "model_deployment", "ml_pipelines"], "weight": 1.0},
        {"name": "Azure AI Engineer Associate", "skills": ["cloud_ml", "model_deployment", "nlp"], "weight": 0.8},
        {"name": "Kubernetes Certified Administrator", "skills": ["containerization", "system_design", "ml_pipelines"], "weight": 0.9},

        # ML/AI Specific
        {"name": "TensorFlow Developer Certificate", "skills": ["deep_learning", "computer_vision", "nlp"], "weight": 0.8},
        {"name": "Coursera Deep Learning Specialization", "skills": ["deep_learning", "computer_vision", "nlp"], "weight": 0.7},
        {"name": "Stanford CS229 Machine Learning", "skills": ["ml_supervised", "ml_unsupervised", "research_methodology"], "weight": 0.8},
        {"name": "Fast.ai Practical Deep Learning", "skills": ["deep_learning", "computer_vision", "nlp"], "weight": 0.7},

        # Data Engineering
        {"name": "Databricks Certified Data Engineer", "skills": ["data_pipelines", "spark", "streaming_data"], "weight": 0.9},
        {"name": "Snowflake Data Engineer Certification", "skills": ["data_warehousing", "sql_advanced", "etl_design"], "weight": 0.8},
        {"name": "Apache Kafka Certified Developer", "skills": ["streaming_data", "data_pipelines", "system_design"], "weight": 0.8},

        # Programming & Systems
        {"name": "Python Institute Certified Expert", "skills": ["python_advanced", "api_development", "testing_ml"], "weight": 0.7},
        {"name": "Docker Certified Associate", "skills": ["containerization", "model_deployment", "system_design"], "weight": 0.6},

        # Business & Leadership
        {"name": "PMP Project Management", "skills": ["project_management", "stakeholder_communication", "technical_strategy"], "weight": 0.8},
        {"name": "Certified ScrumMaster", "skills": ["project_management", "technical_mentoring"], "weight": 0.6}
    ],

    "production_deployments": [
        # ML Systems
        {"name": "Recommendation Engine Deployment", "skills": ["recommendation_systems", "model_deployment", "api_development"], "weight": 1.0},
        {"name": "Fraud Detection Model V1", "skills": ["fraud_detection", "risk_modeling", "streaming_data"], "weight": 1.0},
        {"name": "Computer Vision QA System", "skills": ["computer_vision", "deep_learning", "model_monitoring"], "weight": 1.0},
        {"name": "NLP Chatbot Platform", "skills": ["nlp", "llm_fine_tuning", "api_development"], "weight": 1.0},
        {"name": "Time Series Forecasting Pipeline", "skills": ["time_series", "ml_supervised", "data_pipelines"], "weight": 0.9},

        # Infrastructure & MLOps
        {"name": "ML Pipeline Automation", "skills": ["ml_pipelines", "containerization", "model_versioning"], "weight": 0.9},
        {"name": "Model Monitoring Dashboard", "skills": ["model_monitoring", "business_metrics", "system_design"], "weight": 0.8},
        {"name": "A/B Testing Framework", "skills": ["ab_testing", "experimentation", "business_metrics"], "weight": 0.8},
        {"name": "Real-time Data Processing", "skills": ["streaming_data", "data_pipelines", "spark"], "weight": 0.9},

        # Data Systems
        {"name": "Data Warehouse Migration", "skills": ["data_warehousing", "etl_design", "sql_advanced"], "weight": 0.8},
        {"name": "Feature Store Implementation", "skills": ["data_pipelines", "ml_pipelines", "system_design"], "weight": 0.9}
    ],

    "incident_responses": [
        {"name": "Model Performance Degradation", "skills": ["model_monitoring", "debugging", "system_design"], "weight": 0.8},
        {"name": "Data Pipeline Failure Recovery", "skills": ["data_pipelines", "etl_design", "system_design"], "weight": 0.8},
        {"name": "API Latency Spike Resolution", "skills": ["api_development", "system_design", "model_deployment"], "weight": 0.7},
        {"name": "Model Bias Detection & Fix", "skills": ["ml_supervised", "experimentation", "business_metrics"], "weight": 0.9},
        {"name": "Real-time Stream Processing Outage", "skills": ["streaming_data", "system_design", "data_pipelines"], "weight": 0.8}
    ],

    "code_reviews": [
        {"name": "ML Model Implementation Review", "skills": ["python_advanced", "ml_supervised", "testing_ml"], "weight": 0.6},
        {"name": "Data Pipeline Code Review", "skills": ["python_advanced", "data_pipelines", "testing_ml"], "weight": 0.6},
        {"name": "API Design Review", "skills": ["api_development", "system_design", "microservices"], "weight": 0.6},
        {"name": "Infrastructure as Code Review", "skills": ["containerization", "system_design", "cloud_ml"], "weight": 0.6}
    ],

    "self_projects": [
        {"name": "Personal Recommender System", "skills": ["recommendation_systems", "python_advanced", "model_deployment"], "weight": 0.5},
        {"name": "Open Source ML Library Contribution", "skills": ["python_advanced", "research_methodology", "technical_mentoring"], "weight": 0.4},
        {"name": "Kaggle Competition (Top 10%)", "skills": ["ml_supervised", "experimentation", "python_advanced"], "weight": 0.4},
        {"name": "Personal Blog on ML Topics", "skills": ["research_methodology", "stakeholder_communication", "technical_mentoring"], "weight": 0.3},
        {"name": "Side Project Mobile App", "skills": ["api_development", "system_design", "product_sense"], "weight": 0.3}
    ]
}

def get_person_skills_with_levels(person_id: str, db_path: str = "ai_skill_planner.db") -> Dict[str, float]:
    """Get person's skills with their base levels"""
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

def generate_person_evidence(person_id: str, person_data: Dict,
                           skills: Dict[str, float],
                           db_path: str = "ai_skill_planner.db") -> List[Dict[str, Any]]:
    """Generate realistic evidence for a person based on their skills and experience"""
    evidence_list = []

    # Get person's experience level (affects evidence quantity and quality)
    cost_hourly = person_data['cost_hourly']

    # Determine experience tier
    if cost_hourly >= 160:
        experience_tier = "senior"
        evidence_multiplier = 1.5
    elif cost_hourly >= 120:
        experience_tier = "mid"
        evidence_multiplier = 1.2
    else:
        experience_tier = "junior"
        evidence_multiplier = 0.8

    # Generate certifications (higher-level people have more)
    cert_count = random.randint(
        int(2 * evidence_multiplier),
        int(5 * evidence_multiplier)
    )

    available_certs = [cert for cert in EVIDENCE_TEMPLATES["certifications"]
                      if any(skill in skills for skill in cert["skills"])]

    if available_certs:
        selected_certs = random.sample(available_certs,
                                     min(cert_count, len(available_certs)))

        for cert in selected_certs:
            # Match certification to person's actual skills
            relevant_skills = [skill for skill in cert["skills"] if skill in skills]
            if relevant_skills:
                skill_id = random.choice(relevant_skills)

                # Generate realistic date (within last 2 years for active certs)
                days_ago = random.randint(30, 730)
                cert_date = date.today() - timedelta(days=days_ago)

                evidence_list.append({
                    'person_id': person_id,
                    'skill_id': skill_id,
                    'evidence_type': 'certification',
                    'url': f"https://credentials.example.com/{cert['name'].lower().replace(' ', '-')}",
                    'description': cert['name'],
                    'date_achieved': cert_date,
                    'verified_by': 'automated_verification',
                    'verification_date': cert_date + timedelta(days=1)
                })

    # Generate production deployments (senior people have more)
    deployment_count = random.randint(
        int(1 * evidence_multiplier),
        int(4 * evidence_multiplier)
    )

    available_deployments = [dep for dep in EVIDENCE_TEMPLATES["production_deployments"]
                           if any(skill in skills and skills[skill] >= 3.0
                                 for skill in dep["skills"])]

    if available_deployments:
        selected_deployments = random.sample(available_deployments,
                                           min(deployment_count, len(available_deployments)))

        for deployment in selected_deployments:
            relevant_skills = [skill for skill in deployment["skills"]
                             if skill in skills and skills[skill] >= 3.0]
            if relevant_skills:
                skill_id = random.choice(relevant_skills)

                # Production deployments are more recent (last year)
                days_ago = random.randint(30, 365)
                deploy_date = date.today() - timedelta(days=days_ago)

                evidence_list.append({
                    'person_id': person_id,
                    'skill_id': skill_id,
                    'evidence_type': 'production_deployment',
                    'url': f"https://internal.company.com/projects/{deployment['name'].lower().replace(' ', '-')}",
                    'description': deployment['name'],
                    'date_achieved': deploy_date,
                    'verified_by': 'sarah_chen',  # Senior leadership verification
                    'verification_date': deploy_date + timedelta(days=7)
                })

    # Generate incident responses (for experienced people with relevant skills)
    if experience_tier in ["mid", "senior"] and any(level >= 3.5
                                                   for level in skills.values()):
        incident_count = random.randint(1, 3)

        available_incidents = [inc for inc in EVIDENCE_TEMPLATES["incident_responses"]
                             if any(skill in skills and skills[skill] >= 3.5
                                   for skill in inc["skills"])]

        if available_incidents:
            selected_incidents = random.sample(available_incidents,
                                             min(incident_count, len(available_incidents)))

            for incident in selected_incidents:
                relevant_skills = [skill for skill in incident["skills"]
                                 if skill in skills and skills[skill] >= 3.5]
                if relevant_skills:
                    skill_id = random.choice(relevant_skills)

                    # Incidents are recent (last 6 months)
                    days_ago = random.randint(7, 180)
                    incident_date = date.today() - timedelta(days=days_ago)

                    evidence_list.append({
                        'person_id': person_id,
                        'skill_id': skill_id,
                        'evidence_type': 'incident_response',
                        'url': f"https://internal.company.com/incidents/{incident['name'].lower().replace(' ', '-')}",
                        'description': incident['name'],
                        'date_achieved': incident_date,
                        'verified_by': 'alex_rodriguez',  # Tech lead verification
                        'verification_date': incident_date + timedelta(days=1)
                    })

    # Generate code reviews (regular activity for all developers)
    review_count = random.randint(
        int(3 * evidence_multiplier),
        int(8 * evidence_multiplier)
    )

    available_reviews = [rev for rev in EVIDENCE_TEMPLATES["code_reviews"]
                        if any(skill in skills for skill in rev["skills"])]

    if available_reviews:
        for _ in range(review_count):
            review = random.choice(available_reviews)
            relevant_skills = [skill for skill in review["skills"] if skill in skills]
            if relevant_skills:
                skill_id = random.choice(relevant_skills)

                # Code reviews are frequent and recent
                days_ago = random.randint(1, 90)
                review_date = date.today() - timedelta(days=days_ago)

                evidence_list.append({
                    'person_id': person_id,
                    'skill_id': skill_id,
                    'evidence_type': 'code_review',
                    'url': f"https://github.com/company/repo/pull/{random.randint(100, 9999)}",
                    'description': review['name'],
                    'date_achieved': review_date,
                    'verified_by': 'peer_review',
                    'verification_date': review_date
                })

    # Generate self projects (varies by person)
    if random.random() < 0.7:  # 70% of people have side projects
        project_count = random.randint(1, 3)

        available_projects = [proj for proj in EVIDENCE_TEMPLATES["self_projects"]
                            if any(skill in skills for skill in proj["skills"])]

        if available_projects:
            selected_projects = random.sample(available_projects,
                                            min(project_count, len(available_projects)))

            for project in selected_projects:
                relevant_skills = [skill for skill in project["skills"] if skill in skills]
                if relevant_skills:
                    skill_id = random.choice(relevant_skills)

                    # Self projects span a wide time range
                    days_ago = random.randint(90, 900)
                    project_date = date.today() - timedelta(days=days_ago)

                    evidence_list.append({
                        'person_id': person_id,
                        'skill_id': skill_id,
                        'evidence_type': 'self_project',
                        'url': f"https://github.com/{person_id}/{project['name'].lower().replace(' ', '-')}",
                        'description': project['name'],
                        'date_achieved': project_date,
                        'verified_by': None,  # Self-reported
                        'verification_date': None
                    })

    return evidence_list

def populate_evidence_data(db_path: str = "ai_skill_planner.db") -> None:
    """Generate and populate evidence for all people"""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Get all people
    cursor.execute("""
        SELECT id, name, cost_hourly, fte
        FROM people
        ORDER BY cost_hourly DESC
    """)
    people = [dict(row) for row in cursor.fetchall()]

    # Clear existing evidence
    cursor.execute("DELETE FROM evidence")

    total_evidence = 0

    for person in people:
        person_id = person['id']
        print(f"Generating evidence for {person['name']}...")

        # Get person's skills
        skills = get_person_skills_with_levels(person_id, db_path)

        # Generate evidence
        evidence_list = generate_person_evidence(person_id, person, skills, db_path)

        # Insert evidence
        for evidence in evidence_list:
            cursor.execute("""
                INSERT INTO evidence (person_id, skill_id, evidence_type, url,
                                    description, date_achieved, verified_by, verification_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (evidence['person_id'], evidence['skill_id'], evidence['evidence_type'],
                  evidence['url'], evidence['description'], evidence['date_achieved'],
                  evidence['verified_by'], evidence['verification_date']))

        total_evidence += len(evidence_list)
        print(f"  Generated {len(evidence_list)} pieces of evidence")

    conn.commit()

    # Show summary statistics
    print(f"\n📊 Evidence Generation Summary:")
    print(f"Total evidence pieces: {total_evidence}")

    # Evidence by type
    cursor.execute("""
        SELECT evidence_type, COUNT(*) as count,
               COUNT(DISTINCT person_id) as people_count
        FROM evidence
        GROUP BY evidence_type
        ORDER BY count DESC
    """)

    print(f"\nEvidence by type:")
    for row in cursor.fetchall():
        print(f"  {row['evidence_type']}: {row['count']} pieces ({row['people_count']} people)")

    # Top skills with most evidence
    cursor.execute("""
        SELECT s.name, COUNT(*) as evidence_count
        FROM evidence e
        JOIN skills s ON e.skill_id = s.id
        GROUP BY s.id, s.name
        ORDER BY evidence_count DESC
        LIMIT 10
    """)

    print(f"\nTop skills by evidence count:")
    for row in cursor.fetchall():
        print(f"  {row['name']}: {row['evidence_count']} pieces")

    # Recent evidence activity
    cursor.execute("""
        SELECT COUNT(*) as recent_count
        FROM evidence
        WHERE date_achieved >= date('now', '-30 days')
    """)

    recent_count = cursor.fetchone()['recent_count']
    print(f"\nRecent activity (last 30 days): {recent_count} pieces of evidence")

    conn.close()

if __name__ == "__main__":
    # Generate evidence when run as script
    populate_evidence_data()

    # Show sample evidence for a senior person
    conn = get_db_connection()
    cursor = conn.cursor()

    print(f"\n=== Sample Evidence: Sarah Chen (VP Engineering) ===")
    cursor.execute("""
        SELECT e.evidence_type, e.description, e.date_achieved, s.name as skill_name
        FROM evidence e
        JOIN skills s ON e.skill_id = s.id
        WHERE e.person_id = 'sarah_chen'
        ORDER BY e.date_achieved DESC
        LIMIT 5
    """)

    for row in cursor.fetchall():
        print(f"  {row['evidence_type']}: {row['description']} ({row['skill_name']}) - {row['date_achieved']}")

    conn.close()