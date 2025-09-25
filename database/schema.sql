-- AI Team Skill Gap Heat Map Generator - Database Schema
-- Based on PRD v2.0 Enhanced Data Model

-- Skills taxonomy table (referenced by other tables)
CREATE TABLE skills (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    decay_rate REAL DEFAULT 0.1,  -- Monthly skill decay rate
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Core entities with proper relationships
CREATE TABLE people (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    location TEXT,
    timezone TEXT,
    fte REAL DEFAULT 1.0,
    cost_hourly REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    complexity TEXT CHECK(complexity IN ('low', 'medium', 'high')),
    regulatory_intensity TEXT CHECK(regulatory_intensity IN ('none', 'moderate', 'high')),
    start_date DATE,
    end_date DATE,
    cost_of_delay_weekly REAL,
    risk_tolerance TEXT CHECK(risk_tolerance IN ('low', 'medium', 'high')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE phases (
    project_id TEXT REFERENCES projects(id),
    phase_name TEXT CHECK(phase_name IN ('discovery', 'data_prep', 'modeling', 'deployment', 'monitoring')),
    start_date DATE,
    end_date DATE,
    gate_threshold REAL DEFAULT 0.7,
    PRIMARY KEY (project_id, phase_name)
);

CREATE TABLE assignments (
    project_id TEXT REFERENCES projects(id),
    person_id TEXT REFERENCES people(id),
    phase_name TEXT,
    availability REAL CHECK(availability >= 0 AND availability <= 1),
    start_date DATE,
    end_date DATE,
    PRIMARY KEY (project_id, person_id, phase_name),
    FOREIGN KEY (project_id, phase_name) REFERENCES phases(project_id, phase_name)
);

CREATE TABLE evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT REFERENCES people(id),
    skill_id TEXT REFERENCES skills(id),
    evidence_type TEXT,
    url TEXT,  -- Will implement encryption later in security layer
    description TEXT,
    date_achieved DATE,
    verified_by TEXT,
    verification_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE person_skills (
    person_id TEXT REFERENCES people(id),
    skill_id TEXT REFERENCES skills(id),
    base_level REAL CHECK(base_level >= 0 AND base_level <= 5),
    effective_level REAL,
    confidence_low REAL,
    confidence_high REAL,
    last_used DATE,
    rater_id TEXT,
    rating_date DATE,
    PRIMARY KEY (person_id, skill_id)
);

CREATE TABLE incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT REFERENCES projects(id),
    phase_name TEXT,
    incident_type TEXT,
    skill_id TEXT REFERENCES skills(id),
    delay_days INTEGER,
    cost_impact REAL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Additional tables for enhanced functionality
CREATE TABLE project_requirements (
    project_id TEXT REFERENCES projects(id),
    phase_name TEXT,
    skill_id TEXT REFERENCES skills(id),
    required_level REAL CHECK(required_level >= 0 AND required_level <= 5),
    min_level REAL CHECK(min_level >= 0 AND min_level <= 5),
    fte_weeks REAL,  -- Required FTE weeks for this skill in this phase
    criticality REAL CHECK(criticality >= 0 AND criticality <= 1),
    PRIMARY KEY (project_id, phase_name, skill_id),
    FOREIGN KEY (project_id, phase_name) REFERENCES phases(project_id, phase_name)
);

-- Rater reliability tracking for calibration
CREATE TABLE rater_reliability (
    rater_id TEXT,
    skill_category TEXT,
    cohens_kappa REAL,
    agreement_count INTEGER DEFAULT 0,
    total_ratings INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (rater_id, skill_category)
);

-- Audit log for all changes (immutable)
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT NOT NULL,
    action TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    old_value_hash TEXT,
    new_value_hash TEXT
);

-- Indexes for performance
CREATE INDEX idx_assignments_project ON assignments(project_id);
CREATE INDEX idx_assignments_person ON assignments(person_id);
CREATE INDEX idx_person_skills_person ON person_skills(person_id);
CREATE INDEX idx_person_skills_skill ON person_skills(skill_id);
CREATE INDEX idx_evidence_person ON evidence(person_id);
CREATE INDEX idx_evidence_skill ON evidence(skill_id);
CREATE INDEX idx_incidents_project ON incidents(project_id);
CREATE INDEX idx_project_requirements_project ON project_requirements(project_id);
CREATE INDEX idx_audit_log_entity ON audit_log(entity_type, entity_id);
CREATE INDEX idx_audit_log_timestamp ON audit_log(timestamp);

-- Views for common queries
CREATE VIEW person_skill_summary AS
SELECT
    p.id as person_id,
    p.name as person_name,
    s.id as skill_id,
    s.name as skill_name,
    s.category as skill_category,
    ps.base_level,
    ps.effective_level,
    ps.confidence_low,
    ps.confidence_high,
    ps.last_used,
    COUNT(e.id) as evidence_count
FROM people p
JOIN person_skills ps ON p.id = ps.person_id
JOIN skills s ON ps.skill_id = s.id
LEFT JOIN evidence e ON p.id = e.person_id AND s.id = e.skill_id
GROUP BY p.id, s.id;

CREATE VIEW project_coverage_summary AS
SELECT
    pr.project_id,
    p.name as project_name,
    pr.phase_name,
    pr.skill_id,
    s.name as skill_name,
    pr.required_level,
    pr.fte_weeks as demand_fte,
    COALESCE(supply.total_supply_fte, 0) as supply_fte,
    CASE
        WHEN pr.fte_weeks > 0 THEN COALESCE(supply.total_supply_fte, 0) / pr.fte_weeks
        ELSE 0
    END as coverage_ratio
FROM project_requirements pr
JOIN projects p ON pr.project_id = p.id
JOIN skills s ON pr.skill_id = s.id
LEFT JOIN (
    SELECT
        a.project_id,
        'all' as phase_name,  -- Simplified for view
        ps.skill_id,
        SUM(
            CASE
                WHEN ps.effective_level >= 1.0 THEN
                    LEAST(ps.effective_level / 3.0, 1.0) * a.availability
                ELSE 0
            END
        ) as total_supply_fte
    FROM assignments a
    JOIN person_skills ps ON a.person_id = ps.person_id
    GROUP BY a.project_id, ps.skill_id
) supply ON pr.project_id = supply.project_id AND pr.skill_id = supply.skill_id;