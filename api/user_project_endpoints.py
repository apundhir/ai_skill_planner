#!/usr/bin/env python3
"""
User-Project Assignment Endpoints
Manages which users have access to which projects based on their roles
"""

import sys
import os
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import sqlite3
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.init_db import get_db_connection

# Create router for user-project assignment endpoints
user_project_router = APIRouter(prefix="/user-projects", tags=["User Project Assignments"])

# Pydantic models
class UserProjectAssignment(BaseModel):
    user_id: str
    username: str
    role: str
    project_id: str
    project_name: str
    access_level: str  # 'read', 'write', 'admin'
    assigned_date: str

class ProjectAccessRequest(BaseModel):
    user_id: str
    project_id: str
    access_level: str = "read"

class UserProjectSummary(BaseModel):
    user_id: str
    username: str
    role: str
    assigned_projects: List[Dict[str, Any]]
    total_projects: int

def init_user_project_tables():
    """Initialize user-project assignment tables if they don't exist"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create user_project_assignments table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_project_assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            username TEXT NOT NULL,
            role TEXT NOT NULL,
            project_id TEXT NOT NULL,
            access_level TEXT DEFAULT 'read' CHECK(access_level IN ('read', 'write', 'admin')),
            assigned_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            assigned_by TEXT,
            UNIQUE(user_id, project_id),
            FOREIGN KEY (project_id) REFERENCES projects(id)
        )
    """)

    # Create index for performance
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_user_project_user ON user_project_assignments(user_id)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_user_project_project ON user_project_assignments(project_id)
    """)

    conn.commit()
    conn.close()

# Initialize tables on import
init_user_project_tables()

def get_user_accessible_projects(user_id: str, role: str) -> List[str]:
    """Get list of project IDs that a user can access based on their role and assignments"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # ADMIN and senior leadership (EXECUTIVE) roles see all projects
    if role in ['ADMIN', 'EXECUTIVE']:
        cursor.execute("SELECT id FROM projects")
        project_ids = [row['id'] for row in cursor.fetchall()]
        conn.close()
        return project_ids

    # Other roles see only assigned projects
    cursor.execute("""
        SELECT DISTINCT project_id
        FROM user_project_assignments
        WHERE user_id = ?
    """, (user_id,))

    project_ids = [row['project_id'] for row in cursor.fetchall()]
    conn.close()
    return project_ids

@user_project_router.post("/assign")
def assign_user_to_project(assignment: ProjectAccessRequest):
    """Assign a user to a project with specified access level"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Verify project exists
        cursor.execute("SELECT id, name FROM projects WHERE id = ?", (assignment.project_id,))
        project = cursor.fetchone()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # For demo, we'll use demo user info
        demo_users = {
            "demo_admin": {"username": "admin", "role": "ADMIN"},
            "demo_executive": {"username": "executive", "role": "EXECUTIVE"},
            "demo_manager": {"username": "manager", "role": "MANAGER"},
            "demo_analyst": {"username": "analyst", "role": "ANALYST"},
        }

        user_info = demo_users.get(assignment.user_id)
        if not user_info:
            raise HTTPException(status_code=404, detail="User not found")

        # Insert or update assignment
        cursor.execute("""
            INSERT OR REPLACE INTO user_project_assignments
            (user_id, username, role, project_id, access_level, assigned_by)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            assignment.user_id,
            user_info["username"],
            user_info["role"],
            assignment.project_id,
            assignment.access_level,
            "system"  # In production, this would be the current user
        ))

        conn.commit()

        return {
            "message": f"User {user_info['username']} assigned to project {project['name']}",
            "assignment": {
                "user_id": assignment.user_id,
                "username": user_info["username"],
                "project_id": assignment.project_id,
                "project_name": project["name"],
                "access_level": assignment.access_level
            }
        }

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Assignment failed: {str(e)}")
    finally:
        conn.close()

@user_project_router.get("/user/{user_id}/projects")
def get_user_projects(user_id: str):
    """Get all projects assigned to a user"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Get user info from demo users
        demo_users = {
            "demo_admin": {"username": "admin", "role": "ADMIN"},
            "demo_executive": {"username": "executive", "role": "EXECUTIVE"},
            "demo_manager": {"username": "manager", "role": "MANAGER"},
            "demo_analyst": {"username": "analyst", "role": "ANALYST"},
        }

        user_info = demo_users.get(user_id)
        if not user_info:
            raise HTTPException(status_code=404, detail="User not found")

        # Get accessible project IDs
        accessible_projects = get_user_accessible_projects(user_id, user_info["role"])

        if not accessible_projects:
            return {
                "user_id": user_id,
                "username": user_info["username"],
                "role": user_info["role"],
                "projects": [],
                "total_projects": 0
            }

        # Get project details
        placeholders = ",".join(["?" for _ in accessible_projects])
        cursor.execute(f"""
            SELECT p.id, p.name, p.complexity, p.regulatory_intensity,
                   p.start_date, p.end_date, p.cost_of_delay_weekly,
                   upa.access_level, upa.assigned_date
            FROM projects p
            LEFT JOIN user_project_assignments upa ON p.id = upa.project_id AND upa.user_id = ?
            WHERE p.id IN ({placeholders})
            ORDER BY p.cost_of_delay_weekly DESC
        """, [user_id] + accessible_projects)

        projects = []
        for row in cursor.fetchall():
            project = dict(row)
            # If no specific assignment (like for ADMIN/EXECUTIVE), set default access
            if not project['access_level']:
                project['access_level'] = 'admin' if user_info["role"] in ['ADMIN', 'EXECUTIVE'] else 'read'
                project['assigned_date'] = None

            projects.append(project)

        return {
            "user_id": user_id,
            "username": user_info["username"],
            "role": user_info["role"],
            "projects": projects,
            "total_projects": len(projects)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user projects: {str(e)}")
    finally:
        conn.close()

@user_project_router.get("/project/{project_id}/users")
def get_project_users(project_id: str):
    """Get all users assigned to a project"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Verify project exists
        cursor.execute("SELECT id, name FROM projects WHERE id = ?", (project_id,))
        project = cursor.fetchone()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Get assigned users
        cursor.execute("""
            SELECT user_id, username, role, access_level, assigned_date, assigned_by
            FROM user_project_assignments
            WHERE project_id = ?
            ORDER BY role, username
        """, (project_id,))

        assignments = [dict(row) for row in cursor.fetchall()]

        # Add ADMIN and EXECUTIVE users who have implicit access
        demo_users = {
            "demo_admin": {"username": "admin", "role": "ADMIN"},
            "demo_executive": {"username": "executive", "role": "EXECUTIVE"},
        }

        # Check if admin/executive are explicitly assigned
        explicitly_assigned = {row['user_id'] for row in assignments}

        for user_id, user_info in demo_users.items():
            if user_id not in explicitly_assigned:
                assignments.append({
                    "user_id": user_id,
                    "username": user_info["username"],
                    "role": user_info["role"],
                    "access_level": "admin",
                    "assigned_date": None,
                    "assigned_by": "implicit"
                })

        return {
            "project_id": project_id,
            "project_name": project["name"],
            "assigned_users": assignments,
            "total_users": len(assignments)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get project users: {str(e)}")
    finally:
        conn.close()

@user_project_router.delete("/user/{user_id}/project/{project_id}")
def remove_user_from_project(user_id: str, project_id: str):
    """Remove a user's assignment from a project"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Cannot remove ADMIN or EXECUTIVE implicit access
        if user_id in ['demo_admin', 'demo_executive']:
            raise HTTPException(
                status_code=403,
                detail="Cannot remove implicit project access for ADMIN or EXECUTIVE roles"
            )

        # Remove assignment
        cursor.execute("""
            DELETE FROM user_project_assignments
            WHERE user_id = ? AND project_id = ?
        """, (user_id, project_id))

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Assignment not found")

        conn.commit()

        return {"message": f"User {user_id} removed from project {project_id}"}

    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to remove assignment: {str(e)}")
    finally:
        conn.close()

@user_project_router.get("/assignments/summary")
def get_assignments_summary():
    """Get summary of all user-project assignments"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Get explicit assignments
        cursor.execute("""
            SELECT upa.user_id, upa.username, upa.role,
                   COUNT(upa.project_id) as assigned_projects,
                   GROUP_CONCAT(p.name) as project_names
            FROM user_project_assignments upa
            JOIN projects p ON upa.project_id = p.id
            GROUP BY upa.user_id, upa.username, upa.role
            ORDER BY upa.role, upa.username
        """)

        explicit_assignments = [dict(row) for row in cursor.fetchall()]

        # Get total project count for implicit assignments
        cursor.execute("SELECT COUNT(*) as total_projects FROM projects")
        total_projects = cursor.fetchone()["total_projects"]

        # Add implicit assignments for ADMIN and EXECUTIVE
        cursor.execute("SELECT name FROM projects ORDER BY name")
        all_project_names = [row["name"] for row in cursor.fetchall()]

        implicit_users = [
            {
                "user_id": "demo_admin",
                "username": "admin",
                "role": "ADMIN",
                "assigned_projects": total_projects,
                "project_names": ",".join(all_project_names),
                "access_type": "implicit"
            },
            {
                "user_id": "demo_executive",
                "username": "executive",
                "role": "EXECUTIVE",
                "assigned_projects": total_projects,
                "project_names": ",".join(all_project_names),
                "access_type": "implicit"
            }
        ]

        # Mark explicit assignments
        for assignment in explicit_assignments:
            assignment["access_type"] = "explicit"

        # Remove duplicates (if admin/exec also have explicit assignments)
        explicit_user_ids = {assignment["user_id"] for assignment in explicit_assignments}
        final_assignments = explicit_assignments.copy()

        for implicit_user in implicit_users:
            if implicit_user["user_id"] not in explicit_user_ids:
                final_assignments.append(implicit_user)

        return {
            "assignments": final_assignments,
            "total_users": len(final_assignments),
            "total_projects": total_projects
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get assignments summary: {str(e)}")
    finally:
        conn.close()

# Initialize some demo assignments
@user_project_router.post("/init-demo-assignments")
def init_demo_assignments():
    """Initialize demo user-project assignments"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Get existing projects
        cursor.execute("SELECT id, name FROM projects LIMIT 4")
        projects = cursor.fetchall()

        if len(projects) < 2:
            raise HTTPException(status_code=404, detail="Need at least 2 projects for demo assignments")

        # Demo assignments:
        # Manager gets assigned to first 2 projects
        # Analyst gets assigned to last 2 projects
        assignments = [
            ("demo_manager", "manager", "MANAGER", projects[0]["id"], "write"),
            ("demo_manager", "manager", "MANAGER", projects[1]["id"], "write"),
            ("demo_analyst", "analyst", "ANALYST", projects[-2]["id"], "read"),
            ("demo_analyst", "analyst", "ANALYST", projects[-1]["id"], "read"),
        ]

        for user_id, username, role, project_id, access_level in assignments:
            cursor.execute("""
                INSERT OR REPLACE INTO user_project_assignments
                (user_id, username, role, project_id, access_level, assigned_by)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, username, role, project_id, access_level, "demo_init"))

        conn.commit()

        return {
            "message": "Demo assignments initialized",
            "assignments_created": len(assignments)
        }

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to initialize demo assignments: {str(e)}")
    finally:
        conn.close()

# Export the router
__all__ = ['user_project_router', 'get_user_accessible_projects']