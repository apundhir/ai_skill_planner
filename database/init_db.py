#!/usr/bin/env python3
"""
Database initialization script for AI Skill Planner
Creates SQLite database with schema from PRD v2.0
"""

import sqlite3
import os
from pathlib import Path

def init_database(db_path: str = "ai_skill_planner.db") -> sqlite3.Connection:
    """
    Initialize SQLite database with schema

    Args:
        db_path: Path to SQLite database file

    Returns:
        sqlite3.Connection: Database connection
    """
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    schema_path = script_dir / "schema.sql"

    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database: {db_path}")

    # Create new database connection
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints

    # Read and execute schema
    with open(schema_path, 'r') as f:
        schema_sql = f.read()

    # Execute schema (split by semicolon to handle multiple statements)
    for statement in schema_sql.split(';'):
        statement = statement.strip()
        if statement:  # Skip empty statements
            try:
                conn.execute(statement)
            except sqlite3.Error as e:
                print(f"Error executing statement: {statement[:50]}...")
                print(f"Error: {e}")
                raise

    conn.commit()
    print(f"Database initialized successfully: {db_path}")

    # Verify tables were created
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"Created tables: {', '.join(tables)}")

    return conn

def get_db_path(db_name: str = "ai_skill_planner.db") -> str:
    """
    Get the path to the database file

    Args:
        db_name: Name of the database file

    Returns:
        str: Full path to database file
    """
    overridden_path = os.getenv("AI_SKILL_PLANNER_DB_PATH")
    if overridden_path:
        return overridden_path

    return os.path.join(os.path.dirname(__file__), db_name)

def get_db_connection(db_path: str = None) -> sqlite3.Connection:
    """
    Get database connection with proper settings

    Args:
        db_path: Path to SQLite database file (optional, defaults to standard location)

    Returns:
        sqlite3.Connection: Database connection
    """
    if db_path is None:
        db_path = get_db_path()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
    return conn

if __name__ == "__main__":
    # Initialize database when run as script
    db_conn = init_database()

    # Test connection
    cursor = db_conn.cursor()
    cursor.execute("SELECT COUNT(*) as table_count FROM sqlite_master WHERE type='table'")
    result = cursor.fetchone()
    print(f"Database contains {result[0]} tables")

    db_conn.close()
    print("Database initialization complete!")