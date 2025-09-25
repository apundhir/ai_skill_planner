#!/usr/bin/env python3
"""
Authentication and Security API Endpoints
Provides JWT-based authentication, role-based access control, and security management
"""

import sys
import os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import sqlite3

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.init_db import get_db_connection
from security.auth import AuthManager, hash_password

# Initialize router and security
auth_router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()

# Initialize auth manager
auth_manager = AuthManager()

# Request/Response models
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user_info: Dict[str, Any]

class UserCreateRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    role: str = "VIEWER"
    full_name: Optional[str] = None

class UserResponse(BaseModel):
    user_id: str
    username: str
    email: str
    role: str
    full_name: Optional[str]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]

class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str

class AuditLogEntry(BaseModel):
    id: int
    user_id: str
    action: str
    resource: str
    timestamp: datetime
    ip_address: Optional[str]
    user_agent: Optional[str]
    details: Optional[Dict[str, Any]]

# Dependency to get database connection
def get_db():
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()

# Dependency to get current user from token
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Extract and validate current user from JWT token"""
    try:
        token = credentials.credentials
        payload = auth_manager.verify_token(token)

        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return payload
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Role-based authorization decorator
def require_role(required_roles: list):
    """Decorator to require specific roles for endpoint access"""
    def decorator(current_user: Dict = Depends(get_current_user)):
        user_role = current_user.get('role')
        if user_role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {required_roles}, your role: {user_role}"
            )
        return current_user
    return decorator

# Authentication endpoints
@auth_router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest, http_request: Request, db: sqlite3.Connection = Depends(get_db)):
    """Authenticate user and return access token"""
    try:
        # Get client IP and user agent for audit logging
        client_ip = http_request.client.host if http_request.client else None
        user_agent = http_request.headers.get('user-agent', '')

        # Authenticate user
        user_info = auth_manager.authenticate_user(request.username, request.password)

        if not user_info:
            # Log failed login attempt
            auth_manager.log_audit_event(
                user_id=request.username,
                action="login_failed",
                resource="auth",
                ip_address=client_ip,
                user_agent=user_agent,
                details={"reason": "invalid_credentials"}
            )

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        # Generate access token
        token_data = {
            'user_id': user_info['user_id'],
            'username': user_info['username'],
            'role': user_info['role'],
            'exp': datetime.utcnow() + timedelta(hours=24)  # 24-hour expiry
        }

        access_token = auth_manager.create_token(token_data)

        # Update last login time
        cursor = db.cursor()
        cursor.execute("""
            UPDATE users
            SET last_login = ?, login_attempts = 0, locked_until = NULL
            WHERE user_id = ?
        """, (datetime.utcnow(), user_info['user_id']))
        db.commit()

        # Log successful login
        auth_manager.log_audit_event(
            user_id=user_info['user_id'],
            action="login_success",
            resource="auth",
            ip_address=client_ip,
            user_agent=user_agent
        )

        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=86400,  # 24 hours in seconds
            user_info={
                'user_id': user_info['user_id'],
                'username': user_info['username'],
                'role': user_info['role'],
                'full_name': user_info['full_name'],
                'email': user_info['email']
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

@auth_router.post("/logout")
async def logout(current_user: Dict = Depends(get_current_user), http_request: Request = None):
    """Logout user and invalidate token"""
    try:
        # In a production system, you'd invalidate the token in a blacklist
        # For now, we'll just log the logout event

        client_ip = http_request.client.host if http_request and http_request.client else None
        user_agent = http_request.headers.get('user-agent', '') if http_request else ''

        auth_manager.log_audit_event(
            user_id=current_user['user_id'],
            action="logout",
            resource="auth",
            ip_address=client_ip,
            user_agent=user_agent
        )

        return {"message": "Successfully logged out"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Logout failed: {str(e)}"
        )

@auth_router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: Dict = Depends(get_current_user), db: sqlite3.Connection = Depends(get_db)):
    """Get current user information"""
    try:
        cursor = db.cursor()
        cursor.execute("""
            SELECT user_id, username, email, role, full_name, is_active, created_at, last_login
            FROM users
            WHERE user_id = ?
        """, (current_user['user_id'],))

        user_data = cursor.fetchone()
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")

        return UserResponse(**dict(user_data))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user info: {str(e)}"
        )

# User management endpoints (Admin only)
@auth_router.post("/users", response_model=UserResponse)
async def create_user(
    user_data: UserCreateRequest,
    current_user: Dict = Depends(require_role(['ADMIN'])),
    db: sqlite3.Connection = Depends(get_db)
):
    """Create a new user (Admin only)"""
    try:
        cursor = db.cursor()

        # Check if username already exists
        cursor.execute("SELECT user_id FROM users WHERE username = ?", (user_data.username,))
        if cursor.fetchone():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )

        # Check if email already exists
        cursor.execute("SELECT user_id FROM users WHERE email = ?", (user_data.email,))
        if cursor.fetchone():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already exists"
            )

        # Create user
        user_id = f"user_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        password_hash = hash_password(user_data.password)

        cursor.execute("""
            INSERT INTO users (user_id, username, email, password_hash, role, full_name, is_active, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            user_data.username,
            user_data.email,
            password_hash,
            user_data.role,
            user_data.full_name,
            True,
            datetime.utcnow()
        ))
        db.commit()

        # Log user creation
        auth_manager.log_audit_event(
            user_id=current_user['user_id'],
            action="user_created",
            resource=f"user/{user_id}",
            details={"created_user": user_data.username, "role": user_data.role}
        )

        return UserResponse(
            user_id=user_id,
            username=user_data.username,
            email=user_data.email,
            role=user_data.role,
            full_name=user_data.full_name,
            is_active=True,
            created_at=datetime.utcnow(),
            last_login=None
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}"
        )

@auth_router.get("/users")
async def list_users(
    current_user: Dict = Depends(require_role(['ADMIN', 'EXECUTIVE'])),
    db: sqlite3.Connection = Depends(get_db)
):
    """List all users (Admin/Executive only)"""
    try:
        cursor = db.cursor()
        cursor.execute("""
            SELECT user_id, username, email, role, full_name, is_active, created_at, last_login
            FROM users
            ORDER BY created_at DESC
        """)

        users = [UserResponse(**dict(row)) for row in cursor.fetchall()]
        return {"users": users}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list users: {str(e)}"
        )

@auth_router.put("/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    role: str,
    current_user: Dict = Depends(require_role(['ADMIN'])),
    db: sqlite3.Connection = Depends(get_db)
):
    """Update user role (Admin only)"""
    try:
        valid_roles = ['ADMIN', 'EXECUTIVE', 'MANAGER', 'ANALYST', 'VIEWER']
        if role not in valid_roles:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role. Must be one of: {valid_roles}"
            )

        cursor = db.cursor()
        cursor.execute("UPDATE users SET role = ? WHERE user_id = ?", (role, user_id))

        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")

        db.commit()

        # Log role change
        auth_manager.log_audit_event(
            user_id=current_user['user_id'],
            action="role_updated",
            resource=f"user/{user_id}",
            details={"new_role": role}
        )

        return {"message": f"User role updated to {role}"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user role: {str(e)}"
        )

@auth_router.put("/password")
async def change_password(
    password_data: PasswordChangeRequest,
    current_user: Dict = Depends(get_current_user),
    db: sqlite3.Connection = Depends(get_db)
):
    """Change current user's password"""
    try:
        # Verify current password
        user_info = auth_manager.authenticate_user(current_user['username'], password_data.current_password)
        if not user_info:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )

        # Update password
        new_password_hash = hash_password(password_data.new_password)
        cursor = db.cursor()
        cursor.execute("""
            UPDATE users
            SET password_hash = ?, password_changed_at = ?
            WHERE user_id = ?
        """, (new_password_hash, datetime.utcnow(), current_user['user_id']))
        db.commit()

        # Log password change
        auth_manager.log_audit_event(
            user_id=current_user['user_id'],
            action="password_changed",
            resource="auth"
        )

        return {"message": "Password changed successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to change password: {str(e)}"
        )

# Audit and security endpoints
@auth_router.get("/audit-logs")
async def get_audit_logs(
    limit: int = 100,
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    current_user: Dict = Depends(require_role(['ADMIN', 'EXECUTIVE'])),
    db: sqlite3.Connection = Depends(get_db)
):
    """Get audit logs (Admin/Executive only)"""
    try:
        cursor = db.cursor()

        query = """
            SELECT id, user_id, action, resource, timestamp, ip_address, user_agent, details
            FROM audit_logs
        """
        params = []
        conditions = []

        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)

        if action:
            conditions.append("action = ?")
            params.append(action)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        logs = [AuditLogEntry(**dict(row)) for row in cursor.fetchall()]

        return {"audit_logs": logs, "total_count": len(logs)}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get audit logs: {str(e)}"
        )

@auth_router.get("/security-status")
async def get_security_status(
    current_user: Dict = Depends(require_role(['ADMIN', 'EXECUTIVE'])),
    db: sqlite3.Connection = Depends(get_db)
):
    """Get system security status (Admin/Executive only)"""
    try:
        cursor = db.cursor()

        # Get user statistics
        cursor.execute("""
            SELECT
                COUNT(*) as total_users,
                COUNT(CASE WHEN is_active = 1 THEN 1 END) as active_users,
                COUNT(CASE WHEN locked_until IS NOT NULL AND locked_until > datetime('now') THEN 1 END) as locked_users,
                COUNT(CASE WHEN last_login IS NULL THEN 1 END) as never_logged_in
            FROM users
        """)
        user_stats = dict(cursor.fetchone())

        # Get recent login attempts
        cursor.execute("""
            SELECT
                COUNT(*) as total_attempts,
                COUNT(CASE WHEN action = 'login_success' THEN 1 END) as successful_logins,
                COUNT(CASE WHEN action = 'login_failed' THEN 1 END) as failed_logins
            FROM audit_logs
            WHERE action IN ('login_success', 'login_failed')
              AND timestamp >= datetime('now', '-24 hours')
        """)
        login_stats = dict(cursor.fetchone())

        # Get system activity
        cursor.execute("""
            SELECT action, COUNT(*) as count
            FROM audit_logs
            WHERE timestamp >= datetime('now', '-24 hours')
            GROUP BY action
            ORDER BY count DESC
            LIMIT 10
        """)
        activity_stats = [{"action": row["action"], "count": row["count"]} for row in cursor.fetchall()]

        return {
            "user_statistics": user_stats,
            "login_statistics": login_stats,
            "recent_activity": activity_stats,
            "system_status": "operational",
            "last_updated": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get security status: {str(e)}"
        )