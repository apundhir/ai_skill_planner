#!/usr/bin/env python3
"""
Authentication and Authorization System
Implements JWT-based authentication with role-based access control
Based on PRD specifications for Milestone 5
"""

import sys
import os
import hashlib
import secrets
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from enum import Enum

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.init_db import get_db_connection

class UserRole(Enum):
    ADMIN = "ADMIN"
    EXECUTIVE = "EXECUTIVE"
    MANAGER = "MANAGER"
    ANALYST = "ANALYST"
    VIEWER = "VIEWER"

class Permission(Enum):
    # Data access permissions
    READ_SKILLS = "read:skills"
    WRITE_SKILLS = "write:skills"
    READ_PEOPLE = "read:people"
    WRITE_PEOPLE = "write:people"
    READ_PROJECTS = "read:projects"
    WRITE_PROJECTS = "write:projects"

    # Analysis permissions
    READ_ANALYTICS = "read:analytics"
    READ_FINANCIAL = "read:financial"
    WRITE_FINANCIAL = "write:financial"
    READ_VALIDATION = "read:validation"
    WRITE_VALIDATION = "write:validation"

    # System permissions
    ADMIN_USERS = "admin:users"
    ADMIN_SYSTEM = "admin:system"
    AUDIT_LOGS = "read:audit"

    # Executive permissions
    EXECUTIVE_DASHBOARD = "read:executive"
    STRATEGIC_DECISIONS = "write:strategic"

class UserCredentials(BaseModel):
    username: str
    password: str

class UserProfile(BaseModel):
    id: str
    username: str
    full_name: str
    email: str
    role: UserRole
    permissions: List[Permission]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_profile: UserProfile

class AuthenticationSystem:
    """
    Comprehensive authentication and authorization system
    with JWT tokens and role-based access control
    """

    def __init__(self):
        # JWT configuration
        self.SECRET_KEY = os.getenv("JWT_SECRET_KEY", self._generate_secret_key())
        self.ALGORITHM = "HS256"
        self.ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours

        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        # Security bearer
        self.security = HTTPBearer()

        # Role permissions mapping
        self.ROLE_PERMISSIONS = {
            UserRole.ADMIN: [p for p in Permission],  # All permissions
            UserRole.EXECUTIVE: [
                Permission.READ_SKILLS, Permission.READ_PEOPLE, Permission.READ_PROJECTS,
                Permission.READ_ANALYTICS, Permission.READ_FINANCIAL, Permission.READ_VALIDATION,
                Permission.EXECUTIVE_DASHBOARD, Permission.STRATEGIC_DECISIONS,
                Permission.AUDIT_LOGS
            ],
            UserRole.MANAGER: [
                Permission.READ_SKILLS, Permission.WRITE_SKILLS,
                Permission.READ_PEOPLE, Permission.WRITE_PEOPLE,
                Permission.READ_PROJECTS, Permission.WRITE_PROJECTS,
                Permission.READ_ANALYTICS, Permission.READ_FINANCIAL,
                Permission.READ_VALIDATION
            ],
            UserRole.ANALYST: [
                Permission.READ_SKILLS, Permission.READ_PEOPLE, Permission.READ_PROJECTS,
                Permission.READ_ANALYTICS, Permission.READ_FINANCIAL,
                Permission.READ_VALIDATION, Permission.WRITE_VALIDATION
            ],
            UserRole.VIEWER: [
                Permission.READ_SKILLS, Permission.READ_PEOPLE, Permission.READ_PROJECTS,
                Permission.READ_ANALYTICS
            ]
        }

        # Initialize database
        self._initialize_auth_tables()
        self._create_default_users()

    def _generate_secret_key(self) -> str:
        """Generate a secure secret key for JWT signing"""
        return secrets.token_urlsafe(32)

    def _initialize_auth_tables(self):
        """Initialize authentication database tables"""
        conn = get_db_connection()
        cursor = conn.cursor()

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_login TEXT,
                failed_login_attempts INTEGER DEFAULT 0,
                locked_until TEXT
            )
        """)

        # User sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token_jti TEXT UNIQUE NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                expires_at TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        # Check if audit_logs table exists, create new one if needed
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='audit_logs'")
        if not cursor.fetchone():
            cursor.execute("""
                CREATE TABLE audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    resource_id TEXT,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)

        # Also check old table name for compatibility
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='audit_log'")
        if cursor.fetchone():
            # Rename old table to new name if it exists
            cursor.execute("ALTER TABLE audit_log RENAME TO audit_logs_old")
            cursor.execute("""
                CREATE TABLE audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    resource_id TEXT,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)

        conn.commit()
        conn.close()

    def _create_default_users(self):
        """Create default system users"""
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if admin user exists
        cursor.execute("SELECT id FROM users WHERE username = 'admin'")
        if not cursor.fetchone():
            # Create default admin user
            admin_id = self._generate_user_id()
            admin_password = self.hash_password("admin123")  # Default password

            cursor.execute("""
                INSERT INTO users (id, username, full_name, email, password_hash, role)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                admin_id, "admin", "System Administrator", "admin@company.com",
                admin_password, UserRole.ADMIN.value
            ))

        # Create default executive user
        cursor.execute("SELECT id FROM users WHERE username = 'executive'")
        if not cursor.fetchone():
            exec_id = self._generate_user_id()
            exec_password = self.hash_password("exec123")

            cursor.execute("""
                INSERT INTO users (id, username, full_name, email, password_hash, role)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                exec_id, "executive", "Chief Technology Officer", "cto@company.com",
                exec_password, UserRole.EXECUTIVE.value
            ))

        conn.commit()
        conn.close()

    def _generate_user_id(self) -> str:
        """Generate unique user ID"""
        return f"user_{secrets.token_hex(8)}"

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)

    def authenticate_user(self, username: str, password: str,
                         ip_address: Optional[str] = None) -> Optional[UserProfile]:
        """Authenticate user credentials"""
        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            # Get user from database
            cursor.execute("""
                SELECT id, username, full_name, email, password_hash, role,
                       is_active, created_at, last_login, failed_login_attempts, locked_until
                FROM users
                WHERE username = ? AND is_active = 1
            """, (username,))

            user_data = cursor.fetchone()
            if not user_data:
                self._log_audit_event(None, "LOGIN_FAILED", "authentication", username,
                                    f"User not found: {username}", ip_address)
                return None

            user_dict = dict(user_data)

            # Check if account is locked
            if user_dict['locked_until']:
                locked_until = datetime.fromisoformat(user_dict['locked_until'])
                if datetime.now() < locked_until:
                    self._log_audit_event(user_dict['id'], "LOGIN_BLOCKED", "authentication",
                                        username, "Account locked", ip_address)
                    raise HTTPException(
                        status_code=status.HTTP_423_LOCKED,
                        detail="Account is locked due to too many failed attempts"
                    )

            # Verify password
            if not self.verify_password(password, user_dict['password_hash']):
                # Increment failed attempts
                failed_attempts = user_dict['failed_login_attempts'] + 1

                # Lock account after 5 failed attempts
                if failed_attempts >= 5:
                    locked_until = (datetime.now() + timedelta(minutes=30)).isoformat()
                    cursor.execute("""
                        UPDATE users
                        SET failed_login_attempts = ?, locked_until = ?
                        WHERE id = ?
                    """, (failed_attempts, locked_until, user_dict['id']))
                else:
                    cursor.execute("""
                        UPDATE users
                        SET failed_login_attempts = ?
                        WHERE id = ?
                    """, (failed_attempts, user_dict['id']))

                conn.commit()
                self._log_audit_event(user_dict['id'], "LOGIN_FAILED", "authentication",
                                    username, f"Invalid password (attempt {failed_attempts})", ip_address)
                return None

            # Reset failed attempts on successful login
            cursor.execute("""
                UPDATE users
                SET failed_login_attempts = 0, locked_until = NULL, last_login = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), user_dict['id']))

            conn.commit()

            # Create user profile
            user_profile = UserProfile(
                id=user_dict['id'],
                username=user_dict['username'],
                full_name=user_dict['full_name'],
                email=user_dict['email'],
                role=UserRole(user_dict['role']),
                permissions=self.ROLE_PERMISSIONS[UserRole(user_dict['role'])],
                is_active=bool(user_dict['is_active']),
                created_at=datetime.fromisoformat(user_dict['created_at']),
                last_login=datetime.fromisoformat(user_dict['last_login']) if user_dict['last_login'] else None
            )

            self._log_audit_event(user_dict['id'], "LOGIN_SUCCESS", "authentication",
                                username, "Successful login", ip_address)

            return user_profile

        except HTTPException:
            raise
        except Exception as e:
            self._log_audit_event(None, "LOGIN_ERROR", "authentication", username,
                                f"Authentication error: {str(e)}", ip_address)
            return None
        finally:
            conn.close()

    def create_access_token(self, user_profile: UserProfile,
                          ip_address: Optional[str] = None,
                          user_agent: Optional[str] = None) -> Dict[str, Any]:
        """Create JWT access token"""

        # Token expiration
        expire = datetime.utcnow() + timedelta(minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES)

        # JWT payload
        payload = {
            "sub": user_profile.id,
            "username": user_profile.username,
            "role": user_profile.role.value,
            "permissions": [p.value for p in user_profile.permissions],
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": secrets.token_hex(16)  # Unique token ID
        }

        # Generate token
        token = jwt.encode(payload, self.SECRET_KEY, algorithm=self.ALGORITHM)

        # Store session
        self._store_user_session(user_profile.id, payload["jti"], expire,
                                ip_address, user_agent)

        self._log_audit_event(user_profile.id, "TOKEN_CREATED", "authentication",
                            user_profile.username, "Access token generated", ip_address)

        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": self.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "user_profile": user_profile
        }

    def verify_token(self, token: str) -> Optional[UserProfile]:
        """Verify JWT token and return user profile"""
        try:
            # Decode token
            payload = jwt.decode(token, self.SECRET_KEY, algorithms=[self.ALGORITHM])

            # Check if session is still active
            if not self._is_session_active(payload.get("jti")):
                return None

            # Get user from database
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, username, full_name, email, role, is_active, created_at, last_login
                FROM users
                WHERE id = ? AND is_active = 1
            """, (payload.get("sub"),))

            user_data = cursor.fetchone()
            conn.close()

            if not user_data:
                return None

            user_dict = dict(user_data)

            return UserProfile(
                id=user_dict['id'],
                username=user_dict['username'],
                full_name=user_dict['full_name'],
                email=user_dict['email'],
                role=UserRole(user_dict['role']),
                permissions=self.ROLE_PERMISSIONS[UserRole(user_dict['role'])],
                is_active=bool(user_dict['is_active']),
                created_at=datetime.fromisoformat(user_dict['created_at']),
                last_login=datetime.fromisoformat(user_dict['last_login']) if user_dict['last_login'] else None
            )

        except jwt.ExpiredSignatureError:
            return None
        except jwt.JWTError:
            return None
        except Exception:
            return None

    def logout_user(self, token: str, ip_address: Optional[str] = None) -> bool:
        """Logout user by invalidating token"""
        try:
            payload = jwt.decode(token, self.SECRET_KEY, algorithms=[self.ALGORITHM])
            jti = payload.get("jti")
            user_id = payload.get("sub")

            # Deactivate session
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE user_sessions
                SET is_active = 0
                WHERE token_jti = ?
            """, (jti,))

            conn.commit()
            conn.close()

            self._log_audit_event(user_id, "LOGOUT", "authentication",
                                payload.get("username"), "User logged out", ip_address)

            return True

        except Exception:
            return False

    def require_permission(self, required_permission: Permission):
        """Dependency to require specific permission"""
        def permission_checker(current_user: UserProfile = Depends(self.get_current_user)):
            if required_permission not in current_user.permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission required: {required_permission.value}"
                )
            return current_user
        return permission_checker

    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """Dependency to get current authenticated user"""
        token = credentials.credentials
        user_profile = self.verify_token(token)

        if not user_profile:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"}
            )

        return user_profile

    def _store_user_session(self, user_id: str, jti: str, expires_at: datetime,
                          ip_address: Optional[str] = None,
                          user_agent: Optional[str] = None):
        """Store user session in database"""
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO user_sessions (id, user_id, token_jti, expires_at, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            f"session_{secrets.token_hex(8)}", user_id, jti, expires_at.isoformat(),
            ip_address, user_agent
        ))

        conn.commit()
        conn.close()

    def _is_session_active(self, jti: str) -> bool:
        """Check if session is still active"""
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT is_active, expires_at
            FROM user_sessions
            WHERE token_jti = ?
        """, (jti,))

        session_data = cursor.fetchone()
        conn.close()

        if not session_data:
            return False

        if not session_data['is_active']:
            return False

        expires_at = datetime.fromisoformat(session_data['expires_at'])
        if datetime.now() > expires_at:
            return False

        return True

    def _log_audit_event(self, user_id: Optional[str], action: str, resource: str,
                        resource_id: Optional[str] = None, details: Optional[str] = None,
                        ip_address: Optional[str] = None, user_agent: Optional[str] = None):
        """Log audit event"""
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO audit_logs (user_id, action, resource, resource_id, details, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, action, resource, resource_id, details, ip_address, user_agent))

        conn.commit()
        conn.close()

    def get_audit_logs(self, user_id: Optional[str] = None,
                      days: int = 30, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit logs for security monitoring"""
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
            SELECT al.*, u.username, u.full_name
            FROM audit_logs al
            LEFT JOIN users u ON al.user_id = u.id
            WHERE al.timestamp > date('now', '-{} days')
        """.format(days)

        params = []
        if user_id:
            query += " AND al.user_id = ?"
            params.append(user_id)

        query += " ORDER BY al.timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        logs = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return logs

    def get_active_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get active user sessions"""
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
            SELECT us.*, u.username, u.full_name
            FROM user_sessions us
            JOIN users u ON us.user_id = u.id
            WHERE us.is_active = 1 AND us.expires_at > datetime('now')
        """

        params = []
        if user_id:
            query += " AND us.user_id = ?"
            params.append(user_id)

        query += " ORDER BY us.created_at DESC"

        cursor.execute(query, params)
        sessions = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return sessions

# Global authentication instance
auth_system = AuthenticationSystem()

class AuthManager:
    """Compatibility wrapper for AuthenticationSystem"""

    def __init__(self):
        self.auth_system = auth_system

    def create_user(self, username: str, email: str, password: str, role: str = "VIEWER", full_name: str = None):
        """Create a new user"""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Check if user exists
            cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
            if cursor.fetchone():
                return False

            user_id = self.auth_system._generate_user_id()
            password_hash = self.auth_system.hash_password(password)

            cursor.execute("""
                INSERT INTO users (id, username, full_name, email, password_hash, role, is_active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, username, full_name or username, email, password_hash,
                role, True, datetime.now().isoformat()
            ))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f"User creation failed: {e}")
            return False

    def authenticate_user(self, username: str, password: str):
        """Authenticate user and return user info dict"""
        user_profile = self.auth_system.authenticate_user(username, password)
        if user_profile:
            return {
                'user_id': user_profile.id,
                'username': user_profile.username,
                'email': user_profile.email,
                'role': user_profile.role.value,
                'full_name': user_profile.full_name
            }
        return None

    def create_token(self, token_data: Dict[str, Any]) -> str:
        """Create JWT token from token data"""
        user_profile = UserProfile(
            id=token_data['user_id'],
            username=token_data['username'],
            full_name="",
            email="",
            role=UserRole(token_data['role']),
            permissions=self.auth_system.ROLE_PERMISSIONS[UserRole(token_data['role'])],
            is_active=True,
            created_at=datetime.now()
        )

        token_response = self.auth_system.create_access_token(user_profile)
        return token_response['access_token']

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify token and return payload"""
        try:
            payload = jwt.decode(token, self.auth_system.SECRET_KEY, algorithms=[self.auth_system.ALGORITHM])
            if self.auth_system._is_session_active(payload.get("jti")):
                return {
                    'user_id': payload.get('sub'),
                    'username': payload.get('username'),
                    'role': payload.get('role')
                }
        except Exception:
            pass
        return None

    def log_audit_event(self, user_id: str, action: str, resource: str, ip_address: str = None,
                       user_agent: str = None, details: Dict[str, Any] = None):
        """Log audit event"""
        details_str = json.dumps(details) if details else None
        self.auth_system._log_audit_event(
            user_id, action, resource, None, details_str, ip_address, user_agent
        )

# Helper function for password hashing
def hash_password(password: str) -> str:
    """Hash password using the auth system"""
    return auth_system.hash_password(password)

# Export convenience functions
def get_current_user():
    """Get current authenticated user dependency"""
    return auth_system.get_current_user

def require_permission(permission: Permission):
    """Require specific permission dependency"""
    return auth_system.require_permission(permission)

def require_role(role: UserRole):
    """Require specific role dependency"""
    def role_checker(current_user: UserProfile = Depends(get_current_user())):
        if current_user.role != role and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role required: {role.value}"
            )
        return current_user
    return role_checker

__all__ = ['AuthenticationSystem', 'AuthManager', 'UserRole', 'Permission', 'UserProfile', 'auth_system',
           'get_current_user', 'require_permission', 'require_role', 'hash_password']