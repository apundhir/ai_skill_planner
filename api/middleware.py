#!/usr/bin/env python3
"""
Authentication Middleware for FastAPI
Provides JWT token validation and role-based access control
"""

import sys
import os
from typing import Optional, Set
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import re

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from security.auth import AuthManager

class AuthenticationMiddleware:
    """Middleware to handle JWT authentication and role-based access control"""

    def __init__(self, app):
        self.app = app
        self.auth_manager = AuthManager()

        # Public endpoints that don't require authentication
        self.public_endpoints: Set[str] = {
            "/",
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/auth/login",
            "/static"  # Static files for dashboard
        }

        # Role-based access control rules
        # Maps endpoint patterns to required roles
        self.endpoint_roles = {
            r"/people.*": ["ADMIN", "EXECUTIVE", "MANAGER", "ANALYST", "VIEWER"],
            r"/skills.*": ["ADMIN", "EXECUTIVE", "MANAGER", "ANALYST", "VIEWER"],
            r"/projects.*": ["ADMIN", "EXECUTIVE", "MANAGER", "ANALYST", "VIEWER"],
            r"/assignments.*": ["ADMIN", "EXECUTIVE", "MANAGER", "ANALYST"],
            r"/evidence.*": ["ADMIN", "EXECUTIVE", "MANAGER", "ANALYST"],
            r"/analytics.*": ["ADMIN", "EXECUTIVE", "MANAGER", "ANALYST"],
            r"/gap-analysis.*": ["ADMIN", "EXECUTIVE", "MANAGER", "ANALYST"],
            r"/executive.*": ["ADMIN", "EXECUTIVE"],
            r"/validation.*": ["ADMIN", "EXECUTIVE", "MANAGER"],
            r"/auth/users.*": ["ADMIN"],
            r"/auth/audit-logs.*": ["ADMIN", "EXECUTIVE"],
            r"/auth/security-status.*": ["ADMIN", "EXECUTIVE"]
        }

    async def __call__(self, request: Request, call_next):
        """Process request with authentication and authorization"""
        try:
            path = request.url.path
            method = request.method

            # Skip authentication for public endpoints
            if self._is_public_endpoint(path):
                return await call_next(request)

            # Skip authentication for OPTIONS requests (CORS preflight)
            if method == "OPTIONS":
                return await call_next(request)

            # Extract and validate JWT token
            token = self._extract_token(request)
            if not token:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Authentication required", "error": "missing_token"},
                    headers={"WWW-Authenticate": "Bearer"}
                )

            # Verify token and get user info
            user_payload = self.auth_manager.verify_token(token)
            if not user_payload:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid or expired token", "error": "invalid_token"},
                    headers={"WWW-Authenticate": "Bearer"}
                )

            # Check role-based authorization
            if not self._check_authorization(path, user_payload.get('role')):
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "detail": f"Access denied for role '{user_payload.get('role')}'",
                        "error": "insufficient_permissions"
                    }
                )

            # Add user info to request state for use in endpoints
            request.state.current_user = user_payload

            # Log API access for audit purposes
            self.auth_manager.log_audit_event(
                user_id=user_payload['user_id'],
                action=f"api_{method.lower()}",
                resource=path,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get('user-agent', ''),
                details={"endpoint": path, "method": method}
            )

            return await call_next(request)

        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": f"Authentication middleware error: {str(e)}", "error": "middleware_error"}
            )

    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public and doesn't require authentication"""
        for public_path in self.public_endpoints:
            if path.startswith(public_path):
                return True
        return False

    def _extract_token(self, request: Request) -> Optional[str]:
        """Extract JWT token from Authorization header"""
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return None

        # Expected format: "Bearer <token>"
        if not auth_header.startswith('Bearer '):
            return None

        return auth_header[7:]  # Remove "Bearer " prefix

    def _check_authorization(self, path: str, user_role: str) -> bool:
        """Check if user role has access to the endpoint"""
        if not user_role:
            return False

        # Check each endpoint pattern
        for pattern, allowed_roles in self.endpoint_roles.items():
            if re.match(pattern, path):
                return user_role in allowed_roles

        # Default: allow access if no specific rules defined
        return True

# Rate limiting middleware (basic implementation)
class RateLimitingMiddleware:
    """Basic rate limiting middleware for API protection"""

    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        self.app = app
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # In production, use Redis or similar

    async def __call__(self, request: Request, call_next):
        """Apply rate limiting based on client IP"""
        try:
            client_ip = request.client.host if request.client else "unknown"

            # Skip rate limiting for health checks and static files
            if request.url.path in ["/health", "/static"]:
                return await call_next(request)

            # Simple sliding window rate limiting
            import time
            current_time = time.time()
            window_start = current_time - self.window_seconds

            if client_ip not in self.requests:
                self.requests[client_ip] = []

            # Clean old requests
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if req_time > window_start
            ]

            # Check if rate limit exceeded
            if len(self.requests[client_ip]) >= self.max_requests:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": f"Rate limit exceeded. Max {self.max_requests} requests per {self.window_seconds} seconds",
                        "error": "rate_limit_exceeded"
                    },
                    headers={
                        "X-RateLimit-Limit": str(self.max_requests),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(current_time + self.window_seconds))
                    }
                )

            # Record this request
            self.requests[client_ip].append(current_time)

            # Add rate limit headers
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(self.max_requests)
            response.headers["X-RateLimit-Remaining"] = str(max(0, self.max_requests - len(self.requests[client_ip])))
            response.headers["X-RateLimit-Reset"] = str(int(current_time + self.window_seconds))

            return response

        except Exception as e:
            # Don't block requests on middleware errors
            return await call_next(request)

# Security headers middleware
class SecurityHeadersMiddleware:
    """Add security headers to all responses"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, request: Request, call_next):
        """Add security headers to response"""
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' cdnjs.cloudflare.com cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' cdnjs.cloudflare.com; "
            "img-src 'self' data:; "
            "font-src 'self' cdnjs.cloudflare.com; "
            "connect-src 'self'"
        )

        return response