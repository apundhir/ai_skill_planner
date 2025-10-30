"""Authentication dependencies for FastAPI routes."""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from security.auth import (
    UserProfile,
    UserRole,
    get_authentication_system,
)

_bearer_scheme = HTTPBearer(auto_error=False)


def _decode_token(credentials: HTTPAuthorizationCredentials | None) -> UserProfile:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    auth_system = get_authentication_system()
    user_profile = auth_system.verify_token(credentials.credentials)
    if not user_profile:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user_profile


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> UserProfile:
    """Extract the authenticated user profile from the bearer token."""

    return _decode_token(credentials)


async def verify_admin_role(
    user: UserProfile = Depends(get_current_user),
) -> UserProfile:
    """Ensure the authenticated user has administrative privileges."""

    if user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator privileges required",
        )
    return user


__all__ = ["get_current_user", "verify_admin_role"]
