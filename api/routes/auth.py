"""Authentication API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from api.dependencies.auth import get_current_user
from security.auth import UserProfile, get_authentication_system

router = APIRouter(prefix="/auth", tags=["authentication"])
_auth_system = get_authentication_system()


class LoginRequest(BaseModel):
    username: str
    password: str


class UserInfo(BaseModel):
    user_id: str
    username: str
    full_name: str
    email: str
    role: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user: UserInfo


class MeResponse(BaseModel):
    user: UserInfo


def _serialize_user(profile: UserProfile) -> UserInfo:
    return UserInfo(
        user_id=profile.id,
        username=profile.username,
        full_name=profile.full_name,
        email=profile.email,
        role=profile.role.value,
    )


@router.post("/login", response_model=LoginResponse)
async def login(payload: LoginRequest, request: Request) -> LoginResponse:
    """Authenticate the user and issue a signed JWT access token."""

    client_host = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    profile = _auth_system.authenticate_user(
        payload.username,
        payload.password,
        ip_address=client_host,
    )
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    token_response = _auth_system.create_access_token(
        profile,
        ip_address=client_host,
        user_agent=user_agent,
    )

    return LoginResponse(
        access_token=token_response["access_token"],
        token_type=token_response["token_type"],
        expires_in=token_response["expires_in"],
        user=_serialize_user(profile),
    )


@router.get("/me", response_model=MeResponse)
async def read_current_user(user: UserProfile = Depends(get_current_user)) -> MeResponse:
    """Return the authenticated user's profile."""

    return MeResponse(user=_serialize_user(user))


__all__ = ["router"]
