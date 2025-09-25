#!/usr/bin/env python3
"""
Demo Authentication Endpoints
Simplified authentication for demonstration purposes
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Create demo auth router
demo_auth_router = APIRouter(prefix="/auth", tags=["demo-authentication"])

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user_info: dict

@demo_auth_router.post("/login", response_model=LoginResponse)
async def demo_login(request: LoginRequest):
    """Demo login endpoint - accepts common demo credentials"""

    # Demo credentials
    demo_users = {
        "admin": {"password": "admin123", "role": "ADMIN", "name": "System Administrator"},
        "executive": {"password": "exec123", "role": "EXECUTIVE", "name": "Chief Technology Officer"},
        "manager": {"password": "manager123", "role": "MANAGER", "name": "Engineering Manager"},
        "analyst": {"password": "analyst123", "role": "ANALYST", "name": "Data Analyst"},
    }

    # Check credentials
    if request.username in demo_users:
        user = demo_users[request.username]
        if request.password == user["password"]:
            # Return demo token
            return LoginResponse(
                access_token="demo_token_12345",
                token_type="bearer",
                expires_in=86400,  # 24 hours
                user_info={
                    "user_id": f"demo_{request.username}",
                    "username": request.username,
                    "role": user["role"],
                    "full_name": user["name"],
                    "email": f"{request.username}@company.com"
                }
            )

    raise HTTPException(status_code=401, detail="Invalid credentials")

@demo_auth_router.post("/logout")
async def demo_logout():
    """Demo logout endpoint"""
    return {"message": "Logged out successfully"}

@demo_auth_router.get("/me")
async def demo_get_user():
    """Demo user info endpoint"""
    return {
        "user_id": "demo_executive",
        "username": "executive",
        "role": "EXECUTIVE",
        "full_name": "Chief Technology Officer",
        "email": "executive@company.com",
        "is_active": True
    }