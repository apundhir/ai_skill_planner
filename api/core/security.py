"""Security-specific configuration helpers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from secrets import token_urlsafe

from .settings import get_settings


@dataclass(frozen=True)
class SecurityConfig:
    jwt_secret_key: str
    jwt_algorithm: str
    access_token_expire_minutes: int
    refresh_token_expire_minutes: int


@lru_cache(maxsize=1)
def get_security_config() -> SecurityConfig:
    """Return JWT configuration sourced from environment variables."""

    settings = get_settings()
    secret_key = settings.jwt_secret_key

    if not secret_key:
        if settings.environment == "development":
            secret_key = token_urlsafe(64)
        else:
            raise RuntimeError("JWT_SECRET_KEY must be set outside of development mode")

    return SecurityConfig(
        jwt_secret_key=secret_key,
        jwt_algorithm=settings.jwt_algorithm,
        access_token_expire_minutes=settings.access_token_expire_minutes,
        refresh_token_expire_minutes=settings.refresh_token_expire_minutes,
    )


__all__ = ["SecurityConfig", "get_security_config"]
