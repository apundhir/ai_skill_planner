"""Application settings utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import os
from typing import List, Optional


@dataclass(frozen=True)
class Settings:
    """Container for runtime configuration loaded from environment variables."""

    app_env: str = "development"
    jwt_secret_key: Optional[str] = None
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_minutes: int = 60 * 24 * 3  # 3 days
    cors_allowed_origins: List[str] = field(default_factory=list)

    @property
    def environment(self) -> str:
        """Return the normalized environment name."""
        return self.app_env.lower()


def _parse_origins(raw_value: Optional[str], environment: str) -> List[str]:
    if raw_value:
        origins = [origin.strip() for origin in raw_value.split(",") if origin.strip()]
        return origins

    if environment == "development":
        return ["http://localhost:3000", "http://127.0.0.1:3000"]

    return []


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from environment variables with sensible defaults."""

    environment = os.getenv("APP_ENV", "development").strip() or "development"
    access_token_expire = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    refresh_token_expire = int(os.getenv("REFRESH_TOKEN_EXPIRE_MINUTES", str(60 * 24 * 3)))

    return Settings(
        app_env=environment,
        jwt_secret_key=os.getenv("JWT_SECRET_KEY"),
        jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
        access_token_expire_minutes=access_token_expire,
        refresh_token_expire_minutes=refresh_token_expire,
        cors_allowed_origins=_parse_origins(os.getenv("CORS_ALLOWED_ORIGINS"), environment.lower()),
    )


__all__ = ["Settings", "get_settings"]
