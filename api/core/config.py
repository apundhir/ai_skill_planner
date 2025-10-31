"""Centralised application configuration using Pydantic settings."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB_PATH = _PROJECT_ROOT / "database" / "ai_skill_planner.db"


class FeatureFlags(BaseModel):
    """Feature flag configuration exposed to the rest of the application."""

    enable_demo_data: bool = False
    enable_gap_analysis: bool = True
    enable_capacity_planning: bool = True


class SecretSettings(BaseModel):
    """Runtime secrets loaded from the environment."""

    jwt_secret_key: Optional[str] = None


class AppConfig(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    app_env: str = Field(default="development", alias="APP_ENV")
    database_url: str = Field(  # type: ignore[assignment]
        default_factory=lambda: f"sqlite:///{_DEFAULT_DB_PATH.as_posix()}",
        alias="DATABASE_URL",
    )
    cors_allowed_origins: List[str] = Field(default_factory=list, alias="CORS_ALLOWED_ORIGINS")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=60, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_minutes: int = Field(
        default=60 * 24 * 3,
        alias="REFRESH_TOKEN_EXPIRE_MINUTES",
    )
    jwt_secret_key: Optional[str] = Field(default=None, alias="JWT_SECRET_KEY")
    feature_enable_demo_data: bool = Field(default=False, alias="FEATURE_ENABLE_DEMO_DATA")
    feature_enable_gap_analysis: bool = Field(default=True, alias="FEATURE_ENABLE_GAP_ANALYSIS")
    feature_enable_capacity_planning: bool = Field(
        default=True,
        alias="FEATURE_ENABLE_CAPACITY_PLANNING",
    )

    @field_validator("database_url", mode="before")
    @classmethod
    def _fallback_to_legacy_path(cls, value: Optional[str]) -> str:
        if value and str(value).strip():
            return str(value)
        legacy_path = os.getenv("AI_SKILL_PLANNER_DB_PATH")
        if legacy_path:
            path = Path(legacy_path).expanduser()
            return f"sqlite:///{path.as_posix()}"
        return f"sqlite:///{_DEFAULT_DB_PATH.as_posix()}"

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: Optional[str | List[str]]) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            origins = [origin.strip() for origin in value.split(",") if origin.strip()]
            return origins
        return list(value)

    @field_validator("cors_allowed_origins", mode="after")
    @classmethod
    def _default_origins(cls, value: List[str], info) -> List[str]:
        if value:
            return value
        environment = str(info.data.get("app_env", "development")).lower()
        if environment == "development":
            return ["http://localhost:3000", "http://127.0.0.1:3000"]
        return []

    @computed_field
    def environment(self) -> str:
        """Normalized environment name."""

        return self.app_env.lower()

    @computed_field
    def feature_flags(self) -> FeatureFlags:
        """Return strongly-typed feature flags."""

        return FeatureFlags(
            enable_demo_data=self.feature_enable_demo_data,
            enable_gap_analysis=self.feature_enable_gap_analysis,
            enable_capacity_planning=self.feature_enable_capacity_planning,
        )

    @computed_field
    def secrets(self) -> SecretSettings:
        """Expose sensitive configuration values."""

        return SecretSettings(jwt_secret_key=self.jwt_secret_key)

    @computed_field
    def database_path(self) -> Optional[Path]:
        """Return the filesystem path for SQLite databases when available."""

        return _extract_sqlite_path(self.database_url)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load configuration, caching the result for reuse."""

    return AppConfig()  # type: ignore[call-arg]


def _extract_sqlite_path(database_url: str) -> Optional[Path]:
    parsed = urlparse(database_url)
    scheme = parsed.scheme or "sqlite"

    if scheme != "sqlite":
        return None

    if parsed.path in (":memory:", "/:memory:"):
        return None

    if parsed.netloc and not parsed.path:
        path = parsed.netloc
    else:
        path = parsed.path

    if not parsed.netloc:
        if path.startswith("//"):
            path = path[1:]
        elif path.startswith("/"):
            path = path[1:]
    else:
        path = f"{parsed.netloc}{path}"

    if not path:
        return None

    db_path = Path(path)
    if not db_path.is_absolute():
        db_path = (_PROJECT_ROOT / db_path).resolve()

    return db_path


__all__ = ["AppConfig", "FeatureFlags", "SecretSettings", "get_config"]

