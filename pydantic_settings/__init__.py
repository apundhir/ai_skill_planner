"""Lightweight fallback implementation of Pydantic's settings helpers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel


def SettingsConfigDict(**kwargs: Any) -> Dict[str, Any]:
    return dict(**kwargs)


def _load_env_file(path: str | Path, encoding: str = "utf-8") -> Dict[str, str]:
    values: Dict[str, str] = {}
    file_path = Path(path)
    if not file_path.exists():
        return values

    for line in file_path.read_text(encoding=encoding).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


class BaseSettings(BaseModel):
    """Minimal BaseSettings compatible with the subset we require."""

    model_config: SettingsConfigDict = {}

    def __init__(self, **data: Any) -> None:  # type: ignore[override]
        config = getattr(self.__class__, "model_config", {}) or {}
        case_sensitive = bool(config.get("case_sensitive", False))
        env_file = config.get("env_file")
        env_file_encoding = config.get("env_file_encoding", "utf-8")

        file_values: Dict[str, str] = {}
        if env_file:
            file_values = _load_env_file(env_file, env_file_encoding)

        env_values: Dict[str, Any] = {}
        for field_name, field in self.model_fields.items():  # type: ignore[attr-defined]
            alias = field.alias or field_name
            candidates = [alias]
            if not case_sensitive:
                candidates.extend({alias.lower(), alias.upper()})

            value = None
            for candidate in candidates:
                if candidate in os.environ:
                    value = os.environ[candidate]
                    break
                if candidate in file_values:
                    value = file_values[candidate]
                    break

            if value is not None:
                env_values[field_name] = value

        merged = {**env_values, **data}
        super().__init__(**merged)


__all__ = ["BaseSettings", "SettingsConfigDict"]

