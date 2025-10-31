"""Structured logging configuration for the API application."""
from __future__ import annotations

import json
import logging
import logging.config
import logging.handlers
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:  # pragma: no cover - only used for typing
    from starlette.requests import Request


class LoggingConfig(BaseSettings):
    """Application logging configuration sourced from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    level: str = Field(default="INFO", alias="LOG_LEVEL")
    sink: str = Field(default="stdout", alias="LOG_SINK")
    json_indent: Optional[int] = Field(default=None, alias="LOG_JSON_INDENT")
    timestamp_format: str = Field(default="iso", alias="LOG_TIMESTAMP_FORMAT")

    @property
    def normalized_level(self) -> str:
        """Return an upper-case logging level accepted by the stdlib logger."""

        return self.level.upper()


_REQUEST_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar("request_context", default={})
_configured = False
_RESERVED_KEYS = {"exc_info", "stack_info", "stacklevel", "extra"}


class JSONFormatter(logging.Formatter):
    """Format log records as JSON, merging contextual data."""

    def __init__(self, *, indent: Optional[int], timestamp_format: str) -> None:
        super().__init__()
        self.indent = indent
        self.timestamp_format = timestamp_format

    def _format_timestamp(self, created: float) -> str:
        timestamp = datetime.fromtimestamp(created, tz=timezone.utc)
        if self.timestamp_format == "iso":
            return timestamp.isoformat()
        return timestamp.strftime(self.timestamp_format)

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self._format_timestamp(record.created),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }

        context: Dict[str, Any] = {}
        request_context = _REQUEST_CONTEXT.get()
        if request_context:
            context.update(request_context)

        record_context = getattr(record, "context_data", None)
        if record_context:
            context.update(record_context)

        if context:
            payload.update(context)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, indent=self.indent, default=str)


def _resolve_handler(config: LoggingConfig) -> dict[str, Any]:
    sink = config.sink.lower()
    handler: Dict[str, Any]
    if sink == "stdout":
        handler = {
            "level": config.normalized_level,
            "class": "logging.StreamHandler",
            "formatter": "json",
            "stream": "ext://sys.stdout",
        }
    elif sink == "stderr":
        handler = {
            "level": config.normalized_level,
            "class": "logging.StreamHandler",
            "formatter": "json",
            "stream": "ext://sys.stderr",
        }
    else:
        path = Path(config.sink).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = {
            "level": config.normalized_level,
            "class": "logging.handlers.WatchedFileHandler",
            "formatter": "json",
            "filename": str(path),
        }
    return handler


def setup_logging(config: Optional[LoggingConfig] = None) -> LoggingConfig:
    """Configure the standard logging module to emit JSON records."""

    global _configured

    if _configured:
        return config or LoggingConfig()

    config = config or LoggingConfig()

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": JSONFormatter,
                    "indent": config.json_indent,
                    "timestamp_format": config.timestamp_format,
                }
            },
            "handlers": {
                "default": _resolve_handler(config),
            },
            "loggers": {
                "": {
                    "handlers": ["default"],
                    "level": config.normalized_level,
                    "propagate": False,
                }
            },
        }
    )

    _configured = True
    return config


class StructuredLogger(logging.LoggerAdapter):
    """Logger adapter that supports contextual key-value pairs."""

    def __init__(self, logger: logging.Logger, extra: Optional[dict[str, Any]] = None) -> None:
        super().__init__(logger, extra or {})

    def bind(self, **kwargs: Any) -> "StructuredLogger":
        extra = dict(self.extra)
        extra.update(kwargs)
        return StructuredLogger(self.logger, extra)

    def process(self, msg: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        extra = kwargs.get("extra")
        if extra is None:
            extra = {}
            kwargs["extra"] = extra

        context_data = extra.setdefault("context_data", {})
        if self.extra:
            context_data.update(self.extra)

        provided_context = kwargs.pop("context", None)
        if isinstance(provided_context, dict):
            context_data.update(provided_context)

        for key in list(kwargs.keys()):
            if key in _RESERVED_KEYS or key == "extra":
                continue
            context_data[key] = kwargs.pop(key)

        return msg, kwargs

    def log(self, level: int, msg: Any, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        if not self.logger.isEnabledFor(level):
            return
        msg, kwargs = self.process(msg, kwargs)
        self.logger.log(level, msg, *args, **kwargs)


def get_logger(name: Optional[str] = None) -> StructuredLogger:
    """Return a structured logger bound to the optional name."""

    base_logger = logging.getLogger(name)
    return StructuredLogger(base_logger)


def bind_request_context(request: "Request", *, request_id: Optional[str] = None) -> None:
    """Attach request context (method, path, client, id) to the current log context."""

    context = dict(_REQUEST_CONTEXT.get())
    context.update({
        "method": request.method,
        "path": request.url.path,
    })
    if request.client:
        context["client_ip"] = request.client.host
    if request_id:
        context["request_id"] = request_id
    _REQUEST_CONTEXT.set(context)


def clear_request_context(*keys: str) -> None:
    """Clear request specific logging context."""

    if not keys:
        _REQUEST_CONTEXT.set({})
        return

    context = dict(_REQUEST_CONTEXT.get())
    for key in keys:
        context.pop(key, None)
    _REQUEST_CONTEXT.set(context)


__all__ = [
    "LoggingConfig",
    "StructuredLogger",
    "setup_logging",
    "get_logger",
    "bind_request_context",
    "clear_request_context",
]
