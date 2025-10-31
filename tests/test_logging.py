import io
import json
import logging

from api.core.logging import JSONFormatter, get_logger, setup_logging


def test_setup_logging_emits_json():
    config = setup_logging()
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(JSONFormatter(indent=None, timestamp_format=config.timestamp_format))

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    try:
        logger = get_logger(__name__)
        logger.info("test_event", foo="bar")
    finally:
        root_logger.removeHandler(handler)

    lines = [line.strip() for line in buffer.getvalue().splitlines() if line.strip()]
    assert lines, "No JSON log output captured"
    payload = json.loads(lines[-1])
    assert payload["event"] == "test_event"
    assert payload["foo"] == "bar"
    assert payload["level"] == config.normalized_level.lower()


def test_setup_logging_is_idempotent():
    first = setup_logging()
    second = setup_logging()
    assert first.normalized_level == second.normalized_level

    handlers = logging.getLogger().handlers
    assert handlers, "Root logger has no handlers"
    assert any(isinstance(handler.formatter, JSONFormatter) for handler in handlers)

    logger = get_logger(__name__)
    logger.info("idempotent_call")
