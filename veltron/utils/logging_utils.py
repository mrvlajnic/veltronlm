"""Structured logging with request/run correlation IDs and redaction of secrets."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_SENSITIVE_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|apikey|secret|password|passwd|token|authorization|bearer)\s*[=:]\s*\S+"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{12,}"),
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
]

_CONFIGURING = False


class RedactingFilter(logging.Filter):
    """Strips API keys, bearer tokens and email addresses from every log record.

    Applied at the handler level rather than at each call site so a new ``logger.info``
    cannot leak a secret by accident.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        cleaned = msg
        for pat in _SENSITIVE_PATTERNS:
            cleaned = pat.sub("[REDACTED]", cleaned)
        if cleaned != msg:
            record.msg = cleaned
            record.args = ()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for ingestion by log shippers."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("run_id", "request_id", "model", "step", "elapsed_s"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class HumanFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)-28s %(message)s", "%H:%M:%S")


def setup_logging(level: str | None = None, json_logs: bool | None = None) -> None:
    """Idempotently configure root logging.

    ``json_logs`` defaults to the ``VELTRON_LOG_JSON`` environment variable.
    """
    global _CONFIGURING
    if _CONFIGURING:
        return
    lvl = (level or os.environ.get("VELTRON_LOG_LEVEL", "INFO")).upper()
    as_json = json_logs if json_logs is not None else os.environ.get("VELTRON_LOG_JSON", "0") == "1"

    root = logging.getLogger()
    root.setLevel(lvl)
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if as_json else HumanFormatter())
    handler.addFilter(RedactingFilter())
    root.addHandler(handler)

    # These are chatty and occasionally emit model prompts at INFO.
    for noisy in ("urllib3", "httpx", "filelock", "matplotlib", "pynvml"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURING = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


@contextmanager
def correlate(**fields: Any) -> Iterator[logging.LoggerAdapter]:
    """Attach arbitrary fields to every record emitted inside the block."""
    logger = get_logger("veltron")
    adapter = logging.LoggerAdapter(logger, fields)
    try:
        yield adapter
    finally:
        pass


__all__ = ["setup_logging", "get_logger", "correlate", "JsonFormatter", "RedactingFilter"]
