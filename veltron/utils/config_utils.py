"""Minimal YAML/JSON config loading with dotted-key overrides and CLI binding."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .logging_utils import get_logger

log = get_logger(__name__)

try:  # PyYAML is optional; JSON configs always work.
    import yaml  # type: ignore[import-untyped]

    _HAS_YAML = True
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]
    _HAS_YAML = False


class Config(dict):
    """``dict`` with dotted access and deep merge.

    ``cfg["train.lr"]`` and ``cfg.get_path("train.lr")`` both work, and ``cfg.set``
    creates intermediate dicts so overrides can address nested keys.
    """

    def __getitem__(self, key: str) -> Any:
        if "." in key:
            return self.get_path(key)
        return super().__getitem__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if "." in key:
            self.set(key, value)
            return
        super().__setitem__(key, value)

    def get_path(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        cur: Any = self
        for part in parts[:-1]:
            if part not in cur or not isinstance(cur[part], dict):
                cur[part] = {}
            cur = cur[part]
        cur[parts[-1]] = value

    def merge(self, other: dict) -> Config:
        deep_merge(self, other)
        return self

    def apply_overrides(self, overrides: Iterable[str]) -> Config:
        """Apply ``key.path=value`` strings; values are parsed as JSON then as raw text."""
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"override must look like key=value, got {item!r}")
            k, v = item.split("=", 1)
            self.set(k.strip(), _parse_scalar(v.strip()))
        return self

    def filter_prefix(self, prefix: str) -> Config:
        """Return the sub-dict under ``prefix`` (convenience for nested sections)."""
        return Config(self.get_path(prefix, {}) or {})


def _parse_scalar(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        lowered = text.lower()
        if lowered in ("true", "yes", "on"):
            return True
        if lowered in ("false", "no", "off"):
            return False
        if lowered in ("none", "null"):
            return None
        return text


def deep_merge(base: dict, incoming: dict) -> dict:
    for k, v in incoming.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: Iterable[str] | None = None) -> Config:
    """Load a YAML or JSON config, following an optional ``base:`` include chain."""
    if path is None:
        return Config()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        if not _HAS_YAML:
            raise RuntimeError("PyYAML is required for .yaml configs but is not installed")
        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text)

    if isinstance(raw, dict) and "base" in raw and isinstance(raw["base"], (str, list)):
        bases = raw["base"] if isinstance(raw["base"], list) else [raw["base"]]
        merged: dict = {}
        for b in bases:
            b_path = (path.parent / b).resolve()
            deep_merge(merged, load_config(b_path))
        raw.pop("base")
        deep_merge(merged, raw)
        raw = merged

    cfg = Config(raw)
    if overrides:
        cfg.apply_overrides(overrides)
    log.debug("loaded config %s with %d top-level keys", path, len(cfg))
    return cfg


__all__ = ["Config", "load_config", "deep_merge"]
