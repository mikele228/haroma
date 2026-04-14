"""Shared environment-variable readers for HaromaX6 (no framework deps)."""

from __future__ import annotations

import os


def env_truthy(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)) or str(default))
    except (TypeError, ValueError):
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or str(default))
    except (TypeError, ValueError):
        return default
