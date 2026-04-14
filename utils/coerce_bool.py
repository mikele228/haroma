"""Coerce request / message payload values to real booleans.

Python's ``bool("false")`` is True — use this for JSON-derived flags.
"""

import os


def json_bool(val, default: bool = False) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    s = str(val).strip().lower()
    if s in ("0", "false", "no", "off", "null", "none", ""):
        return False
    if s in ("1", "true", "yes", "on"):
        return True
    return default


def env_flag(name: str, default: bool = False) -> bool:
    """True/false from ``os.environ[name]`` using the same rules as ``json_bool``.

    Missing or empty env var returns *default*.
    """
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return json_bool(raw, default=default)
