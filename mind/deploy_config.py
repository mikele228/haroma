"""
Deployment listen address / port and optional ``.env`` loading.

Env (after optional ``.env`` load from project root):

* ``HAROMA_BIND_HOST`` — Werkzeug bind address (default ``0.0.0.0``).
* ``HAROMA_HTTP_PORT`` — listen port (default ``8193``).

Run ``python scripts/setup_wizard.py`` to generate a ``.env`` interactively.
"""

from __future__ import annotations

import os

_DEFAULT_HOST = "0.0.0.0"
_DEFAULT_PORT = 8193


def project_root() -> str:
    """Repository root (parent of ``mind/``)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def dotenv_path() -> str:
    return os.path.join(project_root(), ".env")


def load_dotenv(override: bool = False) -> int:
    """Parse ``.env`` in the project root and set ``os.environ``.

    * ``#`` starts a comment; ``KEY=VALUE`` lines set variables.
    * Values may be single- or double-quoted.
    * If *override* is False, existing environment variables are not replaced.

    Returns the number of keys set.
    """
    path = dotenv_path()
    if not os.path.isfile(path):
        return 0
    n = 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                if not key:
                    continue
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                    val = val[1:-1]
                if not override and key in os.environ:
                    continue
                os.environ[key] = val
                n += 1
    except OSError:
        return n
    return n


def http_listen_host() -> str:
    h = str(os.environ.get("HAROMA_BIND_HOST", "") or "").strip()
    return h if h else _DEFAULT_HOST


def http_listen_port() -> int:
    raw = str(os.environ.get("HAROMA_HTTP_PORT", "") or "").strip()
    if not raw:
        return _DEFAULT_PORT
    try:
        p = int(raw)
        if 1 <= p <= 65535:
            return p
    except (TypeError, ValueError):
        pass
    return _DEFAULT_PORT


def display_base_url() -> str:
    """Human-facing base URL for console hints (always uses localhost for 0.0.0.0)."""
    port = http_listen_port()
    return f"http://127.0.0.1:{port}"


def env_summary_lines() -> list[str]:
    """Short lines for logging (no secrets)."""
    return [
        f"HAROMA_BIND_HOST={http_listen_host()}",
        f"HAROMA_HTTP_PORT={http_listen_port()}",
    ]
