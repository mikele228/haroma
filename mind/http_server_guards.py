"""
Optional HTTP hardening for the Flask server (bearer token on selected routes).

Set ``HAROMA_HTTP_BEARER_TOKEN`` to a non-empty secret to require authentication
on paths listed in ``HAROMA_HTTP_PROTECT_PATHS`` (comma-separated; see defaults
below). When unset, behavior is unchanged (no auth).

Clients send either:

* ``Authorization: Bearer <token>``, or
* ``X-Haroma-Token: <token>``

Used by :mod:`mind.elarion_server_v2` ``before_request``. Does not replace TLS or
a reverse proxy for Internet exposure.
"""

from __future__ import annotations

import os
import secrets
from typing import Any, Dict, FrozenSet, Optional, Tuple

# POST routes that mutate state or inject trusted environment data (conservative default).
_DEFAULT_PROTECT_PATHS: FrozenSet[str] = frozenset(
    {
        "/agent/environment",
        "/robot/bridge/feedback",
        "/teach",
        "/save",
    }
)


def configured_bearer_secret() -> str:
    return str(os.environ.get("HAROMA_HTTP_BEARER_TOKEN", "") or "").strip()


def protected_path_set() -> FrozenSet[str]:
    raw = str(os.environ.get("HAROMA_HTTP_PROTECT_PATHS", "") or "").strip()
    if not raw:
        return _DEFAULT_PROTECT_PATHS
    parts = {p.strip() for p in raw.split(",") if p.strip()}
    return frozenset(parts) if parts else _DEFAULT_PROTECT_PATHS


def verify_http_request_bearer(request: Any) -> Optional[Tuple[Dict[str, Any], int]]:
    """Return ``(error_json, http_status)`` if the request must be rejected, else ``None``."""
    secret = configured_bearer_secret()
    if not secret:
        return None

    path = getattr(request, "path", "") or ""
    if path not in protected_path_set():
        return None

    hdrs = getattr(request, "headers", None) or {}
    get = hdrs.get if hasattr(hdrs, "get") else (lambda k, d="": {})

    auth = str(get("Authorization", "") or "")
    token = ""
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    if not token:
        token = str(get("X-Haroma-Token", "") or "").strip()

    if not token:
        return ({"error": "unauthorized", "detail": "missing_bearer_token"}, 401)

    # secrets.compare_digest returns False for unequal lengths without raising.
    if not secrets.compare_digest(token, secret):
        return ({"error": "unauthorized", "detail": "invalid_token"}, 403)

    return None
