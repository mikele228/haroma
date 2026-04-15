"""
Simple in-memory sliding-window rate limit for selected HTTP routes.

Env:

* ``HAROMA_HTTP_RATE_LIMIT_PER_MIN`` — max **POST** requests per client IP per route
  per minute (selected paths). ``0`` (default) disables.
* ``HAROMA_HTTP_GET_RATE_LIMIT_PER_MIN`` — max **GET** requests per IP per route per
  minute for high-frequency poll routes (e.g. ``/chat/result``). ``0`` (default) disables.

Uses a fixed 60-second window (approximate; good enough for abuse mitigation).
Thread-safe for Flask ``threaded=True``.

Client identity uses :func:`mind.client_ip.get_effective_client_ip` so limits work
behind a reverse proxy when ``HAROMA_HTTP_USE_X_FORWARDED_FOR`` and trusted peers
are configured.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, Optional, Tuple

from mind.client_ip import get_effective_client_ip

_RATE_LOCK = threading.Lock()
# (ip, path) -> deque of request epoch times
_BUCKETS: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque())

# Sliding window length (seconds); also used for ``Retry-After`` on HTTP 429.
RATE_LIMIT_WINDOW_SEC = 60.0

_LIMITED_POST_PATHS = frozenset(
    {
        "/chat",
        "/sensor",
        "/agent/environment",
        "/robot/bridge/feedback",
        "/teach",
        "/save",
    }
)

# Polling / high-frequency GET (optional cap separate from POST)
_GET_LIMITED_PATHS = frozenset(
    {
        "/chat/result",
    }
)


def rate_limit_per_minute() -> int:
    raw = str(os.environ.get("HAROMA_HTTP_RATE_LIMIT_PER_MIN", "") or "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def get_rate_limit_per_minute() -> int:
    """Optional GET rate limit for poll routes (``HAROMA_HTTP_GET_RATE_LIMIT_PER_MIN``)."""
    raw = str(os.environ.get("HAROMA_HTTP_GET_RATE_LIMIT_PER_MIN", "") or "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def check_rate_limit(request: Any) -> Optional[Tuple[Dict[str, Any], int]]:
    """Return ``(error_body, 429)`` if over limit, else ``None``."""
    path = getattr(request, "path", "") or ""
    method = (getattr(request, "method", "") or "").upper()
    cap = 0
    if method == "POST" and path in _LIMITED_POST_PATHS:
        cap = rate_limit_per_minute()
    elif method == "GET" and path in _GET_LIMITED_PATHS:
        cap = get_rate_limit_per_minute()
    if cap <= 0:
        return None
    ip = get_effective_client_ip(request)
    key = (str(ip)[:64], path, method)
    now = time.time()
    window = RATE_LIMIT_WINDOW_SEC
    with _RATE_LOCK:
        dq = _BUCKETS[key]
        while dq and now - dq[0] > window:
            dq.popleft()
        if len(dq) >= cap:
            return (
                {
                    "error": "rate_limited",
                    "detail": f"max_{cap}_per_minute_per_ip_for_path",
                    "path": path,
                },
                429,
            )
        dq.append(now)
    return None


def clear_rate_limit_state_for_tests() -> None:
    """Clear sliding-window state (call from tests only; avoids cross-test leakage)."""
    with _RATE_LOCK:
        _BUCKETS.clear()
