"""
Simple in-memory sliding-window rate limit for selected HTTP POST routes.

Env:

* ``HAROMA_HTTP_RATE_LIMIT_PER_MIN`` — max requests per client IP per route per minute.
  ``0`` (default) disables rate limiting.

Uses a fixed 60-second window (approximate; good enough for abuse mitigation).
Thread-safe for Flask ``threaded=True``.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, Optional, Tuple

_RATE_LOCK = threading.Lock()
# (ip, path) -> deque of request epoch times
_BUCKETS: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque())

# Sliding window length (seconds); also used for ``Retry-After`` on HTTP 429.
RATE_LIMIT_WINDOW_SEC = 60.0

_LIMITED_PATHS = frozenset(
    {
        "/chat",
        "/sensor",
        "/agent/environment",
        "/robot/bridge/feedback",
        "/teach",
        "/save",
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


def check_rate_limit(request: Any) -> Optional[Tuple[Dict[str, Any], int]]:
    """Return ``(error_body, 429)`` if over limit, else ``None``."""
    cap = rate_limit_per_minute()
    if cap <= 0:
        return None
    path = getattr(request, "path", "") or ""
    if path not in _LIMITED_PATHS or getattr(request, "method", "") != "POST":
        return None
    ip = getattr(request, "remote_addr", None) or "unknown"
    key = (str(ip)[:64], path)
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
