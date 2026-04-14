"""
Optional one-line JSON logs to stderr for operators / log aggregators.

Env ``HAROMA_STRUCTURED_LOG=1`` (or ``true`` / ``yes`` / ``on``): emit JSON objects
with an ``event`` field and arbitrary string-safe fields. The HTTP layer uses
``event=http_access`` with ``request_id``, ``status``, ``duration_ms``, etc. (see
``mind.elarion_server_v2`` ``after_request``).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict


def structured_log_enabled() -> bool:
    return str(os.environ.get("HAROMA_STRUCTURED_LOG", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def log_event(event: str, **fields: Any) -> None:
    """Append one JSON line to stderr when structured logging is enabled."""
    if not structured_log_enabled():
        return
    row: Dict[str, Any] = {"event": str(event)[:120], "t": time.time()}
    for k, v in fields.items():
        try:
            if isinstance(v, (dict, list)):
                row[str(k)[:64]] = v
            else:
                row[str(k)[:64]] = str(v)[:2000]
        except Exception:
            row[str(k)[:64]] = "<unserializable>"
    try:
        sys.stderr.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        sys.stderr.flush()
    except Exception:
        pass
