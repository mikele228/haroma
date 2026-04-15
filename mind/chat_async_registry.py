"""In-memory registry for non-blocking HTTP chat (submit → poll).

``POST /chat`` with ``"async": true`` returns 202 + ``request_id``;
client polls ``GET /chat/result?id=...``. Mitigates synchronous HTTP blocking
without changing the cognitive pipeline.

Not durable across process restarts; single-node only. Thread-safe.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from typing import Any, Dict, Optional

from mind.structured_log import log_event, structured_log_enabled

_MAX_ENTRIES = 2048
_DEFAULT_TTL_SEC = 900.0

_LOG_HTTP_CHAT_END = ("1", "true", "yes", "on")


def _log_http_chat_end_failure(exc: BaseException) -> None:
    """Log when ``http_chat_end`` fails (silent by default; enable for debugging)."""
    env_on = str(os.environ.get("HAROMA_LOG_HTTP_CHAT_END", "") or "").strip().lower() in _LOG_HTTP_CHAT_END
    if structured_log_enabled():
        log_event(
            "http_chat_end_error",
            error_type=type(exc).__name__,
            error=str(exc)[:1200],
        )
    elif env_on:
        print(
            f"[ChatAsyncRegistry] http_chat_end failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )


class ChatAsyncRegistry:
    """Maps ``request_id`` → slot dict from ``InputAgent.push_text``."""

    def __init__(self, shared: Any, *, ttl_sec: float = _DEFAULT_TTL_SEC) -> None:
        self._shared = shared
        self._ttl_sec = max(1.0, float(ttl_sec))
        self._lock = threading.Lock()
        self._pending: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        slot: Dict[str, Any],
        *,
        experiment_id: Optional[str] = None,
        lab_run_id: Optional[str] = None,
    ) -> str:
        """Store slot; return new request id.

        Optional *experiment_id* / *lab_run_id* are echoed on async poll/SSE when the
        original Flask request context is gone.
        """
        rid = str(uuid.uuid4())
        with self._lock:
            self._evict_stale_unlocked()
            while len(self._pending) >= _MAX_ENTRIES and self._pending:
                oldest = min(self._pending.items(), key=lambda kv: kv[1]["t0"])[0]
                self._pop_unlocked(oldest, release_http=True)
            _eid = str(experiment_id).strip()[:200] if experiment_id is not None else ""
            _lr = str(lab_run_id).strip()[:120] if lab_run_id is not None else ""
            self._pending[rid] = {
                "slot": slot,
                "t0": time.time(),
                "experiment_id": _eid or None,
                "lab_run_id": _lr or None,
            }
        return rid

    def get(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._evict_stale_unlocked()
            return self._pending.get(request_id)

    def pop(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._pending.pop(request_id, None)

    def pending_count(self) -> int:
        with self._lock:
            self._evict_stale_unlocked()
            return len(self._pending)

    def cancel(self, request_id: str) -> str:
        """Drop a pending request. Returns ``gone`` | ``not_pending`` | ``ok``."""
        with self._lock:
            self._evict_stale_unlocked()
            rid = (request_id or "").strip()
            if not rid or rid not in self._pending:
                return "gone"
            ent = self._pending[rid]
            slot = ent.get("slot") or {}
            ev = slot.get("event")
            if ev is not None and ev.is_set():
                return "not_pending"
            self._pop_unlocked(rid, release_http=True)
            return "ok"

    def _pop_unlocked(self, request_id: str, *, release_http: bool) -> None:
        ent = self._pending.pop(request_id, None)
        if ent is None:
            return
        if not release_http:
            return
        slot = ent.get("slot") or {}
        ev = slot.get("event")
        if ev is not None and not ev.is_set():
            try:
                self._shared.http_chat_end()
            except Exception as e:
                _log_http_chat_end_failure(e)

    def _evict_stale_unlocked(self) -> None:
        now = time.time()
        stale = [rid for rid, ent in self._pending.items() if now - ent["t0"] > self._ttl_sec]
        for rid in stale:
            self._pop_unlocked(rid, release_http=True)


def truthy_async_flag(raw: Any) -> bool:
    if raw is True:
        return True
    if raw is False:
        return False
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return False
