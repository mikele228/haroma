"""In-process observability for the cognitive loop: trace IDs and metrics.

* **Trace ID**: one per HTTP / chat turn, carried on the response slot and
  ``Message.metadata["cognitive_trace_id"]`` through Input → TrueSelf → Persona.
* **Metrics**: thread-safe counters and bounded samples for ``GET /status``.

Env ``HAROMA_COGNITIVE_TRACE_LOG=1``: console lines ``[CognitiveTrace]`` at
routing decisions (in addition to optional :mod:`agents.chat_latency`).
"""

from __future__ import annotations

import os
import threading
import uuid
from collections import Counter, deque
from typing import Any, Deque, Dict, Optional


def new_trace_id() -> str:
    """Opaque id for one user turn (HTTP /chat or equivalent)."""
    return uuid.uuid4().hex[:20]


def cognitive_trace_log_enabled() -> bool:
    return str(os.environ.get("HAROMA_COGNITIVE_TRACE_LOG", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def log_cognitive_trace(trace_id: Optional[str], message: str) -> None:
    if not cognitive_trace_log_enabled() or not trace_id:
        return
    print(f"[CognitiveTrace] id={trace_id} {message}", flush=True)


class CognitiveMetrics:
    """Counters and rolling samples; safe to call from agent threads."""

    def __init__(self, *, sample_cap: int = 200) -> None:
        self._lock = threading.Lock()
        self._sample_cap = sample_cap
        self.chat_turns: int = 0
        self.route_counts: Counter[str] = Counter()
        self.delegation_timeouts: int = 0
        self.reconcile_runs: int = 0
        self.persistence_save_count: int = 0
        self.persistence_save_total_sec: float = 0.0
        self.persistence_load_count: int = 0
        self.persistence_load_total_sec: float = 0.0
        self._last_input_queue_total: int = 0
        self._llm_wait_ms: Deque[float] = deque(maxlen=sample_cap)
        self._persona_cycle_ms: Deque[float] = deque(maxlen=sample_cap)
        # Shared neural RW / persona gate: increments when hold time exceeds budget
        self.shared_lock_over_budget: Counter[str] = Counter()

    def record_shared_lock_over_budget(self, lock_name: str) -> None:
        """Increment when a shared lock hold (after acquire) exceeded ``HAROMA_SHARED_LOCK_BUDGET_SEC``."""
        with self._lock:
            self.shared_lock_over_budget[lock_name] += 1

    def on_chat_turn_started(self) -> None:
        with self._lock:
            self.chat_turns += 1

    def observe_input_queue_depth(self, priority_len: int, normal_len: int) -> None:
        with self._lock:
            self._last_input_queue_total = int(priority_len) + int(normal_len)

    def record_route(self, route_key: str) -> None:
        """e.g. ``fast_trueself``, ``normal_trueself``, ``delegate``."""
        with self._lock:
            self.route_counts[route_key] += 1

    def record_delegation_timeout(self) -> None:
        with self._lock:
            self.delegation_timeouts += 1

    def record_reconcile_run(self) -> None:
        with self._lock:
            self.reconcile_runs += 1

    def record_persistence_save_sec(self, duration_sec: float) -> None:
        with self._lock:
            self.persistence_save_count += 1
            self.persistence_save_total_sec += max(0.0, float(duration_sec))

    def record_persistence_load_sec(self, duration_sec: float) -> None:
        with self._lock:
            self.persistence_load_count += 1
            self.persistence_load_total_sec += max(0.0, float(duration_sec))

    def observe_llm_wait_ms(self, ms: Optional[float]) -> None:
        if ms is None:
            return
        try:
            v = float(ms)
        except (TypeError, ValueError):
            return
        if v < 0 or v > 1e7:
            return
        with self._lock:
            self._llm_wait_ms.append(v)

    def observe_persona_cycle_ms(self, ms: float) -> None:
        try:
            v = float(ms)
        except (TypeError, ValueError):
            return
        if v < 0 or v > 1e7:
            return
        with self._lock:
            self._persona_cycle_ms.append(v)

    @staticmethod
    def _summarize_samples(samples: Deque[float]) -> Dict[str, Any]:
        if not samples:
            return {"count": 0, "last_ms": None, "p50_ms": None, "p95_ms": None}
        arr = sorted(samples)
        n = len(arr)

        def _pct(p: float) -> float:
            if n == 1:
                return arr[0]
            i = min(n - 1, max(0, int(p * (n - 1))))
            return arr[i]

        return {
            "count": n,
            "last_ms": round(arr[-1], 2),
            "p50_ms": round(_pct(0.50), 2),
            "p95_ms": round(_pct(0.95), 2),
        }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            routes = dict(self.route_counts)
            q = self._last_input_queue_total
            llm = self._summarize_samples(self._llm_wait_ms)
            cyc = self._summarize_samples(self._persona_cycle_ms)
            saves = self.persistence_save_count
            save_tot = self.persistence_save_total_sec
            loads = self.persistence_load_count
            load_tot = self.persistence_load_total_sec
            return {
                "chat_turns": self.chat_turns,
                "routes": routes,
                "delegation_timeouts": self.delegation_timeouts,
                "reconcile_runs": self.reconcile_runs,
                "shared_lock_over_budget": dict(self.shared_lock_over_budget),
                "input_queue_depth_last": q,
                "llm_wait_ms": llm,
                "persona_cycle_ms": cyc,
                "persistence_save": {
                    "count": saves,
                    "total_sec": round(save_tot, 4),
                    "avg_sec": round(save_tot / saves, 4) if saves else None,
                },
                "persistence_load": {
                    "count": loads,
                    "total_sec": round(load_tot, 4),
                    "avg_sec": round(load_tot / loads, 4) if loads else None,
                },
            }


def append_cognitive_trace_to_payload(
    payload: Dict[str, Any],
    *,
    trace_id: Optional[str],
    route: Optional[str],
) -> None:
    if trace_id:
        payload["cognitive_trace_id"] = trace_id
    if route:
        payload["cognitive_route"] = route
