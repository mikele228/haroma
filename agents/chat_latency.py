"""
Optional wall-time tracing for HTTP /chat: spans from InputAgent through Persona cycle.

Enable with POST ``{"trace_latency": true}`` or env ``HAROMA_CHAT_TRACE=1``.
Response includes ``latency_trace``: ``{ "total_ms", "spans": [{ "phase", "ms" }] }``.

Console lines (``[ChatLatency]``) when POST ``trace_latency`` is true or env
``HAROMA_CHAT_TRACE_LOG=1``.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from mind.cognitive_observability import append_cognitive_trace_to_payload


def trace_requested() -> bool:
    return os.environ.get("HAROMA_CHAT_TRACE", "").lower() in (
        "1",
        "true",
        "yes",
    )


def trace_log_requested() -> bool:
    return os.environ.get("HAROMA_CHAT_TRACE_LOG", "").lower() in (
        "1",
        "true",
        "yes",
    )


def trace_enabled(slot: Optional[Dict[str, Any]]) -> bool:
    return bool(slot and slot.get("_trace_enabled"))


def _trace_log(slot: Optional[Dict[str, Any]]) -> bool:
    return bool(slot and slot.get("_trace_enabled") and slot.get("_trace_log"))


def trace_init(
    slot: Optional[Dict[str, Any]],
    *,
    log_to_console: bool = False,
) -> None:
    if slot is None:
        return
    slot["_trace_enabled"] = True
    slot["_trace_log"] = bool(log_to_console or trace_log_requested())
    t0 = time.perf_counter()
    slot["_trace_t0"] = t0
    slot["_trace_last"] = t0
    slot["_trace_spans"] = []
    if _trace_log(slot):
        print("[ChatLatency] trace start (request)", flush=True)


def trace_span(slot: Optional[Dict[str, Any]], phase: str) -> None:
    if not trace_enabled(slot):
        return
    spans: List[Dict[str, Any]] = slot.setdefault("_trace_spans", [])
    now = time.perf_counter()
    last = float(slot.get("_trace_last", now))
    ms = round((now - last) * 1000, 2)
    spans.append({"phase": phase, "ms": ms})
    slot["_trace_last"] = now
    if _trace_log(slot):
        print(f"[ChatLatency] +{ms:9.2f} ms  {phase}", flush=True)


def trace_attach_to_payload(
    slot: Optional[Dict[str, Any]],
    payload: Dict[str, Any],
    *,
    final_phase: str = "response_finalize",
) -> None:
    if slot:
        append_cognitive_trace_to_payload(
            payload,
            trace_id=slot.get("cognitive_trace_id")
            if isinstance(slot.get("cognitive_trace_id"), str)
            else None,
            route=slot.get("_cognitive_route")
            if isinstance(slot.get("_cognitive_route"), str)
            else None,
        )
    if not trace_enabled(slot):
        return
    trace_span(slot, final_phase)
    t0 = float(
        slot.get("_trace_t0")
        or slot.get("_trace_last")
        or time.perf_counter()
    )
    total = round((time.perf_counter() - t0) * 1000, 2)
    payload["latency_trace"] = {
        "total_ms": total,
        "spans": list(slot.get("_trace_spans") or []),
    }
    if _trace_log(slot):
        sp = payload["latency_trace"]["spans"]
        slow = sorted(sp, key=lambda x: -x.get("ms", 0.0))[:8]
        top = ", ".join(f"{s['phase']}:{s['ms']}ms" for s in slow)
        print(
            f"[ChatLatency] TOTAL {total} ms  ({len(sp)} spans)  top: {top}",
            flush=True,
        )
