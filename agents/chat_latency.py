"""
Optional wall-time tracing for HTTP /chat: spans from InputAgent through Persona cycle.

Enable with POST ``{"trace_latency": true}`` or env ``HAROMA_CHAT_TRACE=1``.
Response includes ``latency_trace``: ``{ "total_ms", "spans": [{ "phase", "ms" }] }``.

**Dummy / probe mode:** with ``HAROMA_LLM_DUMMY_REPLY=1``, latency tracing is
**turned on automatically** for each chat turn (same ``latency_trace`` payload)
so you can measure end-to-end time without a second flag. Opt out with
``HAROMA_LLM_DUMMY_NO_LATENCY_TRACE=1`` if you need a smaller JSON body.

Dummy-reply env semantics are centralized in :mod:`mind.packed_llm_dummy_env` and
re-exported here for backward compatibility.

With ``HAROMA_LLM_DUMMY_REPLY``, packed-context ``generate_chat`` is skipped inside
``LLMContextReasoner`` (synthetic reply); ``latency_trace`` may include
``llm_context_latency_ms``, ``non_llm_wall_ms_approx``, and ``synthetic_llm`` when
those fields apply.

Console lines (``[ChatLatency]``) when POST ``trace_latency`` is true or env
``HAROMA_CHAT_TRACE_LOG=1``.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from mind.cognitive_observability import append_cognitive_trace_to_payload
from mind.packed_llm_dummy_env import (
    packed_llm_dummy_probe_active,
    packed_llm_dummy_reply_raw,
    synthetic_llm_dummy_reply_env,
)

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


def dummy_reply_auto_latency_trace() -> bool:
    """When True, InputAgent enables ``latency_trace`` without ``HAROMA_CHAT_TRACE``.

    Used when ``HAROMA_LLM_DUMMY_REPLY`` is on so dummy mode is enough to read
    ``latency_trace.total_ms`` in the HTTP JSON. Opt out with
    ``HAROMA_LLM_DUMMY_NO_LATENCY_TRACE=1``.
    """
    if not synthetic_llm_dummy_reply_env():
        return False
    if str(os.environ.get("HAROMA_LLM_DUMMY_NO_LATENCY_TRACE", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    return True


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
    lc = payload.get("llm_context")
    if isinstance(lc, dict) and lc.get("latency_ms") is not None:
        try:
            llm_ms = float(lc.get("latency_ms") or 0.0)
            payload["latency_trace"]["llm_context_latency_ms"] = round(llm_ms, 2)
            payload["latency_trace"]["non_llm_wall_ms_approx"] = round(
                max(0.0, total - llm_ms),
                2,
            )
        except (TypeError, ValueError):
            pass
    if synthetic_llm_dummy_reply_env():
        payload["latency_trace"]["synthetic_llm"] = True
    if _trace_log(slot):
        sp = payload["latency_trace"]["spans"]
        slow = sorted(sp, key=lambda x: -x.get("ms", 0.0))[:8]
        top = ", ".join(f"{s['phase']}:{s['ms']}ms" for s in slow)
        print(
            f"[ChatLatency] TOTAL {total} ms  ({len(sp)} spans)  top: {top}",
            flush=True,
        )
