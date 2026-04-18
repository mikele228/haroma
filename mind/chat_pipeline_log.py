"""Correlated console logs for debugging where an input turn stalls.

HTTP ``/chat`` is one path; other inputs (sensor, teach, etc.) share the same
pipeline stages on the input → TrueSelf → persona path.

Set ``HAROMA_INPUT_PIPELINE_LOG=1`` (or ``true`` / ``yes`` / ``on``). The legacy
name ``HAROMA_CHAT_PIPELINE_LOG`` is still read if the new variable is unset.
Each line uses prefix ``[InputPipeline]`` and ``trace=<id>`` when
:func:`agents.input_agent.InputAgent.push_text` has assigned ``cognitive_trace_id``
on the wait slot.

**Timed pipeline (per-stage deltas):** set ``HAROMA_INPUT_PIPELINE_TIMING=1``, or the
legacy ``HAROMA_CHAT_PIPELINE_TIMING=1``, or set ``HAROMA_INPUT_PIPELINE_LOG=full``
(legacy: ``HAROMA_CHAT_PIPELINE_LOG=full``) to enable both stage logs and timing in
one flag. Each line then includes ``seg=…ms cum=…ms`` — time since the *previous*
pipeline log for this trace, and cumulative time since the first log for this trace.

Stages are ordered roughly: ``http.*`` → ``input.*`` → ``trueself.*`` → ``persona.*``.
The last stage printed before a hang indicates where the pipeline blocked.

**Reading the log**

- Lines **without** ``trace=`` are not your HTTP ``/chat`` turn (e.g. inner dialogue, idle
  reflection, vision-only ticks).
- After ``input.after_send_to_trueself``, your message waits on the bus until TrueSelf's
  next ``poll``. If TrueSelf is still inside a **long** cognitive cycle (packed LLM,
  inner dialogue), the next tick may not run for a long time — ``trueself.*`` lines for
  your trace appear only when that turn is **dequeued**. See ``trueself.tick_input_batch``,
  ``trueself.input_reordered_http_first``, ``trueself.input_defer_sensor_backlog``, and
  ``trueself.waiting_packed_llm_io``.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

__all__ = [
    "input_pipeline_log_enabled",
    "log_input_pipeline",
    "pipeline_timing_enabled",
    "pipeline_trace_end",
    "trace_id_from_message",
    "trace_id_from_slot",
    # Backward compatibility (chat-only naming)
    "chat_pipeline_log_enabled",
    "log_chat_pipeline",
]

_lock = threading.Lock()
# trace_id -> (t0_perf, last_perf)
_trace_times: Dict[str, Tuple[float, float]] = {}
_MAX_TRACES = 512


def _trim_traces_if_needed() -> None:
    if len(_trace_times) <= _MAX_TRACES:
        return
    keys = list(_trace_times.keys())
    for k in keys[: max(1, len(keys) // 2)]:
        _trace_times.pop(k, None)


def _env_primary_then_legacy(primary: str, legacy: str) -> str:
    v = str(os.environ.get(primary, "") or "").strip()
    if v:
        return v
    return str(os.environ.get(legacy, "") or "").strip()


def input_pipeline_log_enabled() -> bool:
    v = _env_primary_then_legacy("HAROMA_INPUT_PIPELINE_LOG", "HAROMA_CHAT_PIPELINE_LOG").lower()
    return v in ("1", "true", "yes", "on", "full")


def pipeline_timing_enabled() -> bool:
    v = _env_primary_then_legacy("HAROMA_INPUT_PIPELINE_LOG", "HAROMA_CHAT_PIPELINE_LOG").lower()
    if v == "full":
        return True
    t = _env_primary_then_legacy("HAROMA_INPUT_PIPELINE_TIMING", "HAROMA_CHAT_PIPELINE_TIMING").lower()
    return t in ("1", "true", "yes", "on")


def trace_id_from_slot(slot: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(slot, dict):
        return None
    tid = slot.get("cognitive_trace_id")
    return tid if isinstance(tid, str) and tid.strip() else None


def trace_id_from_message(msg: Any) -> Optional[str]:
    """Resolve trace id from Message metadata or embedded HTTP response slot."""
    if msg is None:
        return None
    md = getattr(msg, "metadata", None) or {}
    tid = md.get("cognitive_trace_id")
    if isinstance(tid, str) and tid.strip():
        return tid
    slot = md.get("_response_slot")
    if isinstance(slot, dict):
        x = trace_id_from_slot(slot)
        if x:
            return x
    rs = getattr(msg, "response_slot", None)
    if isinstance(rs, dict):
        return trace_id_from_slot(rs)
    return None


def pipeline_trace_end(trace_id: Optional[str]) -> None:
    """Drop timing state for *trace_id* (call when HTTP response is fully done)."""
    if not trace_id:
        return
    with _lock:
        _trace_times.pop(trace_id, None)


def log_input_pipeline(stage: str, *, trace_id: Optional[str] = None, detail: str = "") -> None:
    if not input_pipeline_log_enabled():
        return
    timing_suffix = ""
    if pipeline_timing_enabled() and trace_id:
        now = time.perf_counter()
        with _lock:
            _trim_traces_if_needed()
            if trace_id not in _trace_times:
                _trace_times[trace_id] = (now, now)
                seg_ms = 0.0
                cum_ms = 0.0
            else:
                t0, last = _trace_times[trace_id]
                seg_ms = (now - last) * 1000.0
                cum_ms = (now - t0) * 1000.0
                _trace_times[trace_id] = (t0, now)
            timing_suffix = f" seg={seg_ms:.2f}ms cum={cum_ms:.2f}ms"
    t = f" trace={trace_id}" if trace_id else ""
    d = f" | {detail}" if detail else ""
    print(f"[InputPipeline]{t} {stage}{timing_suffix}{d}", flush=True)


# Legacy names (same behavior)
chat_pipeline_log_enabled = input_pipeline_log_enabled
log_chat_pipeline = log_input_pipeline
