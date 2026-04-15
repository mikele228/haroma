"""Assemble GET /status JSON body from booted :class:`agents.boot_agent.BootAgent` state.

Env affecting this payload (see also module docstring in ``mind.elarion_server_v2``):
``HAROMA_BG_DEFER_TRAINING_ON_HTTP_CHAT``, ``HAROMA_BG_DEFER_TRAINING_CAP_SEC``,
``HAROMA_CHAT_TIMEOUT``, ``HAROMA_FAST_LLM_DEFAULT_TIMEOUT_SEC``,
``HAROMA_LLM_CONTEXT_TIMEOUT_SEC`` (via :func:`mind.cognitive_contracts.llm_context_timeout_seconds`).
Set ``HAROMA_STATUS_SNAPSHOT_DEBUG=1`` for ``logging`` debug lines with tracebacks on partial failures.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

from mind.bg_training_env import (
    bg_training_defer_cap_sec,
    defer_training_on_http_chat,
)
from mind.http_chat_timeouts import http_chat_wait_sec
from mind.cognitive_contracts import llm_context_timeout_seconds

if TYPE_CHECKING:
    from agents.boot_agent import BootAgent

_log = logging.getLogger(__name__)


def _snapshot_debug() -> bool:
    return str(os.environ.get("HAROMA_STATUS_SNAPSHOT_DEBUG", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class _SensorPoller(Protocol):
    def stats(self) -> Dict[str, Any]: ...


class _ChatAsyncRegistry(Protocol):
    def pending_count(self) -> int: ...

    _ttl_sec: Optional[float]


def _resolve_http_inflight(shared: Any, runtime_signals: Optional[Dict[str, Any]], notes: List[str]) -> int:
    """Prefer ``runtime_signals['http_chat_inflight']`` when present; compare with ``shared``."""
    v_rs: Optional[int] = None
    if runtime_signals is not None:
        raw = runtime_signals.get("http_chat_inflight")
        if raw is not None:
            try:
                v_rs = int(raw)
            except (TypeError, ValueError):
                notes.append("runtime_signals.http_chat_inflight not coercible to int")
    try:
        v_sh = int(shared.http_chat_inflight)
    except Exception:
        notes.append("shared.http_chat_inflight unreadable")
        v_sh = 0
    if v_rs is not None:
        if v_rs != v_sh:
            notes.append(f"http_chat_inflight mismatch shared={v_sh} runtime_signals={v_rs}")
        return v_rs
    return v_sh


def build_http_status_payload(
    boot_agent: "BootAgent",
    sensor_poller: Optional[_SensorPoller],
    chat_async_registry: Optional[_ChatAsyncRegistry],
) -> Dict[str, Any]:
    """Return the dict serialized as JSON for ``GET /status``."""
    notes: List[str] = []
    s = boot_agent.shared
    vlt = getattr(s, "virtual_llm_tree", None)
    _lb = getattr(s, "llm_backend", None)
    _llm_info: Dict[str, Any] = {}
    if _lb is not None:
        _llm_info = {
            "backend_type": getattr(_lb, "backend_type", "none"),
            "model_name": getattr(_lb, "model_name", "") or "",
            "available": bool(getattr(_lb, "available", False)),
            "n_gpu_layers": getattr(_lb, "_n_gpu_layers", None),
            "n_ctx": getattr(_lb, "_n_ctx", None),
        }

    _llm_tlim = llm_context_timeout_seconds()

    _async_pending = 0
    _async_ttl: Optional[float] = None
    if chat_async_registry is not None:
        try:
            _async_pending = chat_async_registry.pending_count()
            _async_ttl = getattr(chat_async_registry, "_ttl_sec", None)
        except Exception:
            notes.append("chat_async_registry stats failed")
            if _snapshot_debug():
                _log.debug("chat_async_registry", exc_info=True)

    _runtime_signals: Optional[Dict[str, Any]] = None
    try:
        if hasattr(s, "signals") and hasattr(s.signals, "snapshot"):
            _runtime_signals = s.signals.snapshot()
    except Exception:
        notes.append("runtime_signals.snapshot failed")
        if _snapshot_debug():
            _log.debug("runtime_signals.snapshot", exc_info=True)

    _defer_bg = defer_training_on_http_chat()
    _http_inflight = _resolve_http_inflight(s, _runtime_signals, notes)
    _bg_training_deferred = bool(_defer_bg and _http_inflight > 0)
    _bg_defer_cap = bg_training_defer_cap_sec()

    _web_learn_status: Optional[dict] = None
    try:
        _bg = getattr(boot_agent, "background_agent", None)
        if _bg is not None:
            _wc = getattr(_bg, "_web_crawler", None)
            if _wc is not None and hasattr(_wc, "stats"):
                _web_learn_status = _wc.stats()
    except Exception:
        notes.append("web_learn stats failed")
        if _snapshot_debug():
            _log.debug("web_learn", exc_info=True)
        _web_learn_status = None

    _ts_sched = None
    try:
        ts = getattr(s, "training_scheduler", None)
        if ts is not None and hasattr(ts, "stats"):
            _ts_sched = ts.stats()
    except Exception:
        notes.append("training_scheduler.stats failed")
        if _snapshot_debug():
            _log.debug("training_scheduler", exc_info=True)

    _cognitive_obs: Optional[Dict[str, Any]] = None
    try:
        if hasattr(s, "cognitive_metrics") and hasattr(s.cognitive_metrics, "snapshot"):
            _cognitive_obs = s.cognitive_metrics.snapshot()
    except Exception:
        notes.append("cognitive_metrics.snapshot failed")
        if _snapshot_debug():
            _log.debug("cognitive_metrics.snapshot", exc_info=True)

    _input_buffer: Dict[str, Any] = {"text_pending": 0, "sensor_pending": 0}
    try:
        _inp = getattr(boot_agent, "input_agent", None)
        if _inp is not None and hasattr(_inp, "buffer_stats"):
            _input_buffer = _inp.buffer_stats()
    except Exception:
        notes.append("input_agent.buffer_stats failed")
        if _snapshot_debug():
            _log.debug("input_agent.buffer_stats", exc_info=True)

    _lab_rid = getattr(s, "lab_run_id", None)
    _ae_err = getattr(s, "agent_environment_error", None)
    _health: Dict[str, Any] = {
        "process": "up",
        "llm_ready": bool(_llm_info.get("available")) if _llm_info else False,
        "last_agent_environment_received_at": getattr(s, "agent_environment_received_at", None),
        "agent_environment_error": _ae_err if _ae_err else None,
    }
    _emb: Dict[str, Any] = {}
    try:
        from mind.robot_readiness import embodiment_readiness_summary

        _emb = embodiment_readiness_summary(s)
    except Exception as _emb_e:
        notes.append("embodiment_readiness_failed")
        _emb = {"ok": False, "error": str(_emb_e)[:160]}

    _del_timeouts = 0
    _chat_turns = 0
    if isinstance(_cognitive_obs, dict):
        try:
            _del_timeouts = int(_cognitive_obs.get("delegation_timeouts") or 0)
        except (TypeError, ValueError):
            _del_timeouts = 0
        try:
            _chat_turns = int(_cognitive_obs.get("chat_turns") or 0)
        except (TypeError, ValueError):
            _chat_turns = 0

    _text_buf = 0
    try:
        _text_buf = int(_input_buffer.get("text_pending") or 0) + int(
            _input_buffer.get("text_priority_pending") or 0
        )
    except (TypeError, ValueError):
        _text_buf = 0
    try:
        _sensor_buf = int(_input_buffer.get("sensor_pending") or 0)
    except (TypeError, ValueError):
        _sensor_buf = 0

    _life_loop: Dict[str, Any] = {
        "alive": _health.get("process") == "up",
        "cycle_count": int(getattr(s, "cycle_count", 0) or 0),
        "idle_cycles": 0,
        "external_cycles": _chat_turns,
        "errors": _del_timeouts,
        "buffer": {
            "text_pending": _text_buf,
            "sensor_pending": _sensor_buf,
        },
    }

    out: Dict[str, Any] = {
        "architecture": "multi-agent-v2",
        "lab_run_id": str(_lab_rid) if _lab_rid else None,
        "health": _health,
        "embodiment_readiness": _emb,
        "life_loop": _life_loop,
        "agents": boot_agent.stats(),
        "cycle_count": s.cycle_count,
        "memory_nodes": s.memory.count_nodes(),
        "llm": _llm_info,
        "chat_async_pending": _async_pending,
        "chat_async_ttl_sec": _async_ttl,
        "http_chat_inflight": _http_inflight,
        "bg_training_defer_enabled": _defer_bg,
        "bg_training_deferred": _bg_training_deferred,
        "bg_training_defer_cap_sec": _bg_defer_cap,
        "web_learn": _web_learn_status,
        "training_scheduler": _ts_sched,
        "runtime_signals": _runtime_signals,
        "chat_timeout_sec": http_chat_wait_sec(),
        "chat_timeout_sec_depth_normal": http_chat_wait_sec(),
        "llm_context_timeout_sec": _llm_tlim,
        "virtual_llm_tree": vlt.summary() if vlt is not None else None,
        "organs": s.organ_registry.summary(),
        "symbolic_queue": s.symbolic_queue.stats(),
        "fingerprint": s.fingerprint_engine.stats(),
        "reconciliation": s.reconciliation.stats(),
        "message_bus": boot_agent.bus.stats(),
        "sensors": sensor_poller.stats() if sensor_poller else {},
        "agent_environment": s.agent_environment_status(),
        "cognitive_observability": _cognitive_obs,
    }
    if notes:
        out["status_build_notes"] = notes
    return out
