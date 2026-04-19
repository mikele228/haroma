"""Prioritize the input pipeline over inner-dialogue and background churn.

HTTP ``/chat`` is one entry point; sensors and other text also flow through
:class:`agents.input_agent.InputAgent`. When :func:`input_pipeline_busy` is true
and ``HAROMA_CHAT_INPUT_PRIORITY`` is true (default), non-user-critical work is
deferred:

- **PersonaAgent:** only ``trueself_delegate`` is handled; other mailbox messages
  are re-queued for later ticks.
- **TrueSelfAgent:** ``input`` and ``persona_response`` run; ``inter_persona`` /
  ``direct`` / reconcile-dream messages are re-queued.
- **BackgroundAgent:** heavy tick sections (dream, goals, training schedule,
  inner-dialogue initiation, etc.) are skipped; dead-letter handling and
  optional persistence still run. **cmem** promotion is also skipped by default
  while active (it indexes memory and contends with recall); set
  ``HAROMA_CMEM_BUILD_DURING_ACTIVE_INPUT=1`` to allow cmem during active input.

When ``HAROMA_BG_PAUSE_ON_ACTIVE_INPUT`` is true (default),
:class:`~agents.background_agent.BackgroundAgent` does **not** start a new
heavy background cycle (training, dreams, reconcile, web learn, …) while
:func:`background_input_active` is true — i.e. HTTP chat in flight, text queues
backlogged, **agent_environment** alerts/threat metrics, or speech-like audio
in the sensor queue. Ambient sensor backlog alone (blank stare, lidar hum) does
**not** pause background unless you set ``HAROMA_BG_ACTIVE_INPUT_INCLUDES_SENSOR_PENDING=1``
(legacy coarse gate).

Set ``HAROMA_CHAT_INPUT_PRIORITY=0`` to restore previous interleaving.
"""

from __future__ import annotations

import os
from typing import Any

from utils.coerce_bool import env_flag

__all__ = [
    "background_input_active",
    "chat_input_priority_defer_non_user",
    "input_pipeline_busy",
    "input_pipeline_yield_busy",
]


def input_pipeline_yield_busy(shared: Any, boot_agent: Any = None) -> bool:
    """True when a persona should briefly yield after a cycle (HTTP chat or text queues).

    Unlike :func:`input_pipeline_busy`, this **does not** treat ``sensor_pending`` as
    busy. A steady sensor stream would otherwise make every persona pay the
    post-cycle sleep (``HAROMA_POST_CYCLE_*_SLEEP_SEC``) even though no HTTP/text
    work is waiting — a major source of multi-second chat latency.
    """
    try:
        if int(getattr(shared, "http_chat_inflight", 0) or 0) > 0:
            return True
    except Exception:
        pass
    ia = getattr(shared, "_input_agent_ref", None)
    if ia is None and boot_agent is not None:
        ia = getattr(boot_agent, "input_agent", None)
    if ia is None:
        return False
    if not hasattr(ia, "buffer_stats"):
        return False
    try:
        st = ia.buffer_stats()
        return (
            int(st.get("text_pending", 0) or 0) > 0
            or int(st.get("text_priority_pending", 0) or 0) > 0
        )
    except Exception:
        return True


def input_pipeline_busy(shared: Any, boot_agent: Any = None) -> bool:
    """True when user input is still being processed end-to-end.

    Combines:

    - ``SharedResources.http_chat_inflight`` (HTTP /chat request lifecycle, including async)
    - :class:`~agents.input_agent.InputAgent` queues: pending text (priority + normal) and sensors

    *boot_agent* is optional; when set and ``shared`` has no ``_input_agent_ref``,
    ``boot_agent.input_agent`` is used for queue depth.
    """
    try:
        if int(getattr(shared, "http_chat_inflight", 0) or 0) > 0:
            return True
    except Exception:
        pass
    ia = getattr(shared, "_input_agent_ref", None)
    if ia is None and boot_agent is not None:
        ia = getattr(boot_agent, "input_agent", None)
    if ia is None:
        return False
    if not hasattr(ia, "buffer_stats"):
        return False
    try:
        st = ia.buffer_stats()
        return (
            int(st.get("text_pending", 0) or 0) > 0
            or int(st.get("text_priority_pending", 0) or 0) > 0
            or int(st.get("sensor_pending", 0) or 0) > 0
        )
    except Exception:
        # Real InputAgent exposes buffer_stats; failure means unreadable depth — treat as busy.
        return True


def _agent_environment_active_risk(shared: Any) -> bool:
    """True when the latest ``agent_environment`` snapshot suggests hazard or alerts."""
    try:
        if not hasattr(shared, "get_agent_environment_snapshot"):
            return False
        env = shared.get_agent_environment_snapshot()
    except Exception:
        return False
    if not isinstance(env, dict) or not env:
        return False
    alerts = env.get("alerts")
    if isinstance(alerts, list) and len(alerts) > 0:
        return True
    m = env.get("metrics") if isinstance(env.get("metrics"), dict) else {}
    for key in ("threat_level", "danger_score", "risk_score", "hazard_level"):
        v = m.get(key)
        if v is True:
            return True
        try:
            if isinstance(v, (int, float)) and float(v) > 0.25:
                return True
        except (TypeError, ValueError):
            continue
    ext = env.get("extensions") if isinstance(env.get("extensions"), dict) else {}
    rb = ext.get("robot_bridge") if isinstance(ext.get("robot_bridge"), dict) else {}
    if rb.get("estop") or rb.get("emergency_stop"):
        return True
    return False


def _input_sensor_queue_suggests_speech_or_threat(ia: Any) -> bool:
    """True when queued sensor items look like speech (user talking) or explicit threat metadata."""
    peek = getattr(ia, "peek_sensor_queue", None)
    if not callable(peek):
        return False
    try:
        items = peek(48)
    except Exception:
        return False
    rms_floor = 0.06
    raw_rms = (os.environ.get("HAROMA_BG_AUDIO_ACTIVE_RMS") or "").strip()
    if raw_rms:
        try:
            rms_floor = float(raw_rms)
        except (TypeError, ValueError):
            pass
    for it in items:
        if not isinstance(it, dict):
            continue
        ch = str(it.get("channel") or "").lower()
        data = it.get("data")
        if data is None:
            data = it
        if not isinstance(data, dict):
            continue
        if data.get("threat") or data.get("danger") or data.get("alarm"):
            return True
        if ch in ("audio", "mic", "microphone", "sound", "audition"):
            if data.get("is_speech_likely"):
                return True
            if data.get("transcription"):
                return True
            try:
                if float(data.get("rms_level") or 0.0) >= rms_floor:
                    return True
            except (TypeError, ValueError):
                pass
    return False


def background_input_active(shared: Any, boot_agent: Any = None) -> bool:
    """True when foreground input should block **starting** a new heavy BackgroundAgent cycle.

    Covers: HTTP /chat lifecycle, pending chat text, ``agent_environment`` alerts/threat
    metrics, speech-like or high-RMS audio in the sensor queue. Does **not** treat
    generic ``sensor_pending`` as active unless
    ``HAROMA_BG_ACTIVE_INPUT_INCLUDES_SENSOR_PENDING=1``.
    """
    try:
        if int(getattr(shared, "http_chat_inflight", 0) or 0) > 0:
            return True
    except Exception:
        pass

    ia = getattr(shared, "_input_agent_ref", None)
    if ia is None and boot_agent is not None:
        ia = getattr(boot_agent, "input_agent", None)
    if ia is not None and hasattr(ia, "buffer_stats"):
        try:
            st = ia.buffer_stats()
            if int(st.get("text_pending", 0) or 0) > 0:
                return True
            if int(st.get("text_priority_pending", 0) or 0) > 0:
                return True
        except Exception:
            pass

    if _agent_environment_active_risk(shared):
        return True

    if ia is not None and _input_sensor_queue_suggests_speech_or_threat(ia):
        return True

    if env_flag("HAROMA_BG_ACTIVE_INPUT_INCLUDES_SENSOR_PENDING", False):
        if ia is not None and hasattr(ia, "buffer_stats"):
            try:
                st = ia.buffer_stats()
                if int(st.get("sensor_pending", 0) or 0) > 0:
                    return True
            except Exception:
                return True

    return False


def chat_input_priority_defer_non_user(shared: Any, boot_agent: Any = None) -> bool:
    """True when non-user work should yield to ongoing input (chat + InputAgent backlog)."""
    if not env_flag("HAROMA_CHAT_INPUT_PRIORITY", True):
        return False
    try:
        return input_pipeline_busy(shared, boot_agent)
    except Exception:
        # Unknown state while priority mode is on — defer non-user work.
        return True
