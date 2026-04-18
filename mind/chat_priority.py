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
  optional persistence still run.

Set ``HAROMA_CHAT_INPUT_PRIORITY=0`` to restore previous interleaving.
"""

from __future__ import annotations

from typing import Any

from utils.coerce_bool import env_flag

__all__ = [
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


def chat_input_priority_defer_non_user(shared: Any, boot_agent: Any = None) -> bool:
    """True when non-user work should yield to ongoing input (chat + InputAgent backlog)."""
    if not env_flag("HAROMA_CHAT_INPUT_PRIORITY", True):
        return False
    try:
        return input_pipeline_busy(shared, boot_agent)
    except Exception:
        # Unknown state while priority mode is on — defer non-user work.
        return True
