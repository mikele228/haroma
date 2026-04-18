"""Persona yield rules while HTTP ``/chat`` is active.

Keeps :class:`agents.persona_agent.PersonaAgent` readable: one place for
``http_chat_inflight`` checks used to defer inner-dialogue relay, skip recall,
and treat internal cycles as over-budget for optional phases.
"""

from __future__ import annotations

from typing import Any


def http_chat_inflight_positive(shared: Any) -> bool:
    """True when at least one HTTP chat request is in flight (async or sync)."""
    try:
        return int(getattr(shared, "http_chat_inflight", 0) or 0) > 0
    except Exception:
        return False


def inner_relay_should_requeue(message_type: str) -> bool:
    """Relay types deferred in :meth:`~agents.persona_agent.PersonaAgent._handle_relay_message`."""
    return message_type in ("inner_dialogue", "dialogue_reply")


def defer_inner_cycle_before_neural(is_internal: bool, shared: Any) -> bool:
    """Inner cycles should not acquire neural locks — re-queue for a later tick."""
    return is_internal and http_chat_inflight_positive(shared)


def skip_semantic_recall_for_internal(is_internal: bool, shared: Any) -> bool:
    """Skip expensive ``memory.recall`` for inner traffic while HTTP chat is active."""
    return is_internal and http_chat_inflight_positive(shared)


def internal_treat_as_over_budget(is_internal: bool, shared: Any) -> bool:
    """Optional pipeline steps that honor ``_over_budget()`` should skip for inner + HTTP."""
    return is_internal and http_chat_inflight_positive(shared)
