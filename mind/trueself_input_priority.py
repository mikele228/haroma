"""Stable ordering for TrueSelf input batches: user / HTTP turns before sensor backlog."""

from __future__ import annotations

from typing import List, Tuple

from agents.message_bus import Message
from mind.chat_pipeline_log import trace_id_from_message


def input_tier(m: Message) -> int:
    """Lower runs first. Align with :func:`trace_id_from_message` (metadata + slot)."""
    if trace_id_from_message(m) is not None:
        return 0
    c = getattr(m, "content", None)
    if isinstance(c, dict):
        if str(c.get("source") or "").lower() == "user" and str(c.get("text") or "").strip():
            return 0
    md = m.metadata or {}
    if md.get("_response_slot") is not None:
        return 1
    return 2


def is_sensor_pulse_no_chat_text(m: Message) -> bool:
    """True for standalone sensor dispatch (empty ``text``, ``source=sensor``, tier 2).

    Those messages do not need TrueSelf's full persona cycle; skipping frees the mailbox
    for HTTP ``/chat`` without waiting on recall/appraisal/embed.
    """
    if input_tier(m) != 2:
        return False
    c = getattr(m, "content", None)
    if not isinstance(c, dict):
        return False
    if str(c.get("source") or "").lower() != "sensor":
        return False
    return not str(c.get("text") or "").strip()


def prioritize_trueself_input_messages(msgs: List[Message]) -> List[Message]:
    """Process HTTP ``/chat`` and traced turns before untraced sensor/vision-only input.

    Uses the same trace resolution as pipeline logging (including ``cognitive_trace_id``
    on ``_response_slot``). Also boosts ``source=user`` text so user chat is not starved
    when metadata is incomplete.
    """
    if not msgs:
        return msgs
    return [m for _, m in sorted(enumerate(msgs), key=lambda im: (input_tier(im[1]), im[0]))]


def partition_trueself_input_for_tick(msgs: List[Message]) -> Tuple[List[Message], List[Message]]:
    """If the batch mixes priority input (tier 0–1) with sensor-only (tier 2), return
    ``(process_now, requeue)`` so one tick does not run packed LLM for every vision
    frame before returning. When there is no priority traffic, process the full batch.

    Calls :func:`prioritize_trueself_input_messages` once; ``process_now + requeue`` preserves
    that order.
    """
    if len(msgs) <= 1:
        return msgs, []
    ordered = prioritize_trueself_input_messages(msgs)
    has_priority = any(input_tier(m) <= 1 for m in ordered)
    if not has_priority:
        return ordered, []
    now = [m for m in ordered if input_tier(m) <= 1]
    later = [m for m in ordered if input_tier(m) > 1]
    if not later:
        return ordered, []
    return now, later
