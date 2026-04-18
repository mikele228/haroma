"""Discourse string shaping for packed LLM (PersonaAgent)."""

from __future__ import annotations

from typing import Any, Optional

_FIRST_ENCOUNTER_NAME_HINT = (
    "[Discourse note: First exchange with this client identity. "
    "Greet briefly; if they have not introduced themselves, "
    "ask what they would like you to call them.]"
)


def merge_first_encounter_discourse_hint(
    discourse_llm: str,
    first_encounter_asks_name: bool,
) -> str:
    """Append first-exchange hint when the gate requests a name prompt."""
    if not first_encounter_asks_name:
        return discourse_llm
    if discourse_llm:
        return discourse_llm + " | " + _FIRST_ENCOUNTER_NAME_HINT
    return _FIRST_ENCOUNTER_NAME_HINT


def discourse_context_for_packed_llm(
    conversation: Any,
    *,
    cycle_id: int,
    speaker_key: Optional[str],
    session_uid: bool,
    first_encounter_asks_name: bool,
) -> str:
    """Conversation summary string for packed LLM, including optional first-exchange hint."""
    summary_key = speaker_key if session_uid else None
    base = (
        conversation.get_context_summary(summary_key)
        if conversation.is_in_conversation(cycle_id)
        else ""
    )
    return merge_first_encounter_discourse_hint(base, first_encounter_asks_name)
