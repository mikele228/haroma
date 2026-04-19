"""Discourse string shaping for packed LLM (PersonaAgent)."""

from __future__ import annotations

from typing import Any, Optional

from mind.dialogue_phases import enrich_discourse_for_dialogue_phases

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
    user_id: Optional[str] = None,
    display_name: Optional[str] = None,
    user_text: str = "",
    trace_id: Optional[str] = None,
    persona_name: str = "",
    essence_name: str = "",
    deliberative_flag: bool = False,
    llm_ctx_enabled: bool = False,
    role: str = "",
) -> str:
    """Conversation summary string for packed LLM, including optional first-exchange hint.

    When :envvar:`HAROMA_DIALOGUE_PHASE` >= 1 and *user_id* or *display_name* is set, a
    short ``[Session]`` line is appended. Phase >= 5 appends ``[Rel] encounter=…`` when
    *session_uid* is true. Phase >= 2 may append a correction hint when *user_text* looks
    like a user correction. Phase >= 3 appends ``[Eval]``; phase >= 4 appends
    ``[Voice] persona=…`` / ``essence=…`` when set; phase >= 6 appends
    ``[Cog] deliberative=…`` from *deliberative_flag*; phase >= 7 appends
    ``[Turn] chars=…`` from stripped *user_text*.
    """
    summary_key = speaker_key if session_uid else None
    base = (
        conversation.get_context_summary(summary_key)
        if conversation.is_in_conversation(cycle_id)
        else ""
    )
    merged = merge_first_encounter_discourse_hint(base, first_encounter_asks_name)
    return enrich_discourse_for_dialogue_phases(
        merged,
        cycle_id=cycle_id,
        user_id=user_id,
        display_name=display_name,
        user_text=user_text,
        trace_id=trace_id,
        persona_name=persona_name,
        essence_name=essence_name,
        session_uid=session_uid,
        first_encounter_asks_name=first_encounter_asks_name,
        deliberative_flag=deliberative_flag,
        llm_ctx_enabled=llm_ctx_enabled,
        role=role,
    )
