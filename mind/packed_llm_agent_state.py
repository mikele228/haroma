"""Deliberative agent state JSON for packed LLM prompts (PersonaAgent 13.2b).

Extracted so snapshot assembly stays testable and does not sprawl in
:class:`~agents.persona_agent.PersonaAgent`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mind.cycle_flow import build_trueself_state_snapshot, serialize_state_snapshot


def _value_prompt_for_snapshot(val_mgr: Any, value_summary: Dict[str, Any]) -> Dict[str, Any]:
    try:
        eng = getattr(val_mgr, "engine", None) if val_mgr is not None else None
        if eng is not None and hasattr(eng, "summarize_for_prompt"):
            return eng.summarize_for_prompt()
    except Exception:
        pass
    return value_summary


def peek_sensor_queue_for_snapshot(state: Any, boot_agent_ref: Any) -> List[Dict[str, Any]]:
    """Up to 32 recent sensor readings for :func:`build_trueself_state_snapshot`."""
    try:
        ia = getattr(state, "_input_agent_ref", None)
        if ia is None and boot_agent_ref is not None:
            ia = getattr(boot_agent_ref, "input_agent", None)
        if ia is not None and hasattr(ia, "peek_sensor_queue"):
            return ia.peek_sensor_queue(32)
    except Exception:
        pass
    return []


def build_agent_state_json_for_packed_llm(
    *,
    deliberative_flag: bool,
    llm_ctx_enabled: bool,
    law_summary: Dict[str, Any],
    val_mgr: Any,
    value_summary: Dict[str, Any],
    state: Any,
    boot_agent_ref: Any,
    identity_summary: Optional[Dict[str, Any]],
    personality_summary: Optional[Dict[str, float]],
    active_goals: Optional[List[Dict[str, Any]]],
    episode: Any,
) -> str:
    """JSON string for ``agent_state_json`` in packed LLM, or ``\"\"`` when not used."""
    if not (deliberative_flag and llm_ctx_enabled):
        return ""
    vp = _value_prompt_for_snapshot(val_mgr, value_summary)
    sensors = peek_sensor_queue_for_snapshot(state, boot_agent_ref)
    snap = build_trueself_state_snapshot(
        sensors=sensors,
        identity_summary=identity_summary,
        personality_summary=personality_summary,
        active_goals=active_goals,
        law_summary=law_summary,
        value_summary=vp,
        drives=episode.drives if hasattr(episode, "drives") else None,
        affect=episode.affect,
    )
    return serialize_state_snapshot(snap)
