"""Keyword bundle for :func:`mind.cycle_flow.run_llm_context_reasoning_phase` (PersonaAgent 13.2b).

Keeps the call site aligned with the phase signature when parameters are added or renamed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_persona_packed_llm_reasoning_phase_kwargs(
    *,
    enabled: bool,
    llm_backend: Any,
    user_text: str,
    recalled_memories: list,
    identity_summary: Dict[str, Any],
    personality_summary: Dict[str, float],
    active_goals: List[Any],
    law_summary: Dict[str, Any],
    value_summary: Dict[str, Any],
    knowledge_triples: Optional[List[Any]],
    discourse_context: str,
    nlu_result: Optional[Dict[str, Any]],
    memory_forest_seed: str,
    llm_centric: bool,
    episode: Any,
    memory_forest: Any,
    trace_label: str,
    timeout_override: Optional[float],
    deliberative: bool,
    agent_state_json: str,
    bind_episode: bool,
    agent_environment: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return kwargs for ``run_llm_context_reasoning_phase``.

    *agent_environment*: Persona leaves ``None`` so the phase derives training metadata from *episode*.
    """
    return {
        "enabled": enabled,
        "llm_backend": llm_backend,
        "user_text": user_text,
        "recalled_memories": recalled_memories,
        "identity_summary": identity_summary,
        "personality_summary": personality_summary,
        "active_goals": active_goals,
        "law_summary": law_summary,
        "value_summary": value_summary,
        "knowledge_triples": knowledge_triples,
        "discourse_context": discourse_context,
        "nlu_result": nlu_result,
        "memory_forest_seed": memory_forest_seed,
        "llm_centric": llm_centric,
        "episode": episode,
        "memory_forest": memory_forest,
        "trace_label": trace_label,
        "timeout_override": timeout_override,
        "deliberative": deliberative,
        "agent_state_json": agent_state_json,
        "bind_episode": bind_episode,
        "agent_environment": agent_environment,
    }
