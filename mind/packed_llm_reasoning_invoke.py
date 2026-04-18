"""Single call site for Persona/controller 13.2b — ``run_llm_context_reasoning_phase`` + kwargs bundle."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from core.cognitive_null import is_cognitive_null
from mind.cycle_flow import TRACE_LABEL_PERSONA_PACKED_LLM, run_llm_context_reasoning_phase
from mind.packed_llm_cycle_inputs import PackedLlmCycleInputs
from mind.packed_llm_reasoning_kwargs import build_persona_packed_llm_reasoning_phase_kwargs


def summarize_law_value_managers(law_mgr: Any, val_mgr: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Law/value summaries for packed prompts, skipping null managers."""
    ls = law_mgr.summarize() if law_mgr is not None and not is_cognitive_null(law_mgr) else {}
    vs = val_mgr.summarize() if val_mgr is not None and not is_cognitive_null(val_mgr) else {}
    return ls, vs


def invoke_run_llm_context_reasoning_phase(
    *,
    pl: PackedLlmCycleInputs,
    llm_backend: Any,
    episode: Any,
    memory_forest: Any,
    identity_summary: Dict[str, Any],
    personality_summary: Dict[str, float],
    active_goals: List[Any],
    law_summary: Dict[str, Any],
    value_summary: Dict[str, Any],
    discourse_context: str,
    nlu_result: Optional[Dict[str, Any]],
    memory_forest_seed: str,
    llm_centric: bool,
    deliberative: bool,
    agent_state_json: str,
    trace_label: str = TRACE_LABEL_PERSONA_PACKED_LLM,
) -> Dict[str, Any]:
    """Run packed-context LLM from a :class:`PackedLlmCycleInputs` bundle."""
    _path = pl.path
    _enabled = _path.llm_ctx_enabled
    _defer = pl.defer_episode_bind
    return run_llm_context_reasoning_phase(
        **build_persona_packed_llm_reasoning_phase_kwargs(
            enabled=_enabled,
            llm_backend=llm_backend,
            user_text=pl.user_text,
            recalled_memories=episode.recalled_memories,
            identity_summary=identity_summary,
            personality_summary=personality_summary,
            active_goals=active_goals,
            law_summary=law_summary,
            value_summary=value_summary,
            knowledge_triples=pl.kg_triples,
            discourse_context=discourse_context,
            nlu_result=nlu_result,
            memory_forest_seed=memory_forest_seed,
            llm_centric=llm_centric,
            episode=episode,
            memory_forest=memory_forest,
            trace_label=trace_label,
            timeout_override=pl.timeout_override,
            deliberative=deliberative,
            agent_state_json=agent_state_json,
            bind_episode=not _defer,
        )
    )
