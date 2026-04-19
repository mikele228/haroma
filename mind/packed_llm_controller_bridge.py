"""Packed LLM phase for :class:`~mind.control.ElarionController` (13.2b alignment).

Mirrors PersonaAgent wiring — :func:`~mind.cycle_flow.run_llm_context_reasoning_phase` after
symbolic reasoning — so embedded ``run_cycle`` can populate ``episode.llm_context`` when
``HAROMA_CONTROLLER_PACKED_LLM`` is enabled. Implementation delegates to
:mod:`mind.packed_llm_reasoning_invoke` with the same ``PackedLlmCycleInputs`` bundle as Persona.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from mind.cycle_flow import ORGANIC_PACKED_LLM_SKIP_THRESHOLD
from mind.haroma_settings import haroma_controller_packed_llm_enabled
from mind.packed_llm_agent_state import build_agent_state_json_for_packed_llm
from mind.packed_llm_cycle_inputs import build_packed_llm_cycle_inputs
from mind.packed_llm_centric import llm_centric_enabled_for_persona_cycle
from mind.packed_llm_discourse import discourse_context_for_packed_llm
from mind.packed_llm_paths import chat_llm_primary_env_enabled
from mind.packed_llm_reasoning_invoke import (
    invoke_run_llm_context_reasoning_phase,
    summarize_law_value_managers,
)


def controller_packed_llm_enabled() -> bool:
    """When True, :meth:`~mind.control.ElarionController.run_cycle` runs the packed LLM phase."""
    return haroma_controller_packed_llm_enabled()


def run_packed_llm_phase_for_elarion_controller(
    ctrl: Any,
    episode: Any,
    *,
    role: str,
    content: str,
    text: str,
    has_external: bool,
    gate_reasoning: bool,
    reasoning_result: Any,
    appraisal_result: Any,
    active_goals: List[Dict[str, Any]],
    identity_summary: Dict[str, Any],
    nlu_result: Optional[Dict[str, Any]],
    knowledge_summary: Dict[str, Any],
    cycle_id: int,
    deliberative_flag: bool = False,
) -> Dict[str, Any]:
    """Run 13.2b packed-context LLM; returns result dict (may be skipped)."""
    if not controller_packed_llm_enabled():
        return {"source": "skipped"}

    user_or_traced_turn = bool(has_external and str(text or content).strip())
    _llm_centric = llm_centric_enabled_for_persona_cycle(
        is_internal=False,
        trueself_agent=False,
        user_or_traced_turn=user_or_traced_turn,
        goal_board=getattr(ctrl, "goal_board", None),
    )
    _chat_llm_primary = chat_llm_primary_env_enabled()

    _pl = build_packed_llm_cycle_inputs(
        text=text,
        content=content,
        llm_centric=_llm_centric,
        chat_llm_primary=_chat_llm_primary,
        role=role,
        has_external=has_external,
        is_internal=False,
        trueself_agent=False,
        user_or_traced_turn=user_or_traced_turn,
        gate_reasoning=gate_reasoning,
        over_budget=False,
        deliberative_flag=deliberative_flag,
        recalled_memories=episode.recalled_memories,
        reasoning_result=reasoning_result,
        appraisal_result=appraisal_result,
        organic_skip_threshold=ORGANIC_PACKED_LLM_SKIP_THRESHOLD,
        knowledge_summary=knowledge_summary,
        knowledge_graph=ctrl.knowledge,
        nlu_result=nlu_result,
    )
    _llm_ctx_enabled = _pl.path.llm_ctx_enabled

    _law_mgr = getattr(ctrl, "law", None)
    _val_mgr = getattr(ctrl, "value", None)
    _law_sum, _val_sum = summarize_law_value_managers(_law_mgr, _val_mgr)

    _agent_state_json = build_agent_state_json_for_packed_llm(
        deliberative_flag=deliberative_flag,
        llm_ctx_enabled=_llm_ctx_enabled,
        law_summary=_law_sum,
        val_mgr=_val_mgr,
        value_summary=_val_sum,
        state=ctrl,
        boot_agent_ref=None,
        identity_summary=identity_summary,
        personality_summary={},
        active_goals=active_goals,
        episode=episode,
    )

    _discourse_llm = discourse_context_for_packed_llm(
        ctrl.conversation,
        cycle_id=cycle_id,
        speaker_key=None,
        session_uid=False,
        first_encounter_asks_name=False,
        deliberative_flag=deliberative_flag,
        llm_ctx_enabled=bool(_llm_ctx_enabled),
        role=str(role or ""),
    )

    _memory_forest_seed = ""
    if hasattr(ctrl.memory, "build_seed_context"):
        try:
            _memory_forest_seed = ctrl.memory.build_seed_context(
                query_text=text or content or "",
                recalled=episode.recalled_memories,
                env_snapshot=None,
            )
        except Exception:
            _memory_forest_seed = ""

    return invoke_run_llm_context_reasoning_phase(
        pl=_pl,
        llm_backend=ctrl.llm_backend,
        episode=episode,
        memory_forest=ctrl.memory,
        identity_summary=identity_summary,
        personality_summary={},
        active_goals=active_goals,
        law_summary=_law_sum,
        value_summary=_val_sum,
        discourse_context=_discourse_llm,
        nlu_result=nlu_result,
        memory_forest_seed=_memory_forest_seed,
        llm_centric=_llm_centric,
        deliberative=deliberative_flag,
        agent_state_json=_agent_state_json,
    )
