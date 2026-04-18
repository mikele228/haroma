"""Single call-site for PersonaAgent 13.2b packed-LLM inputs (gates + KG + timeout).

Avoids reordering mistakes between ``organic_confidence``, path flags, KG selection,
and ``defer_llm_episode_bind``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mind.cycle_flow import organic_confidence
from mind.packed_llm_inputs import should_skip_full_pack_messages_for_llm
from mind.packed_llm_kg import kg_triples_for_packed_llm_prompt
from mind.packed_llm_paths import PackedLlmPathState, compute_packed_llm_path_state, defer_llm_episode_bind
from mind.packed_llm_timeout_override import packed_llm_generate_chat_timeout_override


@dataclass(frozen=True)
class PackedLlmCycleInputs:
    """Pre-LLM bundle for :func:`mind.cycle_flow.run_llm_context_reasoning_phase`."""

    user_text: str
    organic: float
    user_message: bool
    path: PackedLlmPathState
    skip_full_pack_messages: bool
    kg_triples: Optional[List[Any]]
    timeout_override: Optional[float]
    defer_episode_bind: bool


def build_packed_llm_cycle_inputs(
    *,
    text: str,
    content: str,
    llm_centric: bool,
    chat_llm_primary: bool,
    role: str,
    has_external: bool,
    is_internal: bool,
    trueself_agent: bool,
    user_or_traced_turn: bool,
    gate_reasoning: bool,
    over_budget: bool,
    deliberative_flag: bool,
    recalled_memories: list,
    reasoning_result: Any,
    appraisal_result: Any,
    organic_skip_threshold: float,
    knowledge_summary: Dict[str, Any],
    knowledge_graph: Any,
    nlu_result: Optional[Dict[str, Any]],
) -> PackedLlmCycleInputs:
    """Compute user text, organic score, path state, KG triples, timeout, and defer bind."""
    user_text = text or content or ""
    organic = organic_confidence(
        recalled_memories=recalled_memories,
        reasoning_result=reasoning_result,
        appraisal_result=appraisal_result,
    )
    user_message = bool(str(user_text).strip())
    path = compute_packed_llm_path_state(
        llm_centric=llm_centric,
        chat_llm_primary=chat_llm_primary,
        role=role,
        has_external=has_external,
        is_internal=is_internal,
        trueself_agent=trueself_agent,
        user_or_traced_turn=user_or_traced_turn,
        user_message=user_message,
        gate_reasoning=gate_reasoning,
        over_budget=over_budget,
        organic=organic,
        organic_skip_threshold=organic_skip_threshold,
    )
    llm_ctx = path.llm_ctx_enabled
    skip_full = should_skip_full_pack_messages_for_llm()
    kg = kg_triples_for_packed_llm_prompt(
        llm_ctx_enabled=llm_ctx,
        skip_full_pack_messages=skip_full,
        knowledge_summary=knowledge_summary,
        knowledge_graph=knowledge_graph,
        nlu_result=nlu_result,
    )
    timeout_override = packed_llm_generate_chat_timeout_override(
        apply_true_self_user_chat_cap=(
            llm_ctx and trueself_agent and user_or_traced_turn and not is_internal
        ),
    )
    defer = defer_llm_episode_bind(deliberative_flag, llm_ctx)
    return PackedLlmCycleInputs(
        user_text=user_text,
        organic=organic,
        user_message=user_message,
        path=path,
        skip_full_pack_messages=skip_full,
        kg_triples=kg,
        timeout_override=timeout_override,
        defer_episode_bind=defer,
    )
