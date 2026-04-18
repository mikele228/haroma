"""Pure packed-LLM routing flags for PersonaAgent (primary / classic / ctx enabled).

Keeps gate logic in one place so ElarionController and tests can align without
copy-pasting boolean expressions.
"""

from __future__ import annotations

from dataclasses import dataclass

from mind.haroma_settings import haroma_chat_llm_primary_enabled


@dataclass(frozen=True)
class PackedLlmPathState:
    """Routing for packed ``generate_chat`` and related KG / timeout wiring."""

    packed_llm_eligible: bool
    conversant_chat: bool
    llm_primary_path: bool
    llm_classic_path: bool
    llm_ctx_enabled: bool


def compute_packed_llm_path_state(
    *,
    llm_centric: bool,
    chat_llm_primary: bool,
    role: str,
    has_external: bool,
    is_internal: bool,
    trueself_agent: bool,
    user_or_traced_turn: bool,
    user_message: bool,
    gate_reasoning: bool,
    over_budget: bool,
    organic: float,
    organic_skip_threshold: float,
) -> PackedLlmPathState:
    """Mirror :mod:`agents.persona_agent` packed-LLM gates (13.2b)."""
    packed_llm_eligible = user_message and (not trueself_agent or user_or_traced_turn)
    conversant_chat = (
        role == "conversant" and has_external and packed_llm_eligible and not is_internal
    )
    llm_primary_path = chat_llm_primary and conversant_chat
    llm_classic_path = (
        not is_internal
        and packed_llm_eligible
        and gate_reasoning
        and not over_budget
        and organic < organic_skip_threshold
    )
    llm_ctx_enabled = llm_centric or llm_primary_path or llm_classic_path
    return PackedLlmPathState(
        packed_llm_eligible=packed_llm_eligible,
        conversant_chat=conversant_chat,
        llm_primary_path=llm_primary_path,
        llm_classic_path=llm_classic_path,
        llm_ctx_enabled=llm_ctx_enabled,
    )


def defer_llm_episode_bind(deliberative_flag: bool, llm_ctx_enabled: bool) -> bool:
    """When True, :func:`mind.cycle_flow.run_llm_context_reasoning_phase` uses ``bind_episode=False``.

    Deferral lets :func:`mind.deliberative_llm_merge.complete_deferred_deliberative_llm_context`
    merge deliberative scores before a single ``episode.bind_llm_context``.
    """
    return bool(deliberative_flag and llm_ctx_enabled)


def chat_llm_primary_env_enabled() -> bool:
    """``HAROMA_CHAT_LLM_PRIMARY`` — conversant fast path uses packed LLM as primary voice."""
    return haroma_chat_llm_primary_enabled()
