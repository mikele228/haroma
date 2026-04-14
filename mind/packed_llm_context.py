"""
Packed-context LLM — **host-side sections** of the user message.

Pipeline (single direction):

1. Integrators send ``agent_environment`` (HTTP /chat or ``POST /agent/environment``).
2. :class:`agents.shared_resources.SharedResources` validates, merges
   ``extensions.robot_body`` from sensors, and exposes
   ``get_agent_environment_snapshot()`` for cognition. Executor feedback lands in
   ``extensions.robot_bridge`` (see :mod:`integrations.robot_http_bridge`).
3. :func:`host_environment_sections_for_prompt` turns the bound snapshot into
   labeled text blocks (environment summary, robot body JSON, optional bridge
   feedback) for :func:`engine.LLMContextReasoner.build_messages`.

Structured LLM outputs (``candidate_actions``, ``body_actions``, etc.) are merged
into episode payloads via :func:`optional_llm_structured_fields`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def host_environment_sections_for_prompt(
    agent_environment: Optional[Dict[str, Any]],
) -> List[str]:
    """Build ``[ENVIRONMENT STATE]``, ``[ROBOT BODY STATE]``, ``[ROBOT BRIDGE FEEDBACK]`` (in order).

    Returns zero to three strings, each ready to append to the user message
    (including the bracketed title line). Failures in one section do not drop the others.
    """
    out: List[str] = []
    if not isinstance(agent_environment, dict) or not agent_environment:
        return out

    try:
        from mind.environment_context import environment_summary_for_prompt
        from mind.environment_prompt_budgets import PACKED_LLM_ENV_SUMMARY_MAX_CHARS

        summary = environment_summary_for_prompt(
            agent_environment,
            max_chars=PACKED_LLM_ENV_SUMMARY_MAX_CHARS,
        )
        if summary:
            out.append("[ENVIRONMENT STATE]\n" + summary)
    except Exception:
        pass

    try:
        from mind.environment_prompt_budgets import PACKED_LLM_ROBOT_BODY_MAX_CHARS
        from mind.robot_body_state import robot_body_prompt_block

        ext = agent_environment.get("extensions")
        rb = ext.get("robot_body") if isinstance(ext, dict) else None
        if isinstance(rb, dict) and rb:
            block = robot_body_prompt_block(rb, max_chars=PACKED_LLM_ROBOT_BODY_MAX_CHARS)
            if block:
                out.append("[ROBOT BODY STATE]\n" + block)
    except Exception:
        pass

    try:
        from mind.environment_prompt_budgets import PACKED_LLM_ROBOT_BRIDGE_MAX_CHARS
        from mind.robot_execution_contract import robot_bridge_prompt_block

        ext2 = agent_environment.get("extensions")
        rbr = ext2.get("robot_bridge") if isinstance(ext2, dict) else None
        if isinstance(rbr, dict) and rbr:
            bblock = robot_bridge_prompt_block(rbr, max_chars=PACKED_LLM_ROBOT_BRIDGE_MAX_CHARS)
            if bblock:
                out.append("[ROBOT BRIDGE FEEDBACK]\n" + bblock)
    except Exception:
        pass

    return out


def optional_llm_structured_fields(llm_ctx: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Subset of *llm_ctx* for diagnostics / HTTP: only non-empty structured keys."""
    if not isinstance(llm_ctx, dict):
        return {}
    keys = ("candidate_actions", "body_actions", "chosen_action", "deliberative_scores")
    return {k: llm_ctx[k] for k in keys if llm_ctx.get(k)}
