"""Align deliberative ``action`` with packed-context LLM output — pure helper.

When :class:`agents.persona_agent.PersonaAgent` runs the primary packed LLM path,
deliberative ``action[\"text\"]`` can still win in :func:`mind.chat_visibility.resolve_chat_visible_text`
unless the model answer is copied onto ``action`` here. Sources allowed are
:data:`LLM_CONTEXT_SOURCES_PREFER_PACKED_ANSWER` (same rules as chat visibility).

:func:`build_llm_centric_action` builds the deliberative action dict when
``HAROMA_LLM_CENTRIC`` drives the turn directly from packed LLM output.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from mind.chat_visibility import LLM_CONTEXT_SOURCES_PREFER_PACKED_ANSWER


def build_llm_centric_action(
    *,
    llm_answer: Any,
    llm_context_result: Dict[str, Any],
    is_in_conversation: bool,
) -> Dict[str, Any]:
    """Deliberative action payload when the packed LLM answer is the sole response."""
    text = str(llm_answer).strip()
    try:
        conf = float(llm_context_result.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return {
        "text": text,
        "action_type": "respond" if is_in_conversation else "reflect",
        "strategy": "llm_context",
        "composition": None,
        "deliberation": {"candidates": [], "winner": "llm_centric"},
        "confidence": conf,
        "reasoning": "llm_centric_direct",
        "law_bound": False,
        "symbolic_law": {"compliant": True, "violation_count": 0},
        "timestamp": time.time(),
    }


def merge_packed_llm_answer_into_action(
    action: Dict[str, Any],
    llm_context_result: Dict[str, Any],
    *,
    llm_primary_path: bool,
    llm_centric: bool,
) -> Dict[str, Any]:
    """If packed LLM produced a user-visible answer, copy it onto ``action[\"text\"]``.

    * ``llm_centric`` builds ``action`` from the LLM already — no merge.
    * Without ``llm_primary_path``, the packed pass did not drive this turn — no merge.
    """
    if not llm_primary_path or llm_centric:
        return action
    src = str(llm_context_result.get("source") or "").strip().lower()
    ans = str(llm_context_result.get("answer") or "").strip()
    if ans and src in LLM_CONTEXT_SOURCES_PREFER_PACKED_ANSWER:
        out = dict(action)
        out["text"] = ans
        out["strategy"] = "llm_context"
        return out
    return action
