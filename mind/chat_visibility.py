"""User-visible chat text — pure helpers decoupled from agents and HTTP.

Keeps ``PersonaAgent`` and other callers thin: one place for truncation rules
and LLM error-source fallbacks used by ``/chat`` JSON. :func:`normalize_http_chat_response`
delegates to :func:`resolve_chat_visible_text` so HTTP payloads and action dicts
follow identical rules. Application code may import the same functions from
:mod:`mind.cognitive_contracts` instead.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from mind.response_text import (
    CHAT_RESPONSE_LLM_ERROR,
    CHAT_RESPONSE_LLM_NO_REPLY,
    CHAT_RESPONSE_LLM_TIMEOUT,
    CHAT_RESPONSE_LLM_UNAVAILABLE,
    CHAT_RESPONSE_LLM_UNPARSEABLE,
    CHAT_RESPONSE_UNKNOWN,
    truncate_chat_at_end_marker,
)


def resolve_chat_visible_text(
    action: Dict[str, Any],
    llm_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Resolve the string shown as ``response`` for chat APIs.

    For packed JSON output (``source == \"llm_context_reasoning\"``), the user
    line is the parsed JSON **answer** field — not raw model text that may
    include fences or duplicate prose. Otherwise prefer non-empty
    ``action[\"text\"]``; then map ``llm_context[\"source\"]`` to error copy;
    then ``llm_context[\"answer\"]``; else unknown.
    """
    lc = llm_context if isinstance(llm_context, dict) else {}
    src = str(lc.get("source") or "").strip().lower()
    if src == "llm_context_reasoning":
        _parsed = str(lc.get("answer") or "").strip()
        if _parsed:
            return truncate_chat_at_end_marker(_parsed)
    raw = action.get("text")
    text_out = str(raw).strip() if raw is not None and str(raw).strip() else ""
    if text_out:
        return truncate_chat_at_end_marker(text_out)
    if src == "llm_timeout":
        return CHAT_RESPONSE_LLM_TIMEOUT
    if src == "llm_error":
        return CHAT_RESPONSE_LLM_ERROR
    if src == "json_parse_failed":
        return CHAT_RESPONSE_LLM_UNPARSEABLE
    if src == "llm_empty_response":
        return CHAT_RESPONSE_LLM_NO_REPLY
    if src == "llm_unavailable":
        return CHAT_RESPONSE_LLM_UNAVAILABLE
    _lc_ans = str(lc.get("answer") or "").strip()
    if _lc_ans:
        return truncate_chat_at_end_marker(_lc_ans)
    return CHAT_RESPONSE_UNKNOWN


def normalize_http_chat_response(result: Any) -> Any:
    """Ensure JSON ``response`` is never null/blank (cognitive layer may emit null).

    Uses the same rules as :func:`resolve_chat_visible_text`: ``[END_OF_TEXT]``
    truncation, ``llm_context[\"source\"]`` error copy when ``response`` is empty,
    then ``answer`` fallback. Non-dict *result* is returned unchanged.
    """
    if not isinstance(result, dict):
        return result
    out = dict(result)
    r = result.get("response")
    lc = out.get("llm_context") if isinstance(out.get("llm_context"), dict) else {}
    if r is None or (isinstance(r, str) and not r.strip()):
        out["response"] = resolve_chat_visible_text({"text": ""}, lc)
        return out
    out["response"] = resolve_chat_visible_text({"text": str(r)}, lc)
    return out
