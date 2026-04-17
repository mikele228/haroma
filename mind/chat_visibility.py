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


def _self_identity_fallback(
    user_text: str,
    identity: Optional[Dict[str, Any]],
    persona_display_name: str,
) -> Optional[str]:
    """When the LLM yields no text, still answer basic name/identity questions from config."""
    t = (user_text or "").strip().lower()
    if not t:
        return None
    markers = (
        "your name",
        "what are you called",
        "who are you",
        "what's your name",
        "what is your name",
        "how should i call you",
        "what do you call yourself",
        "introduce yourself",
        "should i call you",
        "do you call yourself",
    )
    if not any(m in t for m in markers):
        if "name" in t and len(t) <= 48:
            pass  # e.g. "name?" / "your name?"
        else:
            return None
    id_ = identity if isinstance(identity, dict) else {}
    ens = str(id_.get("essence_name") or "").strip()
    ves = str(id_.get("vessel") or "").strip()
    display = ens or (persona_display_name or "").strip()
    if not display:
        return None
    if ves and ves.lower() != display.lower():
        return f"I'm {display}; I go by {ves} here."
    return f"I'm {display}."


def _visible_line(text: str) -> str:
    """Apply end-marker truncation; empty after truncation is treated as no line."""
    t = truncate_chat_at_end_marker(text)
    return t if str(t).strip() else ""


def resolve_chat_visible_text(
    action: Dict[str, Any],
    llm_context: Optional[Dict[str, Any]] = None,
    *,
    user_text: str = "",
    identity: Optional[Dict[str, Any]] = None,
    persona_display_name: str = "",
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
    # Prefer model answer for both structured JSON and HAROMA_LLM_CHAT_ONLY plain text;
    # deliberative action["text"] can still be a placeholder if those paths disagree.
    if src in ("llm_context_reasoning", "chat_only", "dummy_probe"):
        _parsed = str(lc.get("answer") or "").strip()
        if _parsed:
            vis = _visible_line(_parsed)
            if vis:
                return vis
    raw = action.get("text")
    text_out = str(raw).strip() if raw is not None and str(raw).strip() else ""
    if text_out.strip() == CHAT_RESPONSE_UNKNOWN.strip():
        text_out = ""
    if text_out:
        vis = _visible_line(text_out)
        if vis:
            return vis
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
        vis = _visible_line(_lc_ans)
        if vis:
            return vis
    fb = _self_identity_fallback(user_text, identity, persona_display_name)
    if fb:
        return fb
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
    ut = str(out.get("_chat_resolve_user_text") or "").strip()
    ident = out.get("_chat_resolve_identity")
    ident_d = ident if isinstance(ident, dict) else None
    pn = str(out.get("persona_name") or "").strip()
    kw = dict(
        user_text=ut,
        identity=ident_d,
        persona_display_name=pn,
    )
    if r is None or (isinstance(r, str) and not r.strip()):
        out["response"] = resolve_chat_visible_text({"text": ""}, lc, **kw)
    else:
        out["response"] = resolve_chat_visible_text({"text": str(r)}, lc, **kw)
    out.pop("_chat_resolve_user_text", None)
    out.pop("_chat_resolve_identity", None)
    _final = out.get("response")
    if not isinstance(_final, str) or not str(_final).strip():
        out["response"] = CHAT_RESPONSE_UNKNOWN
    return out
