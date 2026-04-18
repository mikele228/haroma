"""User-visible defaults when the action loop yields no utterance.

Constants and :func:`truncate_chat_at_end_marker` are re-exported from
:mod:`mind.cognitive_contracts` for a single import surface in application code.
"""

from __future__ import annotations

import os

CHAT_RESPONSE_UNKNOWN = "I don't know."


def trim_repeated_suffix_cycles(
    text: str,
    *,
    min_period: int = 20,
    max_period: int = 4000,
    max_passes: int = 64,
) -> str:
    """Drop a trailing block that repeats the preceding block verbatim (model stutter)."""
    t = text.rstrip()
    n = len(t)
    if n < 2 * min_period:
        return t
    passes = 0
    while passes < max_passes:
        passes += 1
        n = len(t)
        if n < 2 * min_period:
            break
        upper = min(n // 2, max_period)
        changed = False
        for size in range(upper, min_period - 1, -1):
            if n < 2 * size:
                continue
            if t[-size:] == t[-2 * size : -size]:
                t = t[:-size].rstrip()
                changed = True
                break
        if not changed:
            break
    return t


def truncate_chat_at_end_marker(text: str) -> str:
    """Return text for chat UIs; drop trailing model junk after ``[END_OF_TEXT]``.

    Some models append narration or `` ```json `` blocks after an end marker; the
    user-facing reply is only the portion before the first case-insensitive
    ``[END_OF_TEXT]``.
    """
    s = str(text)
    lower = s.lower()
    marker = "[end_of_text]"
    i = lower.find(marker)
    if i >= 0:
        return s[:i].rstrip()
    return s.strip()


def _nonjson_max_chars() -> int:
    raw = str(os.environ.get("HAROMA_LLM_NONJSON_MAX_CHARS", "2000") or "2000").strip()
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return 2000
    return max(200, min(v, 8000))


def sanitize_llm_plain_answer(text: str) -> str:
    """End marker, anti-stutter, and length cap for raw model prose (non-JSON path)."""
    s = truncate_chat_at_end_marker(str(text))
    s = trim_repeated_suffix_cycles(s)
    mx = _nonjson_max_chars()
    if len(s) > mx:
        s = s[: mx - 1].rstrip() + "…"
    return s.strip()


def clean_packed_json_answer_text(text: str, *, max_chars: int = 8000) -> str:
    """Trim markers and stutter from parsed JSON ``answer`` strings; cap length."""
    s = truncate_chat_at_end_marker(str(text))
    s = trim_repeated_suffix_cycles(s)
    if len(s) > max_chars:
        s = s[: max_chars - 1].rstrip() + "…"
    return s.strip()

# Packed-context LLM did not return in time (see HAROMA_LLM_CONTEXT_TIMEOUT_SEC).
CHAT_RESPONSE_LLM_TIMEOUT = (
    "The local model hit the time limit before it answered. Large GGUF weights "
    "often need many minutes on CPU for the first reply. Try again, or raise "
    "HAROMA_LLM_CONTEXT_TIMEOUT_SEC (or set it to 0 for no cap on trusted hosts)."
)

CHAT_RESPONSE_LLM_ERROR = "The language model hit an error. Check the server console or try again."

# Packed-context path expects JSON; plain chat or malformed output yields these sources.
CHAT_RESPONSE_LLM_UNPARSEABLE = (
    "The model returned text we could not parse as JSON (expected an \"answer\" "
    "field). Try another model, reduce prompt size, or enable HAROMA_LLM_CHAT_ONLY=1 "
    "for plain replies."
)

CHAT_RESPONSE_LLM_NO_REPLY = (
    "The model produced no reply text for this turn. If the run took minutes, "
    "the local decode may be CPU-bound — try GPU offload, a smaller GGUF, or "
    "lower HAROMA_LLM_CONTEXT_MAX_TOKENS."
)

CHAT_RESPONSE_LLM_UNAVAILABLE = (
    "No language model is available (backend missing or not loaded). "
    "Check resource tier and LLM configuration."
)
