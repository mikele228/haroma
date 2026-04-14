"""User-visible defaults when the action loop yields no utterance.

Constants and :func:`truncate_chat_at_end_marker` are re-exported from
:mod:`mind.cognitive_contracts` for a single import surface in application code.
"""

CHAT_RESPONSE_UNKNOWN = "I don't know."


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
