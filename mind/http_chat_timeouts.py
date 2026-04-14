"""HTTP-layer wait duration for /chat, /chat/wait, and /status (must exceed packed-LLM cap when unset)."""

import os

from mind.cognitive_contracts import llm_context_timeout_seconds


def http_chat_wait_sec(depth: str = "normal") -> int:
    """Seconds for HTTP /chat to wait on the cognitive slot (legacy *depth* ignored)."""
    raw = str(os.environ.get("HAROMA_CHAT_TIMEOUT", "") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            print(
                "[Elarion-v2] HAROMA_CHAT_TIMEOUT invalid — using LLM-cap-based default",
                flush=True,
            )
    tlim = llm_context_timeout_seconds()
    if tlim is None:
        return 600
    return max(540, int(tlim) + 120)
