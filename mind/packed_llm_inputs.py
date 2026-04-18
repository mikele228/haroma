"""Input gating for expensive packed-LLM prompt build (KG, full messages).

When ``HAROMA_LLM_CHAT_ONLY`` or synthetic dummy without ``HAROMA_LLM_DUMMY_FULL_PACK``,
:mod:`engine.LLMContextReasoner` uses a short prompt — skip KG selection here too.
"""

from __future__ import annotations

from utils.coerce_bool import env_flag


def should_skip_full_pack_messages_for_llm() -> bool:
    """True when ``build_messages`` full pack should not drive KG / triple selection."""
    return env_flag("HAROMA_LLM_CHAT_ONLY", False) or (
        env_flag("HAROMA_LLM_DUMMY_REPLY", False)
        and not env_flag("HAROMA_LLM_DUMMY_FULL_PACK", False)
    )
