"""Input gating for expensive packed-LLM prompt build (KG, full messages).

When ``HAROMA_LLM_CHAT_ONLY`` or the synthetic path uses a one-message placeholder,
:mod:`engine.LLMContextReasoner` skips ``build_messages`` — skip KG selection here too.

``HAROMA_LLM_DUMMY_*`` toggles are **dev/profiling hooks** for
:class:`~engine.LLMContextReasoner` only (see module docstring there). They are not
Elarion's primary latency mechanism; real ``/chat`` tuning lives in
:class:`~agents.persona_agent.PersonaAgent` (HTTP trace + recall, unphased delegation,
budgets). With ``HAROMA_LLM_DUMMY_REPLY``, the default is **full** ``build_messages``
(same work as production minus decode). Minimal placeholder: ``HAROMA_LLM_DUMMY_FAST_PACK=1``
or ``HAROMA_LLM_DUMMY_FULL_PACK=0``.
"""

from __future__ import annotations

import os

from utils.coerce_bool import env_flag


def synthetic_uses_placeholder_prompt(is_dummy_env: bool) -> bool:
    """True when ``run_llm_context_reasoning`` should use a tiny user-only message list.

    * *dummy_env* (``HAROMA_LLM_DUMMY_REPLY``): full pack by default; placeholder only
      if ``HAROMA_LLM_DUMMY_FAST_PACK`` or explicit ``HAROMA_LLM_DUMMY_FULL_PACK=0``.
    * No backend but dummy env unset: unchanged — full pack only if
      ``HAROMA_LLM_DUMMY_FULL_PACK`` is truthy (legacy probe behavior).
    """
    if is_dummy_env:
        if env_flag("HAROMA_LLM_DUMMY_FAST_PACK", False):
            return True
        raw = os.environ.get("HAROMA_LLM_DUMMY_FULL_PACK")
        if raw is None or not str(raw).strip():
            return False
        return not env_flag("HAROMA_LLM_DUMMY_FULL_PACK", False)
    return not env_flag("HAROMA_LLM_DUMMY_FULL_PACK", False)


def should_skip_full_pack_messages_for_llm() -> bool:
    """True when ``build_messages`` full pack should not drive KG / triple selection."""
    if env_flag("HAROMA_LLM_CHAT_ONLY", False):
        return True
    if not env_flag("HAROMA_LLM_DUMMY_REPLY", False):
        return False
    return synthetic_uses_placeholder_prompt(True)
