"""Packed-context ``generate_chat`` timeout override (seconds) for PersonaAgent.

Mirrors env resolution previously inlined in :class:`agents.persona_agent.PersonaAgent`:
``HAROMA_FAST_LLM_TIMEOUT_SEC``, ``HAROMA_FAST_LLM_DEFAULT_TIMEOUT_SEC``, and optional
``HAROMA_TRUESELF_USER_CHAT_LLM_TIMEOUT_SEC`` (minimum when TrueSelf user /chat cap applies).

When this returns ``None``, :func:`engine.LLMContextReasoner.run_llm_context_reasoning`
uses the global ``HAROMA_LLM_CONTEXT_TIMEOUT_SEC`` (see :func:`mind.prompt_packaging.packed_llm_timeout_seconds`).
"""

from __future__ import annotations

import os
from typing import Optional

from mind.config_env import env_float


def packed_llm_generate_chat_timeout_override(
    *,
    apply_true_self_user_chat_cap: bool,
) -> Optional[float]:
    """Seconds to pass as ``timeout_override`` for packed LLM, or ``None`` for engine default.

    *apply_true_self_user_chat_cap*: True when TrueSelf + user/traced conversant turn
    (``HAROMA_TRUESELF_USER_CHAT_LLM_TIMEOUT_SEC`` is combined with ``min``).
    """
    fast_raw = str(os.environ.get("HAROMA_FAST_LLM_TIMEOUT_SEC", "") or "").strip()
    override: Optional[float] = None
    if fast_raw:
        try:
            override = float(fast_raw)
        except (TypeError, ValueError):
            override = None
    else:
        try:
            fd = env_float("HAROMA_FAST_LLM_DEFAULT_TIMEOUT_SEC", 120.0)
            override = None if fd <= 0.0 else fd
        except Exception:
            override = 120.0

    if apply_true_self_user_chat_cap:
        ts_raw = str(os.environ.get("HAROMA_TRUESELF_USER_CHAT_LLM_TIMEOUT_SEC", "") or "").strip()
        if not ts_raw:
            return override
        try:
            tov_ts = float(ts_raw)
            if tov_ts > 0:
                if override is None:
                    override = tov_ts
                else:
                    override = min(override, tov_ts)
        except (TypeError, ValueError):
            pass

    return override
