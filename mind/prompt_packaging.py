"""Stable import surface for packed-context LLM (``engine.LLMContextReasoner``).

Runtime code outside ``engine/`` should import prompt orchestration from here so
call sites stay stable if the engine module is split or wrapped. Integration
tests should also prefer this module over importing ``engine`` directly unless
they need private helpers (``_*``) from the engine file.

Re-exports only — behavior lives in ``engine/LLMContextReasoner.py``. For a
single combined import surface (chat shaping, timeouts, deliberative merge), see
:mod:`mind.cognitive_contracts`.
"""

from __future__ import annotations

from engine.LLMContextReasoner import (
    LLMContextResult,
    _llm_context_timeout_seconds,
    build_messages,
    packed_messages_stats,
    parse_response,
    run_llm_context_reasoning,
)

# Public alias — engine helper is underscore-prefixed; mind code should use this name.
packed_llm_timeout_seconds = _llm_context_timeout_seconds

__all__ = [
    "LLMContextResult",
    "build_messages",
    "packed_llm_timeout_seconds",
    "packed_messages_stats",
    "parse_response",
    "run_llm_context_reasoning",
]
