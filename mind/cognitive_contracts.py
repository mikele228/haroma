"""Curated import surface for chat shaping, deliberative LLM merge, and packed prompts.

Submodules remain the source of truth; this module is a convenience barrel so
callers can depend on one place for HTTP-visible text and LLM orchestration
without hunting across ``mind.chat_visibility``, ``mind.deliberative_llm_merge``,
``mind.llm_context_timeout``, and ``mind.prompt_packaging``. Low-level
``truncate_chat_at_end_marker`` and the ``CHAT_RESPONSE_*`` string constants from
``mind.response_text`` are re-exported for scripts and tests. Integration tests
should import packed-LLM symbols from here unless they assert identity with
``engine.LLMContextReasoner`` (see ``tests/test_prompt_packaging.py``).
"""

from __future__ import annotations

from mind.chat_visibility import (
    normalize_http_chat_response,
    resolve_chat_visible_text,
)
from mind.deliberative_llm_merge import merge_deliberative_into_llm_context
from mind.llm_context_timeout import llm_context_timeout_seconds
from mind.prompt_packaging import (
    LLMContextResult,
    build_messages,
    packed_llm_timeout_seconds,
    packed_messages_stats,
    parse_response,
    run_llm_context_reasoning,
)
from mind.response_text import (
    CHAT_RESPONSE_LLM_ERROR,
    CHAT_RESPONSE_LLM_NO_REPLY,
    CHAT_RESPONSE_LLM_TIMEOUT,
    CHAT_RESPONSE_LLM_UNAVAILABLE,
    CHAT_RESPONSE_LLM_UNPARSEABLE,
    CHAT_RESPONSE_UNKNOWN,
    truncate_chat_at_end_marker,
)

__all__ = [
    "CHAT_RESPONSE_LLM_ERROR",
    "CHAT_RESPONSE_LLM_NO_REPLY",
    "CHAT_RESPONSE_LLM_TIMEOUT",
    "CHAT_RESPONSE_LLM_UNAVAILABLE",
    "CHAT_RESPONSE_LLM_UNPARSEABLE",
    "CHAT_RESPONSE_UNKNOWN",
    "LLMContextResult",
    "build_messages",
    "llm_context_timeout_seconds",
    "merge_deliberative_into_llm_context",
    "normalize_http_chat_response",
    "packed_llm_timeout_seconds",
    "packed_messages_stats",
    "parse_response",
    "resolve_chat_visible_text",
    "run_llm_context_reasoning",
    "truncate_chat_at_end_marker",
]
