"""Curated import surface for chat shaping, deliberative LLM merge, and packed prompts.

**Architecture:** Which entry points run packed-context ``generate_chat`` is documented in
:mod:`mind.cognitive_entrypoints`. Prefer importing from this module for chat / packed-LLM
behavior in application code; use submodule imports in tests when asserting identity.

Submodules remain the source of truth; this module is a convenience barrel so
callers can depend on one place for HTTP-visible text and LLM orchestration
without hunting across ``mind.chat_visibility``, ``mind.deliberative_llm_merge``,
``mind.packed_llm_action``, ``mind.packed_llm_inputs``, ``mind.packed_llm_kg``, ``mind.packed_llm_timeout_override``, ``mind.packed_llm_paths``, ``mind.packed_llm_agent_state``, ``mind.packed_llm_discourse``, ``mind.packed_llm_cycle_inputs``, ``mind.packed_llm_centric``, ``mind.packed_llm_controller_bridge``, ``mind.packed_llm_dummy_env``, ``mind.packed_llm_pipeline_log``, ``mind.packed_llm_context``, ``mind.packed_llm_reasoning_kwargs``, ``mind.packed_llm_reasoning_invoke``,
``mind.haroma_settings``,
``mind.llm_context_timeout``, ``mind.deliberative_cycle_env``, and ``mind.prompt_packaging``. Low-level
``truncate_chat_at_end_marker`` and the ``CHAT_RESPONSE_*`` string constants from
``mind.response_text`` are re-exported for scripts and tests. Integration tests
should import packed-LLM symbols from here unless they assert identity with
``engine.LLMContextReasoner`` (see ``tests/test_prompt_packaging.py``).
"""

from __future__ import annotations

from mind.deliberative_cycle_env import MultiGoalDeliberativeEnv, read_multi_goal_deliberative_env
from mind.chat_visibility import (
    LLM_CONTEXT_SOURCES_PREFER_PACKED_ANSWER,
    normalize_http_chat_response,
    resolve_chat_visible_text,
)
from mind.deliberative_llm_merge import (
    complete_deferred_deliberative_llm_context,
    merge_deliberative_into_llm_context,
)
from mind.packed_llm_action import build_llm_centric_action, merge_packed_llm_answer_into_action
from mind.packed_llm_pipeline_log import packed_llm_before_llm_log_detail
from mind.packed_llm_inputs import should_skip_full_pack_messages_for_llm
from mind.packed_llm_kg import kg_triples_for_packed_llm_prompt
from mind.packed_llm_agent_state import (
    build_agent_state_json_for_packed_llm,
    peek_sensor_queue_for_snapshot,
)
from mind.packed_llm_centric import llm_centric_enabled_for_persona_cycle
from mind.packed_llm_context import optional_llm_structured_fields
from mind.packed_llm_cycle_inputs import PackedLlmCycleInputs, build_packed_llm_cycle_inputs
from mind.packed_llm_reasoning_kwargs import build_persona_packed_llm_reasoning_phase_kwargs
from mind.packed_llm_reasoning_invoke import (
    invoke_run_llm_context_reasoning_phase,
    summarize_law_value_managers,
)
from mind.packed_llm_controller_bridge import (
    controller_packed_llm_enabled,
    run_packed_llm_phase_for_elarion_controller,
)
from mind.packed_llm_discourse import (
    discourse_context_for_packed_llm,
    merge_first_encounter_discourse_hint,
)
from mind.packed_llm_paths import (
    PackedLlmPathState,
    chat_llm_primary_env_enabled,
    compute_packed_llm_path_state,
    defer_llm_episode_bind,
)
from mind.packed_llm_timeout_override import packed_llm_generate_chat_timeout_override
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
    "MultiGoalDeliberativeEnv",
    "read_multi_goal_deliberative_env",
    "chat_llm_primary_env_enabled",
    "build_llm_centric_action",
    "packed_llm_before_llm_log_detail",
    "llm_centric_enabled_for_persona_cycle",
    "PackedLlmCycleInputs",
    "build_packed_llm_cycle_inputs",
    "build_persona_packed_llm_reasoning_phase_kwargs",
    "invoke_run_llm_context_reasoning_phase",
    "summarize_law_value_managers",
    "controller_packed_llm_enabled",
    "run_packed_llm_phase_for_elarion_controller",
    "optional_llm_structured_fields",
    "complete_deferred_deliberative_llm_context",
    "discourse_context_for_packed_llm",
    "merge_first_encounter_discourse_hint",
    "defer_llm_episode_bind",
    "build_agent_state_json_for_packed_llm",
    "peek_sensor_queue_for_snapshot",
    "PackedLlmPathState",
    "compute_packed_llm_path_state",
    "LLM_CONTEXT_SOURCES_PREFER_PACKED_ANSWER",
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
    "merge_packed_llm_answer_into_action",
    "kg_triples_for_packed_llm_prompt",
    "packed_llm_generate_chat_timeout_override",
    "should_skip_full_pack_messages_for_llm",
    "normalize_http_chat_response",
    "packed_llm_timeout_seconds",
    "packed_messages_stats",
    "parse_response",
    "resolve_chat_visible_text",
    "run_llm_context_reasoning",
    "truncate_chat_at_end_marker",
]
