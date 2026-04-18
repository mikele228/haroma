"""``mind.cognitive_contracts`` barrel matches underlying modules."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mind.chat_visibility as _cv
import mind.cognitive_contracts as cc
import mind.deliberative_cycle_env as _dce
import mind.deliberative_llm_merge as _dm
import mind.packed_llm_discourse as _pld
import mind.packed_llm_action as _pla
import mind.packed_llm_inputs as _pli
import mind.packed_llm_kg as _plk
import mind.packed_llm_agent_state as _plas
import mind.packed_llm_centric as _plcent
import mind.packed_llm_context as _plctx
import mind.packed_llm_cycle_inputs as _plci
import mind.packed_llm_controller_bridge as _plcb
import mind.packed_llm_reasoning_kwargs as _plrk
import mind.packed_llm_reasoning_invoke as _plri
import mind.packed_llm_pipeline_log as _plpl
import mind.packed_llm_paths as _plp
import mind.packed_llm_timeout_override as _pto
import mind.http_chat_timeouts as _hc
import mind.llm_context_timeout as _lt
import mind.prompt_packaging as _pp
import mind.response_text as _rt


def test_barrel_matches_submodules():
    assert cc.normalize_http_chat_response is _cv.normalize_http_chat_response
    assert cc.resolve_chat_visible_text is _cv.resolve_chat_visible_text
    assert cc.LLM_CONTEXT_SOURCES_PREFER_PACKED_ANSWER is _cv.LLM_CONTEXT_SOURCES_PREFER_PACKED_ANSWER
    assert cc.build_llm_centric_action is _pla.build_llm_centric_action
    assert cc.packed_llm_before_llm_log_detail is _plpl.packed_llm_before_llm_log_detail
    assert cc.merge_packed_llm_answer_into_action is _pla.merge_packed_llm_answer_into_action
    assert cc.packed_llm_generate_chat_timeout_override is _pto.packed_llm_generate_chat_timeout_override
    assert cc.should_skip_full_pack_messages_for_llm is _pli.should_skip_full_pack_messages_for_llm
    assert cc.kg_triples_for_packed_llm_prompt is _plk.kg_triples_for_packed_llm_prompt
    assert cc.MultiGoalDeliberativeEnv is _dce.MultiGoalDeliberativeEnv
    assert cc.read_multi_goal_deliberative_env is _dce.read_multi_goal_deliberative_env
    assert cc.chat_llm_primary_env_enabled is _plp.chat_llm_primary_env_enabled
    assert cc.optional_llm_structured_fields is _plctx.optional_llm_structured_fields
    assert cc.llm_centric_enabled_for_persona_cycle is _plcent.llm_centric_enabled_for_persona_cycle
    assert cc.PackedLlmCycleInputs is _plci.PackedLlmCycleInputs
    assert cc.build_packed_llm_cycle_inputs is _plci.build_packed_llm_cycle_inputs
    assert cc.build_persona_packed_llm_reasoning_phase_kwargs is _plrk.build_persona_packed_llm_reasoning_phase_kwargs
    assert cc.invoke_run_llm_context_reasoning_phase is _plri.invoke_run_llm_context_reasoning_phase
    assert cc.summarize_law_value_managers is _plri.summarize_law_value_managers
    assert cc.controller_packed_llm_enabled is _plcb.controller_packed_llm_enabled
    assert cc.run_packed_llm_phase_for_elarion_controller is _plcb.run_packed_llm_phase_for_elarion_controller
    assert cc.build_agent_state_json_for_packed_llm is _plas.build_agent_state_json_for_packed_llm
    assert cc.peek_sensor_queue_for_snapshot is _plas.peek_sensor_queue_for_snapshot
    assert cc.PackedLlmPathState is _plp.PackedLlmPathState
    assert cc.compute_packed_llm_path_state is _plp.compute_packed_llm_path_state
    assert cc.defer_llm_episode_bind is _plp.defer_llm_episode_bind
    assert cc.discourse_context_for_packed_llm is _pld.discourse_context_for_packed_llm
    assert cc.merge_deliberative_into_llm_context is _dm.merge_deliberative_into_llm_context
    assert cc.complete_deferred_deliberative_llm_context is _dm.complete_deferred_deliberative_llm_context
    assert cc.merge_first_encounter_discourse_hint is _pld.merge_first_encounter_discourse_hint
    assert cc.llm_context_timeout_seconds is _lt.llm_context_timeout_seconds
    assert cc.run_llm_context_reasoning is _pp.run_llm_context_reasoning
    assert cc.packed_llm_timeout_seconds is _pp.packed_llm_timeout_seconds


def test_http_chat_timeouts_imports_barrel_timeout():
    assert _hc.llm_context_timeout_seconds is cc.llm_context_timeout_seconds


def test_truncate_reexported_from_response_text():
    assert cc.truncate_chat_at_end_marker is _rt.truncate_chat_at_end_marker


def test_chat_response_constants_reexported():
    assert cc.CHAT_RESPONSE_UNKNOWN is _rt.CHAT_RESPONSE_UNKNOWN
    assert cc.CHAT_RESPONSE_LLM_TIMEOUT is _rt.CHAT_RESPONSE_LLM_TIMEOUT
    assert cc.CHAT_RESPONSE_LLM_ERROR is _rt.CHAT_RESPONSE_LLM_ERROR
    assert cc.CHAT_RESPONSE_LLM_UNPARSEABLE is _rt.CHAT_RESPONSE_LLM_UNPARSEABLE
    assert cc.CHAT_RESPONSE_LLM_NO_REPLY is _rt.CHAT_RESPONSE_LLM_NO_REPLY
    assert cc.CHAT_RESPONSE_LLM_UNAVAILABLE is _rt.CHAT_RESPONSE_LLM_UNAVAILABLE


def test_all_exports_resolve():
    """Guardrail: every ``__all__`` name must be defined (catches typos / drift)."""
    for name in cc.__all__:
        assert hasattr(cc, name), name
        getattr(cc, name)
