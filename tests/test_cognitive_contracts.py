"""``mind.cognitive_contracts`` barrel matches underlying modules."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mind.chat_visibility as _cv
import mind.cognitive_contracts as cc
import mind.deliberative_llm_merge as _dm
import mind.http_chat_timeouts as _hc
import mind.llm_context_timeout as _lt
import mind.prompt_packaging as _pp
import mind.response_text as _rt


def test_barrel_matches_submodules():
    assert cc.normalize_http_chat_response is _cv.normalize_http_chat_response
    assert cc.resolve_chat_visible_text is _cv.resolve_chat_visible_text
    assert cc.merge_deliberative_into_llm_context is _dm.merge_deliberative_into_llm_context
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
