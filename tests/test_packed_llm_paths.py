"""Tests for :mod:`mind.packed_llm_paths`."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind.packed_llm_paths import (
    chat_llm_primary_env_enabled,
    compute_packed_llm_path_state,
    defer_llm_episode_bind,
)


def _base(**overrides):
    kwargs = dict(
        llm_centric=False,
        chat_llm_primary=False,
        role="conversant",
        has_external=True,
        is_internal=False,
        trueself_agent=False,
        user_or_traced_turn=True,
        user_message=True,
        gate_reasoning=True,
        over_budget=False,
        organic=0.0,
        organic_skip_threshold=0.9,
    )
    kwargs.update(overrides)
    return compute_packed_llm_path_state(**kwargs)


def test_llm_ctx_disabled_when_no_user_message():
    s = _base(user_message=False)
    assert s.packed_llm_eligible is False
    assert s.llm_classic_path is False
    assert s.llm_ctx_enabled is False


def test_trueself_requires_user_or_traced_turn():
    s = _base(trueself_agent=True, user_or_traced_turn=False, user_message=True)
    assert s.packed_llm_eligible is False


def test_primary_path_requires_chat_and_conversant():
    s = _base(chat_llm_primary=True, role="conversant", has_external=True)
    assert s.llm_primary_path is True
    assert s.llm_ctx_enabled is True
    s2 = _base(chat_llm_primary=True, role="observer")
    assert s2.llm_primary_path is False


def test_classic_path_blocked_by_organic_threshold():
    s = _base(organic=0.95, organic_skip_threshold=0.9)
    assert s.llm_classic_path is False
    s2 = _base(organic=0.5, organic_skip_threshold=0.9)
    assert s2.llm_classic_path is True


def test_llm_centric_forces_ctx_enabled():
    s = _base(llm_centric=True, user_message=False)
    assert s.llm_ctx_enabled is True


def test_internal_disables_classic_and_conversant():
    s = _base(is_internal=True, llm_centric=False, organic=0.0)
    assert s.conversant_chat is False
    assert s.llm_classic_path is False


def test_defer_llm_episode_bind_requires_both():
    assert defer_llm_episode_bind(True, True) is True
    assert defer_llm_episode_bind(False, True) is False
    assert defer_llm_episode_bind(True, False) is False
    assert defer_llm_episode_bind(False, False) is False


def test_chat_llm_primary_default_on(monkeypatch):
    monkeypatch.delenv("HAROMA_CHAT_LLM_PRIMARY", raising=False)
    assert chat_llm_primary_env_enabled() is True


def test_chat_llm_primary_explicit_off(monkeypatch):
    monkeypatch.setenv("HAROMA_CHAT_LLM_PRIMARY", "0")
    assert chat_llm_primary_env_enabled() is False
