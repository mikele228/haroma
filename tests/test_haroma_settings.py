"""mind.haroma_settings — central env reads."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mind.deliberative_cycle_env as dce
import mind.haroma_settings as hs


def test_multi_goal_reexport_matches_haroma_settings():
    assert dce.MultiGoalDeliberativeEnv is hs.MultiGoalDeliberativeEnv
    assert dce.read_multi_goal_deliberative_env is hs.read_multi_goal_deliberative_env


def test_read_multi_goal_defaults(monkeypatch):
    monkeypatch.delenv("HAROMA_MULTI_GOAL_PER_CYCLE", raising=False)
    monkeypatch.setenv("HAROMA_MAX_CYCLE_GOALS", "3")
    monkeypatch.setenv("HAROMA_MAX_ACTIONS_PER_GOAL", "2")
    e = hs.read_multi_goal_deliberative_env()
    assert e.enabled is False
    assert e.max_cycle_goals == 3
    assert e.max_actions_per_goal == 2


def test_controller_packed_llm_flag(monkeypatch):
    monkeypatch.delenv("HAROMA_CONTROLLER_PACKED_LLM", raising=False)
    assert hs.haroma_controller_packed_llm_enabled() is False
    monkeypatch.setenv("HAROMA_CONTROLLER_PACKED_LLM", "1")
    assert hs.haroma_controller_packed_llm_enabled() is True


def test_synthetic_dummy_matches_env_truthy(monkeypatch):
    monkeypatch.setenv("HAROMA_LLM_DUMMY_REPLY", "on")
    assert hs.synthetic_llm_dummy_reply_env() is True
    monkeypatch.setenv("HAROMA_LLM_DUMMY_REPLY", "0")
    assert hs.synthetic_llm_dummy_reply_env() is False
