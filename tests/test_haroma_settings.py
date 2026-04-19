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


def test_haroma_memory_recall_intensity_clamped(monkeypatch):
    monkeypatch.delenv("HAROMA_MEMORY_RECALL_INTENSITY", raising=False)
    assert hs.haroma_memory_recall_intensity() == 10
    monkeypatch.setenv("HAROMA_MEMORY_RECALL_INTENSITY", "0")
    assert hs.haroma_memory_recall_intensity() == 0
    monkeypatch.setenv("HAROMA_MEMORY_RECALL_INTENSITY", "5")
    assert hs.haroma_memory_recall_intensity() == 5
    monkeypatch.setenv("HAROMA_MEMORY_RECALL_INTENSITY", "99")
    assert hs.haroma_memory_recall_intensity() == 10
    monkeypatch.setenv("HAROMA_MEMORY_RECALL_INTENSITY", "bad")
    assert hs.haroma_memory_recall_intensity() == 10


def test_haroma_cmem_defaults(monkeypatch):
    monkeypatch.delenv("HAROMA_RECALL_CMEM_ONLY", raising=False)
    monkeypatch.delenv("HAROMA_CMEM_BUILD_ENABLED", raising=False)
    monkeypatch.delenv("HAROMA_CMEM_RECALL_MAX_PROBE", raising=False)
    monkeypatch.delenv("HAROMA_CMEM_MERGE_PRIME", raising=False)
    monkeypatch.delenv("HAROMA_CMEM_MAX_TOTAL_NODES", raising=False)
    monkeypatch.delenv("HAROMA_CMEM_BOOTSTRAP_MAX_NODES", raising=False)
    monkeypatch.delenv("HAROMA_CMEM_RECALL_FALLBACK_FOREST", raising=False)
    assert hs.haroma_recall_cmem_only() is False
    assert hs.haroma_cmem_recall_fallback_forest() is False
    assert hs.haroma_cmem_build_enabled() is True
    assert hs.haroma_cmem_recall_max_probe() == 2000
    assert hs.haroma_cmem_merge_prime() is False
    assert hs.haroma_cmem_max_total_nodes() == 0
    assert hs.haroma_cmem_bootstrap_max_nodes() == 128
    monkeypatch.setenv("HAROMA_RECALL_CMEM_ONLY", "1")
    assert hs.haroma_recall_cmem_only() is True


def test_haroma_dialogue_eval_log_env(monkeypatch):
    monkeypatch.delenv("HAROMA_DIALOGUE_EVAL_LOG", raising=False)
    assert hs.haroma_dialogue_eval_log_enabled() is False
    monkeypatch.setenv("HAROMA_DIALOGUE_EVAL_LOG", "1")
    assert hs.haroma_dialogue_eval_log_enabled() is True


def test_haroma_dialogue_phase_clamped(monkeypatch):
    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "2")
    assert hs.haroma_dialogue_phase() == 2
    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "6")
    assert hs.haroma_dialogue_phase() == 6
    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "7")
    assert hs.haroma_dialogue_phase() == 7
    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "8")
    assert hs.haroma_dialogue_phase() == 8
    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "9")
    assert hs.haroma_dialogue_phase() == 9
    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "99")
    assert hs.haroma_dialogue_phase() == 9
    monkeypatch.setenv("HAROMA_DIALOGUE_PHASE", "bad")
    assert hs.haroma_dialogue_phase() == 1
