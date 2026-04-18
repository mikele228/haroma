"""Tests for :mod:`mind.deliberative_cycle_env`."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind.deliberative_cycle_env import read_multi_goal_deliberative_env


def test_defaults(monkeypatch):
    monkeypatch.delenv("HAROMA_MULTI_GOAL_PER_CYCLE", raising=False)
    monkeypatch.delenv("HAROMA_MAX_CYCLE_GOALS", raising=False)
    monkeypatch.delenv("HAROMA_MAX_ACTIONS_PER_GOAL", raising=False)
    e = read_multi_goal_deliberative_env()
    assert e.enabled is False
    assert e.max_cycle_goals == 3
    assert e.max_actions_per_goal == 2


def test_multi_goal_on(monkeypatch):
    monkeypatch.setenv("HAROMA_MULTI_GOAL_PER_CYCLE", "1")
    e = read_multi_goal_deliberative_env()
    assert e.enabled is True


def test_invalid_ints_fallback(monkeypatch):
    monkeypatch.setenv("HAROMA_MAX_CYCLE_GOALS", "x")
    monkeypatch.setenv("HAROMA_MAX_ACTIONS_PER_GOAL", "y")
    e = read_multi_goal_deliberative_env()
    assert e.max_cycle_goals == 3
    assert e.max_actions_per_goal == 2
