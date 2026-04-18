"""Tests for :mod:`mind.packed_llm_agent_state`."""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind.packed_llm_agent_state import (
    build_agent_state_json_for_packed_llm,
    peek_sensor_queue_for_snapshot,
)


def test_empty_json_when_not_deliberative():
    s = build_agent_state_json_for_packed_llm(
        deliberative_flag=False,
        llm_ctx_enabled=True,
        law_summary={},
        val_mgr=None,
        value_summary={},
        state=object(),
        boot_agent_ref=None,
        identity_summary={"x": 1},
        personality_summary={},
        active_goals=[],
        episode=SimpleNamespace(affect={}),
    )
    assert s == ""


def test_empty_json_when_llm_disabled():
    s = build_agent_state_json_for_packed_llm(
        deliberative_flag=True,
        llm_ctx_enabled=False,
        law_summary={},
        val_mgr=None,
        value_summary={},
        state=object(),
        boot_agent_ref=None,
        identity_summary={"x": 1},
        personality_summary={},
        active_goals=[],
        episode=SimpleNamespace(affect={}),
    )
    assert s == ""


def test_serializes_snapshot_when_gates_on():
    raw = build_agent_state_json_for_packed_llm(
        deliberative_flag=True,
        llm_ctx_enabled=True,
        law_summary={"rules": []},
        val_mgr=None,
        value_summary={"v": 1},
        state=object(),
        boot_agent_ref=None,
        identity_summary={"who": "test"},
        personality_summary={"openness": 0.5},
        active_goals=[{"goal_id": "g1", "description": "d", "priority": 0.3}],
        episode=SimpleNamespace(affect={"dominant_emotion": "calm"}, drives=None),
    )
    data = json.loads(raw)
    assert data["identity"]["who"] == "test"
    assert "laws" in data
    assert "values" in data


def test_peek_sensor_queue_via_boot_input_agent():
    class IA:
        def peek_sensor_queue(self, n):
            return [{"id": 1}]

    boot = SimpleNamespace(input_agent=IA())
    state = object()
    assert peek_sensor_queue_for_snapshot(state, boot) == [{"id": 1}]
