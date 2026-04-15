"""Hierarchical goals: children and action items must be satisfied before a parent completes."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.Goal import GoalEngine
from unittest.mock import patch


def _engine() -> GoalEngine:
    with patch.object(GoalEngine, "load_state", lambda self: setattr(self, "state", {})):
        return GoalEngine()


def test_parent_blocked_until_children_and_actions():
    eng = _engine()
    eng.register_goal("root", "ship feature", priority=0.9, child_goal_ids=["a", "b"])
    eng.register_goal("a", "sub a", priority=0.8)
    eng.register_goal("b", "sub b", priority=0.8)
    assert not eng.complete_goal("root")
    assert eng.complete_goal("a")
    assert not eng.complete_goal("root")
    assert eng.complete_goal("b")
    # Last child completion auto-completes the parent when no pending actions remain.
    assert eng.goals["root"]["completed"]
    assert not eng.complete_goal("root"), "root already finalized"


def test_action_items_gate_completion():
    eng = _engine()
    eng.register_goal(
        "g1",
        "one task",
        priority=0.5,
        action_items=[{"id": "x", "description": "step", "done": False}],
    )
    assert not eng.complete_goal("g1")
    assert eng.mark_action_done("g1", "x")
    assert eng.goals["g1"]["completed"]


def test_prioritize_workfront_skips_blocked_parent():
    eng = _engine()
    eng.register_input_goal("parent", "p", child_goal_ids=["c"])
    eng.register_input_goal("c", "child")
    wf = eng.prioritize_workfront()
    assert wf[0] == "c"
    assert "parent" not in wf


def test_current_input_goal_skips_blocked_head():
    eng = _engine()
    eng.register_input_goal("parent", "p", child_goal_ids=["c"])
    eng.register_input_goal("c", "child")
    assert eng.current_input_goal() == "c"


def test_autocomplete_parent_when_last_child_finishes():
    eng = _engine()
    eng.register_goal("root", "r", child_goal_ids=["only"])
    eng.register_goal("only", "leaf", priority=0.5)
    assert eng.complete_goal("only")
    assert eng.goals["root"].get("completed") is True


def test_record_mission_completed_triggers_complete_goal():
    eng = _engine()
    eng.register_goal("m1", "mission", priority=0.5)
    eng.record_mission("agent1", "m1", "completed")
    assert eng.goals["m1"]["completed"]
