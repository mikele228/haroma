"""Unit tests for OrganizationalGoalBoard (president / CEO flow)."""

from __future__ import annotations

from unittest.mock import MagicMock

from core.OrganizationalGoalBoard import OrganizationalGoalBoard, _normalize_key


def test_normalize_key_stable():
    assert _normalize_key("  Help User  ") == _normalize_key("help user")


def test_consensus_ratifies():
    gb = OrganizationalGoalBoard(consensus_votes=2, ceo_ticks_to_complete=2)
    shared = MagicMock()
    shared.goal = MagicMock()
    assert not gb.president_try_ratify(shared)
    gb.record_proposal(
        "analyst",
        {"goal_id": "g1", "description": "Ship the feature", "priority": 0.8},
    )
    assert not gb.president_try_ratify(shared)
    gb.record_proposal(
        "primary",
        {"goal_id": "g2", "description": "Ship the feature", "priority": 0.7},
    )
    assert gb.president_try_ratify(shared)
    assert gb.has_active_mandate()
    shared.goal.register_goal.assert_called_once()
    assert gb.tick_ceo_execution(shared, "action_agent") is False
    assert gb.has_active_mandate()
    assert gb.tick_ceo_execution(shared, "action_agent") is True
    assert not gb.has_active_mandate()


def test_single_vote_config():
    gb = OrganizationalGoalBoard(consensus_votes=1, ceo_ticks_to_complete=1)
    shared = MagicMock()
    shared.goal = MagicMock()
    gb.record_proposal(
        "primary",
        {"goal_id": "solo", "description": "One person goal", "priority": 0.6},
    )
    assert gb.president_try_ratify(shared)
    assert gb.has_active_mandate()
    assert gb.tick_ceo_execution(shared, "action_agent") is True
