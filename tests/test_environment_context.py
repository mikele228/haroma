"""Tests for general-agent environment validation and structured actions."""

from unittest.mock import MagicMock

from core.EpisodeContext import EpisodeContext
from mind import environment_context as ec


def test_validate_empty():
    d, err = ec.validate_agent_environment(None)
    assert err is None
    assert d == {}


def test_validate_normalizes():
    d, err = ec.validate_agent_environment(
        {
            "schema_version": 1,
            "domain": "home",
            "entities": {"light.kitchen": {"on": True}},
            "metrics": {"temp_c": 21.0},
            "alerts": ["door_open"],
            "extensions": {"foo": 1},
        }
    )
    assert err is None
    assert d["domain"] == "home"
    assert "fingerprint" in d
    assert d["entities"]["light.kitchen"]["on"] is True


def test_validate_rejects_non_object():
    _, err = ec.validate_agent_environment([])
    assert err == "expected_object"


def test_episode_environment_for_training_metadata():
    ep = EpisodeContext(cycle_id=1)
    assert ep.environment_for_training_metadata() == {}
    ep.bind_agent_environment({"domain": "lab", "entities": {"a": 1}})
    snap = ep.environment_for_training_metadata()
    assert snap["domain"] == "lab"
    assert snap is not ep.agent_environment


def test_propose_actions_respects_env_flag(monkeypatch):
    monkeypatch.setenv("HAROMA_AGENT_STRUCTURED_ACTIONS", "0")
    ep = MagicMock()
    ep.cycle_id = 1
    out = ec.propose_structured_actions(
        episode=ep,
        action={"strategy": "inform", "text": "Hello there"},
        outcome={"score": 0.9},
        agent_environment={"domain": "home", "entities": {"a": 1}},
    )
    assert out == []


def test_propose_actions_notify(monkeypatch):
    monkeypatch.delenv("HAROMA_AGENT_STRUCTURED_ACTIONS", raising=False)
    ep = MagicMock()
    ep.cycle_id = 42
    out = ec.propose_structured_actions(
        episode=ep,
        action={"strategy": "inform", "text": "Status update for you."},
        outcome={"score": 0.8},
        agent_environment={"domain": "custom"},
    )
    assert any(x.get("tool") == "notify.user" for x in out)


def test_environment_summary_includes_robot_bridge_hints():
    s = ec.environment_summary_for_prompt(
        {
            "domain": "lab",
            "entities": {},
            "metrics": {},
            "alerts": [],
            "extensions": {
                "robot_bridge": {
                    "correlation_id": "corr-abc",
                    "results": [
                        {"command_id": "x", "status": "completed"},
                        {"command_id": "y", "status": "failed"},
                    ],
                }
            },
        }
    )
    assert "robot_bridge_results≈2" in s
    assert "robot_bridge_corr=" in s
    assert "robot_bridge_status=" in s
