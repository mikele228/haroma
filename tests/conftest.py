"""Shared fixtures for HaromaX6 integration tests."""

from __future__ import annotations

import time

import pytest

from agents.boot_agent import BootAgent


@pytest.fixture(autouse=True)
def _isolate_lab_and_rate_limit_state():
    """Reset global lab event ring and HTTP rate-limit buckets between tests."""
    from mind.http_rate_limit import clear_rate_limit_state_for_tests
    from mind.lab_research import clear_lab_events_for_tests

    clear_lab_events_for_tests()
    clear_rate_limit_state_for_tests()
    yield


@pytest.fixture(scope="session")
def boot() -> BootAgent:
    """Boot multi-agent stack once, start all agents, tear down after the session."""
    from tests.test_multi_agent_chat import _patch_delegation_logging

    b = BootAgent()
    shared = b.boot()
    assert shared is not None, "SharedResources is None"
    assert b.trueself_agent is not None, "TrueSelf not spawned"
    assert len(b.persona_agents) == 2, f"Expected 2 personas, got {len(b.persona_agents)}"

    b.input_agent.set_boot_agent(b)
    b.trueself_agent.set_boot_agent(b)
    for p in b.persona_agents:
        p.set_boot_agent(b)

    _patch_delegation_logging(b)
    b.start_all()
    time.sleep(1.0)

    agents_to_check = [
        ("boot", b),
        ("input", b.input_agent),
        ("trueself", b.trueself_agent),
        ("background", b.background_agent),
    ] + [(f"persona:{p.agent_id}", p) for p in b.persona_agents]
    for label, agent in agents_to_check:
        assert agent.is_alive(), f"{label} is not alive after start_all"

    yield b

    b.save_and_shutdown()
