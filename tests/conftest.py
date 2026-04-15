"""Shared fixtures for HaromaX6 integration tests."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests._import_guard import prepare_test_imports

# Before any agent/core imports: avoid sentence_transformers→torch during collection
# (Windows hosts may crash loading torch DLLs even when SBERT is unused).
prepare_test_imports(__file__)

_ROOT = Path(__file__).resolve().parents[1]
_ESSENCE = _ROOT / "soul" / "essence.json"


@pytest.fixture(scope="session", autouse=True)
def _ensure_soul_json():
    """Fresh clones omit tracked soul/*.json; generate stock soul before boot tests."""
    if _ESSENCE.is_file():
        return
    gen = _ROOT / "scripts" / "generate_soul.py"
    if not gen.is_file():
        return
    subprocess.run(
        [sys.executable, str(gen), "--defaults"],
        cwd=str(_ROOT),
        check=True,
    )


if TYPE_CHECKING:
    from agents.boot_agent import BootAgent


@pytest.fixture(autouse=True)
def _isolate_lab_and_rate_limit_state():
    """Reset global lab event ring and HTTP rate-limit buckets between tests."""
    from mind.client_ip import clear_trusted_proxy_cache_for_tests
    from mind.http_rate_limit import clear_rate_limit_state_for_tests
    from mind.lab_research import clear_lab_events_for_tests

    clear_lab_events_for_tests()
    clear_rate_limit_state_for_tests()
    clear_trusted_proxy_cache_for_tests()
    yield


@pytest.fixture(scope="session")
def boot() -> "BootAgent":
    """Boot multi-agent stack once, start all agents, tear down after the session."""
    from agents.boot_agent import BootAgent
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
