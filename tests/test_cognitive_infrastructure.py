"""Tests for outcome spine, environment contract, benchmark harness."""

from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO)

from mind.cognitive_trace import (
    apply_ablation_overrides,
    build_canonical_outcome,
    build_planner_arbitration,
    parse_ablation_tags,
    reconciliation_ablated,
)
from mind.environment_contract import normalize_environment_observation


def test_normalize_environment_observation_empty():
    n = normalize_environment_observation(None)
    assert n["reward"] == 0.0
    assert n["done"] is False
    assert n["world"] == {}


def test_build_canonical_outcome():
    co = build_canonical_outcome(
        episode_id="e1",
        cycle_id=3,
        role="observer",
        outcome={"score": 0.72, "breakdown": {"a": 1}},
        gate_decisions={"x": True, "y": False},
        steps_run=5,
        steps_total=7,
        action_strategy="reflect",
        action_type="respond",
    )
    assert co["schema"] == "haroma.canonical_outcome.v1"
    assert co["score"] == 0.72
    assert "a" in co["breakdown_keys"]


def test_ablation_overrides():
    os.environ["ELARION_ABLATION"] = "curiosity, reasoning "
    try:
        assert "curiosity" in parse_ablation_tags()
        gd = {"curiosity": True, "reasoning": True, "dream_consolidation": True}
        out = apply_ablation_overrides(gd)
        assert out["curiosity"] is False
        assert out["reasoning"] is False
        assert out["dream_consolidation"] is True
        assert reconciliation_ablated() is False
    finally:
        os.environ.pop("ELARION_ABLATION", None)


def test_reconciliation_ablated_flag():
    os.environ["ELARION_ABLATION"] = "reconciliation"
    try:
        assert reconciliation_ablated() is True
    finally:
        os.environ.pop("ELARION_ABLATION", None)


def test_planner_arbitration_prefers_plan():
    ar = build_planner_arbitration(
        memory_hint="reflect",
        imagined_strategy="inquire",
        current_plan=["observe", "inform"],
        plan_step=0,
        resolved_hint="observe",
    )
    assert ar["chosen_source"] == "imagined_plan"
    assert ar["memory_disagrees_imagination"] is True


def test_text_env_frozen_paths_smoke():
    from environment.TextEnvironment import TextEnvironment

    env = TextEnvironment()
    env.execute_action("explore", "north")
    assert env.stats()["player_location"] == "library"
    env.reset()
    env.execute_action("explore", "south")
    env.execute_action("explore", "south")
    assert env.stats()["player_location"] == "deep_cave"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
