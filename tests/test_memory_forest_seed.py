"""Tests for MemoryForest seed context, post-turn tree touch, and cited-node bump."""

from __future__ import annotations

import json
import os
import time
import pytest
from unittest.mock import MagicMock

from core.Memory import MemoryForest, MemoryNode
from mind.cognitive_contracts import build_messages


# -- fixtures --------------------------------------------------------------

_IDENTITY = {
    "essence_name": "Elarion",
    "vessel": "HaromaX6",
    "current_role": "conversant",
    "current_phase": "active",
}
_PERSONALITY = {"openness": 0.8}
_GOALS = [{"goal_id": "g1", "description": "be helpful", "priority": 0.9}]
_LAW = {"ids": ["dont_lie"]}
_VALUE = {"value_keys": ["honesty"]}


def _make_forest_with_nodes() -> MemoryForest:
    """Create a forest with a few nodes spread across trees."""
    mf = MemoryForest(encoder=None)
    mf.add_node(
        "identity_tree", "main", MemoryNode(content="My name is Elarion", tags=["identity"])
    )
    mf.add_node("belief_tree", "main", MemoryNode(content="I believe in fairness", tags=["belief"]))
    mf.add_node(
        "goal_tree", "main", MemoryNode(content="Goal: learn from conversations", tags=["goal"])
    )
    mf.add_node(
        "thought_tree", "main", MemoryNode(content="User asked about weather", tags=["thought"])
    )
    mf.add_node("action_tree", "main", MemoryNode(content="Replied with forecast", tags=["action"]))
    return mf


# -- build_seed_context ----------------------------------------------------


class TestBuildSeedContext:
    def test_empty_forest_returns_empty(self):
        mf = MemoryForest(encoder=None)
        assert mf.build_seed_context() == ""

    def test_seed_contains_tree_labels(self):
        mf = _make_forest_with_nodes()
        seed = mf.build_seed_context()
        assert "[MEMORY FOREST SEED]" in seed
        assert "identity_tree:" in seed
        assert "belief_tree:" in seed

    def test_recalled_nodes_appear_in_seed(self):
        mf = _make_forest_with_nodes()
        recalled = [
            MemoryNode(content="User is Alice", tree="identity_tree"),
        ]
        seed = mf.build_seed_context(recalled=recalled)
        assert "User is Alice" in seed

    def test_budget_enforced(self):
        mf = _make_forest_with_nodes()
        seed = mf.build_seed_context(max_chars=50)
        body = seed.replace("[MEMORY FOREST SEED]\n", "")
        assert len(body) <= 70  # small slack from line breaks

    def test_zero_budget(self):
        mf = _make_forest_with_nodes()
        assert mf.build_seed_context(max_chars=0) == ""

    def test_query_text_is_accepted(self):
        mf = _make_forest_with_nodes()
        seed = mf.build_seed_context(query_text="weather")
        assert isinstance(seed, str)


# -- build_messages with seed -----------------------------------------------


class TestBuildMessagesWithSeed:
    def test_seed_present_in_user_message(self):
        mf = _make_forest_with_nodes()
        seed = mf.build_seed_context()
        msgs = build_messages(
            user_text="Hello",
            recalled_memories=[],
            identity_summary=_IDENTITY,
            personality_summary=_PERSONALITY,
            active_goals=_GOALS,
            law_summary=_LAW,
            value_summary=_VALUE,
            memory_forest_seed=seed,
        )
        user_content = msgs[1]["content"]
        assert "[MEMORY FOREST SEED]" in user_content
        assert "identity_tree:" in user_content

    def test_no_seed_when_empty(self):
        msgs = build_messages(
            user_text="Hello",
            recalled_memories=[],
            identity_summary=_IDENTITY,
            personality_summary=_PERSONALITY,
            active_goals=[],
            law_summary={},
            value_summary={},
            memory_forest_seed="",
        )
        assert "MEMORY FOREST SEED" not in msgs[1]["content"]

    def test_system_rules_mention_seed(self):
        msgs = build_messages(
            user_text="test",
            recalled_memories=[],
            identity_summary=_IDENTITY,
            personality_summary={},
            active_goals=[],
            law_summary={},
            value_summary={},
        )
        assert "MEMORY FOREST SEED" in msgs[0]["content"]


# -- touch_trees_after_turn ------------------------------------------------


class TestTouchTrees:
    def test_recalled_plus_core_policy(self):
        mf = _make_forest_with_nodes()
        before = mf.count_nodes()
        n = mf.touch_trees_after_turn(
            cycle_id=1,
            branch_name="test",
            summary="test turn",
            recalled_tree_names={"belief_tree"},
            policy="recalled_plus_core",
        )
        assert n == 3  # belief + thought + action
        assert mf.count_nodes() == before + 3

    def test_all_policy(self):
        mf = _make_forest_with_nodes()
        n = mf.touch_trees_after_turn(
            cycle_id=2,
            branch_name="test",
            summary="test all",
            policy="all",
        )
        assert n == len(mf.trees)

    def test_recalled_only_policy(self):
        mf = _make_forest_with_nodes()
        n = mf.touch_trees_after_turn(
            cycle_id=3,
            branch_name="test",
            summary="recalled only",
            recalled_tree_names={"goal_tree"},
            policy="recalled_only",
        )
        assert n == 1

    def test_node_content_contains_cycle(self):
        mf = _make_forest_with_nodes()
        mf.touch_trees_after_turn(
            cycle_id=42,
            branch_name="test",
            summary="hello world",
            policy="all",
        )
        nodes = mf.get_nodes("identity_tree", "test")
        assert any("[turn:42]" in n.content for n in nodes)


# -- bump_cited_nodes ------------------------------------------------------


class TestBumpCitedNodes:
    def test_bump_increases_confidence(self):
        mf = MemoryForest(encoder=None)
        node = MemoryNode(content="fact A", confidence=0.5)
        mf.add_node("thought_tree", "main", node)
        mid = node.moment_id
        bumped = mf.bump_cited_nodes([mid])
        assert bumped == 1
        assert node.confidence == pytest.approx(0.55)

    def test_bump_caps_at_one(self):
        mf = MemoryForest(encoder=None)
        node = MemoryNode(content="fact B", confidence=0.99)
        mf.add_node("thought_tree", "main", node)
        mf.bump_cited_nodes([node.moment_id], boost=0.1)
        assert node.confidence == 1.0

    def test_bump_unknown_id(self):
        mf = MemoryForest(encoder=None)
        assert mf.bump_cited_nodes(["nonexistent"]) == 0
