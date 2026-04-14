"""Multi-goal / multi-action deliberation per cycle."""

import os
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.ActionLoop import ActionGenerator, ActionCandidate, _LAW_SAFE_STRATEGIES
from mind.cycle_flow import run_multi_goal_deliberative_actions


def test_sorted_candidate_pool_law_filter():
    ag = ActionGenerator(composer=None)
    c1 = ActionCandidate("inform", ["hello"])
    c1.total_score = 0.9
    c2 = ActionCandidate("reflect", ["think"])
    c2.total_score = 0.5
    ctx = {"symbolic_law": {"violations": [{"x": 1}]}}
    pool = ag._sorted_candidate_pool([c1, c2], ctx)
    assert all(c.strategy in _LAW_SAFE_STRATEGIES for c in pool)


def test_generate_multi_actions_distinct_strategies():
    ag = ActionGenerator(composer=None)
    ep: Dict[str, Any] = {
        "affect": {"dominant_emotion": "neutral", "intensity": 0.3, "valence": 0.0, "arousal": 0.0},
        "active_goals": [{"goal_id": "g1", "description": "test"}],
        "curiosity": {},
        "narrative_context": "",
        "identity": {},
        "drives": {},
        "dominant_drive": "",
        "perception": {"content": "hi", "tags": []},
        "recalled_memories": [],
        "symbolic_law": {"violations": [], "compliant": True},
    }
    acts = ag.generate_multi_actions(
        ep,
        [],
        strategy_hint=None,
        max_actions=3,
        is_in_conversation=True,
        utterance_style="conversational",
    )
    assert isinstance(acts, list)
    assert len(acts) >= 1
    strats = [a.get("strategy") for a in acts]
    assert len(strats) == len(set(strats))


def test_run_multi_goal_fuses_text():
    ag = ActionGenerator(composer=None)
    ep: Dict[str, Any] = {
        "affect": {"dominant_emotion": "neutral", "intensity": 0.3, "valence": 0.0, "arousal": 0.0},
        "active_goals": [],
        "curiosity": {},
        "narrative_context": "",
        "identity": {},
        "drives": {},
        "dominant_drive": "",
        "perception": {"content": "hi", "tags": []},
        "recalled_memories": [],
        "symbolic_law": {"violations": [], "compliant": True},
    }
    batch = [
        {"goal_id": "a", "description": "one", "priority": 0.5, "source": "input"},
        {"goal_id": "b", "description": "two", "priority": 0.4, "source": "input"},
    ]
    ep2 = dict(ep)
    ep2["active_goals"] = batch
    mock_ep = MagicMock()
    mock_ep.trace = MagicMock()
    fused, groups = run_multi_goal_deliberative_actions(
        episode=mock_ep,
        action_generator=ag,
        ep_payload=ep2,
        goal_batch=batch,
        max_actions_per_goal=1,
        ws_dicts=[],
        strategy_hint=None,
        working_memory_context="",
        conversation_context="",
        is_in_conversation=True,
        topic="",
        last_input_content="hello",
        topic_shifted=False,
        knowledge_summary={},
        reasoning_result={},
        nlu_result={},
        interlocutor={},
        counterfactual_result={},
        novelty_bias=0.0,
        personality={"openness": 0.5},
        utterance_style="conversational",
        trace_pre_action=False,
    )
    assert fused.get("strategy") == "multi_goal"
    assert len(groups) == 2
    assert all("actions" in g for g in groups)
    tx = (fused.get("text") or "").strip()
    assert tx  # non-empty fused reply when goals produce text


def test_run_multi_goal_empty_batch_falls_back():
    ag = ActionGenerator(composer=None)
    ep: Dict[str, Any] = {
        "affect": {"dominant_emotion": "neutral", "intensity": 0.2, "valence": 0.0, "arousal": 0.0},
        "active_goals": [{"goal_id": "x", "description": "d"}],
        "curiosity": {},
        "narrative_context": "",
        "identity": {},
        "drives": {},
        "dominant_drive": "",
        "perception": {"content": "x", "tags": []},
        "recalled_memories": [],
        "symbolic_law": {"violations": [], "compliant": True},
    }
    mock_ep = MagicMock()
    act, groups = run_multi_goal_deliberative_actions(
        episode=mock_ep,
        action_generator=ag,
        ep_payload=ep,
        goal_batch=[],
        max_actions_per_goal=1,
        ws_dicts=[],
        strategy_hint=None,
        working_memory_context="",
        conversation_context="",
        is_in_conversation=True,
        topic="",
        last_input_content="",
        topic_shifted=False,
        knowledge_summary={},
        reasoning_result={},
        nlu_result={},
        interlocutor={},
        counterfactual_result={},
        novelty_bias=0.0,
        utterance_style="conversational",
        trace_pre_action=False,
    )
    assert groups == []
    assert "strategy" in act
