"""Derivation-first episode spine: merge, payload, and action integration."""

import os
import sys
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.EpisodeContext import EpisodeContext
from core.derivation_merge import merge_derivation_artifacts
from core.ActionLoop import ActionGenerator


# -- merge_derivation_artifacts ------------------------------------------


def _make_episode(**overrides) -> EpisodeContext:
    ep = EpisodeContext(cycle_id=1)
    for k, v in overrides.items():
        setattr(ep, k, v)
    return ep


def test_merge_empty_episode_produces_empty_proposals():
    ep = _make_episode()
    merge_derivation_artifacts(ep)
    assert ep.derivation["proposals"] == []
    assert ep.derivation["summary"] == ""


def test_merge_skips_non_dict_inference():
    ep = _make_episode(
        reasoning={
            "inferences": [
                "bad",
                {"subject": "A", "predicate": "b", "object": "C", "confidence": 0.5},
            ]
        }
    )
    merge_derivation_artifacts(ep)
    assert len(ep.derivation["proposals"]) == 1


def test_merge_skips_empty_triple_inference():
    ep = _make_episode(
        reasoning={
            "inferences": [
                {"subject": "", "predicate": "", "object": "", "confidence": 0.99},
            ],
        }
    )
    merge_derivation_artifacts(ep)
    assert ep.derivation["proposals"] == []


def test_merge_reasoning_inferences():
    ep = _make_episode(
        reasoning={
            "inferences": [
                {
                    "subject_name": "Alice",
                    "predicate": "knows",
                    "object_name": "Bob",
                    "confidence": 0.7,
                },
                {"subject": "X", "predicate": "causes", "object": "Y", "confidence": 0.3},
            ],
            "reasoning_depth": 2,
        }
    )
    merge_derivation_artifacts(ep)
    props = ep.derivation["proposals"]
    assert len(props) == 2
    assert props[0]["kind"] == "kg_inference"
    assert props[0]["source"] == "reasoning"
    assert props[0]["confidence"] == 0.7
    assert props[0]["payload"]["subject"] == "Alice"
    assert props[1]["confidence"] == 0.3


def test_merge_llm_context_inferences_and_env_updates():
    ep = _make_episode(
        llm_context={
            "answer": "Paris is the capital.",
            "confidence": 0.8,
            "requires_confirmation": False,
            "inferences": [
                {
                    "subject": "Paris",
                    "predicate": "capital_of",
                    "object": "France",
                    "confidence": 0.85,
                },
            ],
            "env_updates": {"location": "France"},
        }
    )
    merge_derivation_artifacts(ep)
    props = ep.derivation["proposals"]
    kinds = [p["kind"] for p in props]
    assert "kg_inference" in kinds
    assert "env_update" in kinds
    assert "memory_note" in kinds
    note = next(p for p in props if p["kind"] == "memory_note")
    assert "Paris" in note["payload"]["text"]


def test_merge_llm_answer_skipped_when_low_confidence():
    ep = _make_episode(
        llm_context={
            "answer": "maybe X",
            "confidence": 0.2,
            "requires_confirmation": True,
            "inferences": [],
        }
    )
    merge_derivation_artifacts(ep)
    kinds = [p["kind"] for p in ep.derivation["proposals"]]
    assert "memory_note" not in kinds


def test_merge_synthesized_goals():
    ep = _make_episode()
    goals = [
        {
            "goal_id": "g1",
            "description": "learn python",
            "priority": 0.9,
            "source": "goal_synthesis",
        },
        {"goal_id": "g2", "description": "rest", "priority": 0.4, "source": "drive"},
    ]
    merge_derivation_artifacts(ep, synthesized_goals=goals)
    props = ep.derivation["proposals"]
    assert len(props) == 2
    assert all(p["kind"] == "goal" for p in props)
    assert props[0]["confidence"] == 0.9


def test_merge_combined_sorted_by_confidence():
    ep = _make_episode(
        reasoning={
            "inferences": [
                {"subject": "A", "predicate": "rel", "object": "B", "confidence": 0.5},
            ],
        },
        llm_context={
            "answer": None,
            "confidence": 0.0,
            "inferences": [
                {"subject": "C", "predicate": "rel", "object": "D", "confidence": 0.9},
            ],
        },
    )
    goals = [{"goal_id": "g", "description": "x", "priority": 0.7, "source": "goal_synthesis"}]
    merge_derivation_artifacts(ep, synthesized_goals=goals)
    confs = [p["confidence"] for p in ep.derivation["proposals"]]
    assert confs == sorted(confs, reverse=True)


def test_merge_summary_nonempty_when_proposals():
    ep = _make_episode(
        reasoning={
            "inferences": [
                {"subject": "A", "predicate": "is", "object": "B", "confidence": 0.6},
            ],
        }
    )
    merge_derivation_artifacts(ep)
    assert ep.derivation["summary"]


# -- EpisodeContext wiring -----------------------------------------------


def test_derivation_in_to_payload():
    ep = EpisodeContext(cycle_id=5)
    ep.bind_derivation({"proposals": [{"kind": "goal"}], "summary": "test"})
    payload = ep.to_payload()
    assert "derivation" in payload
    assert payload["derivation"]["summary"] == "test"


def test_derivation_default_in_payload():
    ep = EpisodeContext(cycle_id=1)
    payload = ep.to_payload()
    assert payload["derivation"] == {"proposals": [], "summary": ""}


# -- ActionGenerator derivation candidate --------------------------------


def _ep_payload_with_derivation(
    derivation: Dict[str, Any],
    llm_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "affect": {"dominant_emotion": "neutral", "intensity": 0.2, "valence": 0.0, "arousal": 0.0},
        "active_goals": [{"goal_id": "g1", "description": "test"}],
        "curiosity": {},
        "narrative_context": "",
        "identity": {},
        "drives": {},
        "dominant_drive": "",
        "perception": {"content": "hi", "tags": []},
        "recalled_memories": [],
        "symbolic_law": {"violations": [], "compliant": True},
        "derivation": derivation,
        "llm_context": llm_context or {},
    }


def test_derivation_candidate_fires_when_proposals_strong():
    deriv = {
        "proposals": [
            {
                "kind": "kg_inference",
                "source": "reasoning",
                "confidence": 0.7,
                "payload": {"subject": "Sun", "predicate": "is", "object": "star"},
            },
        ],
        "summary": "Sun is star",
    }
    ep = _ep_payload_with_derivation(deriv)
    ag = ActionGenerator(composer=None)
    action = ag.generate(
        ep,
        [],
        strategy_hint=None,
        is_in_conversation=False,
    )
    assert isinstance(action, dict)
    assert "strategy" in action


def test_derivation_candidate_env_update_builds_parts():
    deriv = {
        "proposals": [
            {
                "kind": "env_update",
                "source": "llm_context",
                "confidence": 0.5,
                "payload": {"key": "mode", "value": "test"},
            },
        ],
        "summary": "env: mode=test",
    }
    cand = ActionGenerator._candidate_derivation(
        {
            "derivation": deriv,
            "llm_context": {},
        }
    )
    assert cand is not None
    assert "env:" in " ".join(cand.content_elements).lower()


def test_derivation_candidate_suppressed_when_llm_strong():
    deriv = {
        "proposals": [
            {
                "kind": "kg_inference",
                "source": "reasoning",
                "confidence": 0.7,
                "payload": {"subject": "A", "predicate": "r", "object": "B"},
            },
        ],
        "summary": "A r B",
    }
    llm = {"answer": "The answer is clear.", "confidence": 0.8}
    ep = _ep_payload_with_derivation(deriv, llm_context=llm)
    cand = ActionGenerator._candidate_derivation(
        {
            "derivation": deriv,
            "llm_context": llm,
        }
    )
    assert cand is None


def test_derivation_candidate_none_when_no_proposals():
    cand = ActionGenerator._candidate_derivation(
        {
            "derivation": {"proposals": [], "summary": ""},
        }
    )
    assert cand is None


def test_action_generate_with_empty_derivation():
    ep = _ep_payload_with_derivation({"proposals": [], "summary": ""})
    ag = ActionGenerator(composer=None)
    action = ag.generate(
        ep,
        [],
        strategy_hint=None,
        is_in_conversation=True,
        utterance_style="conversational",
    )
    assert isinstance(action, dict)
    assert action.get("strategy") != "derivation"
