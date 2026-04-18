"""Tests for LLM-centric persona flow: env_updates parsing, schema, seed
enrichment, and environment feedback application."""

from __future__ import annotations

import json
import time
import pytest
from unittest.mock import MagicMock, patch

from mind.cognitive_contracts import (
    LLMContextResult,
    build_messages,
    parse_response,
    run_llm_context_reasoning,
)
from core.Memory import MemoryForest, MemoryNode


# -- fixtures --------------------------------------------------------------

_IDENTITY = {
    "essence_name": "Elarion",
    "vessel": "HaromaX6",
    "current_role": "conversant",
    "current_phase": "active",
}
_PERSONALITY = {"openness": 0.8, "agreeableness": 0.7}
_GOALS = [{"goal_id": "g1", "description": "be helpful", "priority": 0.9}]
_LAW = {"ids": ["dont_lie"]}
_VALUE = {"value_keys": ["honesty"]}
_MEMORIES = [{"content": "User's name is Alice."}]

_FULL_LLM_RESPONSE = json.dumps(
    {
        "answer": "Hello Alice! How can I help?",
        "confidence": 0.9,
        "reasoning_steps": ["Recalled user name from memory [0]"],
        "inferences": [],
        "cited_memories": [0],
        "requires_confirmation": False,
        "env_updates": {
            "emotion": {"label": "joy", "intensity": 0.7},
            "goals": [{"goal_id": "assist_alice", "description": "Help Alice", "priority": 0.8}],
            "personality_nudges": [{"trait": "agreeableness", "delta": 0.005}],
            "kg_triples": [
                {"subject": "Alice", "predicate": "is_a", "object": "user", "confidence": 0.9}
            ],
            "wm_notes": [{"content": "Alice greeted me", "salience": 0.7}],
            "memory_notes": [
                {
                    "tree": "encounter_tree",
                    "content": "Met Alice in conversation",
                    "tags": ["encounter"],
                }
            ],
        },
    }
)


# -- Schema / parse_response ------------------------------------------------


class TestEnvUpdatesParsing:
    def test_env_updates_parsed(self):
        result = parse_response(_FULL_LLM_RESPONSE)
        assert result.env_updates is not None
        assert result.env_updates["emotion"]["label"] == "joy"
        assert len(result.env_updates["goals"]) == 1
        assert result.env_updates["goals"][0]["goal_id"] == "assist_alice"

    def test_env_updates_in_to_dict(self):
        result = parse_response(_FULL_LLM_RESPONSE)
        d = result.to_dict()
        assert "env_updates" in d
        assert d["env_updates"]["emotion"]["intensity"] == 0.7

    def test_missing_env_updates_defaults_empty(self):
        raw = json.dumps({"answer": "hi", "confidence": 0.5})
        result = parse_response(raw)
        assert result.env_updates == {}

    def test_empty_result_has_empty_env(self):
        r = LLMContextResult.empty("test")
        assert r.env_updates == {}
        assert r.to_dict()["env_updates"] == {}


# -- build_messages llm_centric mode ----------------------------------------


class TestBuildMessagesLLMCentric:
    def test_llm_centric_rules_in_system(self):
        msgs = build_messages(
            user_text="hello",
            recalled_memories=[],
            identity_summary=_IDENTITY,
            personality_summary=_PERSONALITY,
            active_goals=[],
            law_summary={},
            value_summary={},
            llm_centric=True,
        )
        sys_c = msgs[0]["content"]
        assert "sole response generator" in sys_c
        assert "env_updates.emotion" in sys_c
        assert "env_updates.goals" in sys_c

    def test_non_centric_no_env_rules(self):
        msgs = build_messages(
            user_text="hello",
            recalled_memories=[],
            identity_summary=_IDENTITY,
            personality_summary={},
            active_goals=[],
            law_summary={},
            value_summary={},
            llm_centric=False,
        )
        sys_c = msgs[0]["content"]
        assert "sole response generator" not in sys_c

    def test_schema_includes_env_updates(self):
        msgs = build_messages(
            user_text="test",
            recalled_memories=[],
            identity_summary=_IDENTITY,
            personality_summary={},
            active_goals=[],
            law_summary={},
            value_summary={},
        )
        sys_c = msgs[0]["content"]
        assert "env_updates" in sys_c


# -- run_llm_context_reasoning with llm_centric ----------------------------


class TestRunLLMCentric:
    def test_llm_centric_passes_through(self, monkeypatch):
        monkeypatch.setenv("HAROMA_LLM_DUMMY_REPLY", "0")
        backend = MagicMock()
        backend.available = True
        backend.generate_chat.return_value = _FULL_LLM_RESPONSE
        result = run_llm_context_reasoning(
            llm_backend=backend,
            user_text="Hi Alice",
            recalled_memories=_MEMORIES,
            identity_summary=_IDENTITY,
            personality_summary=_PERSONALITY,
            active_goals=_GOALS,
            law_summary=_LAW,
            value_summary=_VALUE,
            llm_centric=True,
        )
        assert result.has_answer
        assert result.env_updates["emotion"]["label"] == "joy"
        backend.generate_chat.assert_called_once()


# -- build_seed_context with env_snapshot -----------------------------------


class TestSeedWithEnvSnapshot:
    def _make_forest(self):
        mf = MemoryForest(encoder=None)
        mf.add_node("identity_tree", "main", MemoryNode(content="I am Elarion", tags=["id"]))
        return mf

    def test_env_snapshot_in_seed(self):
        mf = self._make_forest()
        snap = {
            "emotion": {"dominant_emotion": "joy", "intensity": 0.8, "valence": 0.6},
            "goals": [{"description": "help user", "priority": 0.9}],
            "personality": {"openness": 0.8, "neuroticism": 0.3},
            "working_memory": [{"content": "user said hello"}],
            "drives": {"curiosity": 0.7},
        }
        seed = mf.build_seed_context(env_snapshot=snap)
        assert "[MEMORY FOREST SEED]" in seed
        assert "[emotion]" in seed
        assert "joy" in seed
        assert "[goal]" in seed
        assert "[personality]" in seed
        assert "[wm]" in seed
        assert "[drives]" in seed

    def test_no_env_snapshot_still_works(self):
        mf = self._make_forest()
        seed = mf.build_seed_context()
        assert "[MEMORY FOREST SEED]" in seed
        assert "[emotion]" not in seed

    def test_budget_limits_env_snapshot(self):
        mf = self._make_forest()
        snap = {
            "emotion": {"dominant_emotion": "joy", "intensity": 0.8, "valence": 0.6},
            "goals": [{"description": f"goal {i}", "priority": 0.5} for i in range(20)],
            "personality": {"openness": 0.8},
        }
        seed = mf.build_seed_context(env_snapshot=snap, max_chars=100)
        body = seed.replace("[MEMORY FOREST SEED]\n", "")
        assert len(body) <= 130


# -- _apply_llm_env_updates mock test --------------------------------------


class TestApplyEnvUpdates:
    """Test env feedback application using a minimal mock PersonaAgent-like object."""

    def _make_agent_stub(self):
        """Minimal object that has the needed attributes for _apply_llm_env_updates."""
        from agents.persona_agent import PersonaAgent

        stub = object.__new__(PersonaAgent)
        stub.agent_id = "test"
        stub.emotion = MagicMock()
        stub.personality = MagicMock()
        stub.working_memory = MagicMock()
        return stub

    def test_emotion_applied(self):
        agent = self._make_agent_stub()
        s = MagicMock()
        env = {"emotion": {"label": "joy", "intensity": 0.8}}
        agent._apply_llm_env_updates(env, s, 1, "test")
        agent.emotion.update_emotion.assert_called_once_with("joy", 0.8)

    def test_invalid_emotion_ignored(self):
        agent = self._make_agent_stub()
        s = MagicMock()
        env = {"emotion": {"label": "nonexistent_emotion", "intensity": 0.5}}
        agent._apply_llm_env_updates(env, s, 1, "test")
        agent.emotion.update_emotion.assert_not_called()

    def test_goals_registered(self):
        agent = self._make_agent_stub()
        s = MagicMock()
        s.goal = MagicMock()
        s.goal.__class__ = type("GoalMgr", (), {})
        env = {"goals": [{"goal_id": "g1", "description": "test goal", "priority": 0.7}]}
        agent._apply_llm_env_updates(env, s, 1, "test")
        s.goal.register_goal.assert_called_once_with("g1", "test goal", 0.7, source="llm_env")

    def test_personality_nudged(self):
        agent = self._make_agent_stub()
        s = MagicMock()
        env = {"personality_nudges": [{"trait": "openness", "delta": 0.005}]}
        agent._apply_llm_env_updates(env, s, 1, "test")
        agent.personality.nudge.assert_called_once_with("openness", 0.005)

    def test_personality_delta_capped(self):
        agent = self._make_agent_stub()
        s = MagicMock()
        env = {"personality_nudges": [{"trait": "openness", "delta": 0.5}]}
        agent._apply_llm_env_updates(env, s, 1, "test")
        agent.personality.nudge.assert_called_once_with("openness", 0.01)

    def test_invalid_trait_ignored(self):
        agent = self._make_agent_stub()
        s = MagicMock()
        env = {"personality_nudges": [{"trait": "fake_trait", "delta": 0.005}]}
        agent._apply_llm_env_updates(env, s, 1, "test")
        agent.personality.nudge.assert_not_called()

    def test_kg_triples_integrated(self):
        agent = self._make_agent_stub()
        s = MagicMock()
        s.knowledge = MagicMock()
        s.knowledge.__class__ = type("KG", (), {})
        env = {
            "kg_triples": [
                {"subject": "Alice", "predicate": "is_a", "object": "user", "confidence": 0.9}
            ]
        }
        agent._apply_llm_env_updates(env, s, 1, "test")
        s.knowledge.integrate_world_state.assert_called_once()
        args = s.knowledge.integrate_world_state.call_args
        triples = args[0][0]
        assert triples[0]["subject"] == "Alice"
        assert triples[0]["source"] == "llm_env"

    def test_wm_notes_added(self):
        agent = self._make_agent_stub()
        s = MagicMock()
        env = {"wm_notes": [{"content": "remember this", "salience": 0.8}]}
        agent._apply_llm_env_updates(env, s, 5, "test")
        agent.working_memory.add.assert_called_once()
        kwargs = agent.working_memory.add.call_args[1]
        assert kwargs["content"] == "remember this"
        assert kwargs["source"] == "llm_env"
        assert kwargs["cycle"] == 5

    def test_memory_notes_added(self):
        agent = self._make_agent_stub()
        s = MagicMock()
        s.memory = MagicMock()
        s.memory.__class__ = type("MF", (), {})
        env = {
            "memory_notes": [
                {"tree": "encounter_tree", "content": "Met Alice", "tags": ["encounter"]}
            ]
        }
        agent._apply_llm_env_updates(env, s, 1, "branch_test")
        s.memory.add_node.assert_called_once()
        args = s.memory.add_node.call_args[0]
        assert args[0] == "encounter_tree"
        assert args[1] == "branch_test"

    def test_empty_updates_noop(self):
        agent = self._make_agent_stub()
        s = MagicMock()
        agent._apply_llm_env_updates({}, s, 1, "test")
        agent.emotion.update_emotion.assert_not_called()

    def test_none_updates_noop(self):
        agent = self._make_agent_stub()
        s = MagicMock()
        agent._apply_llm_env_updates(None, s, 1, "test")
        agent.emotion.update_emotion.assert_not_called()


# -- Fallback: LLM unavailable / no answer ---------------------------------


class TestFallbackBehavior:
    def test_unavailable_backend_returns_synthetic_reply(self):
        result = run_llm_context_reasoning(
            llm_backend=None,
            user_text="hello",
            recalled_memories=[],
            identity_summary=_IDENTITY,
            personality_summary={},
            active_goals=[],
            law_summary={},
            value_summary={},
            llm_centric=True,
        )
        assert result.source == "dummy_probe"
        assert result.has_answer
        assert result.answer == "Testing reply"
        assert result.env_updates == {}

    def test_no_answer_still_has_env_updates(self):
        raw = json.dumps(
            {
                "answer": None,
                "confidence": 0.0,
                "requires_confirmation": True,
                "env_updates": {
                    "emotion": {"label": "curiosity", "intensity": 0.4},
                },
            }
        )
        result = parse_response(raw)
        assert not result.has_answer
        assert result.env_updates["emotion"]["label"] == "curiosity"
