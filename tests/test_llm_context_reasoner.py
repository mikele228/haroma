"""Unit tests for engine.LLMContextReasoner — prompt builder, parser, and orchestrator."""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock

from engine.LLMContextReasoner import (
    LLMContextResult,
    build_messages,
    parse_response,
    run_llm_context_reasoning,
    _extract_json,
)


# -- fixtures ----------------------------------------------------------

_IDENTITY = {
    "essence_name": "Elarion",
    "vessel": "HaromaX6",
    "current_role": "conversant",
    "current_phase": "active",
}
_PERSONALITY = {"openness": 0.8, "agreeableness": 0.7}
_GOALS = [{"goal_id": "g1", "description": "be helpful", "priority": 0.9}]
_LAW = {"ids": ["dont_lie", "be_kind"], "count": 2}
_VALUE = {"value_keys": ["honesty", "curiosity"]}
_MEMORIES = [
    {"content": "User's name is Alice."},
    {"content": "Alice asked about weather yesterday."},
]


# -- build_messages ----------------------------------------------------


class TestBuildMessages:
    def test_returns_two_messages(self):
        msgs = build_messages(
            user_text="What is my name?",
            recalled_memories=_MEMORIES,
            identity_summary=_IDENTITY,
            personality_summary=_PERSONALITY,
            active_goals=_GOALS,
            law_summary=_LAW,
            value_summary=_VALUE,
        )
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_system_contains_persona(self):
        msgs = build_messages(
            user_text="hello",
            recalled_memories=[],
            identity_summary=_IDENTITY,
            personality_summary=_PERSONALITY,
            active_goals=[],
            law_summary={},
            value_summary={},
        )
        assert "Elarion" in msgs[0]["content"]

    def test_system_includes_birth_when_present(self):
        ident = {**_IDENTITY, "birth": "2025-04-27T00:00:00Z"}
        msgs = build_messages(
            user_text="How old are you?",
            recalled_memories=[],
            identity_summary=ident,
            personality_summary={},
            active_goals=[],
            law_summary={},
            value_summary={},
        )
        assert "2025-04-27" in msgs[0]["content"]

    def test_system_prefers_soul_snapshot_over_birth_only_line(self):
        ident = {
            **_IDENTITY,
            "birth": "2025-04-27T00:00:00Z",
            "soul": {
                "essence": {"name": "HaromaVX", "birth": "2025-04-27T00:00:00Z"},
            },
        }
        msgs = build_messages(
            user_text="How old are you?",
            recalled_memories=[],
            identity_summary=ident,
            personality_summary={},
            active_goals=[],
            law_summary={},
            value_summary={},
        )
        sys_c = msgs[0]["content"]
        assert "Bound soul snapshot" in sys_c
        assert "HaromaVX" in sys_c
        assert "Soul-record origin timestamp" not in sys_c

    def test_system_rules_allow_persona_and_soul_grounding(self):
        msgs = build_messages(
            user_text="Who are you?",
            recalled_memories=[],
            identity_summary=_IDENTITY,
            personality_summary={},
            active_goals=[],
            law_summary={},
            value_summary={},
        )
        sys_c = msgs[0]["content"].lower()
        assert "soul snapshot" in sys_c
        assert "requires_confirmation=false" in sys_c

    def test_user_contains_memories_and_question(self):
        msgs = build_messages(
            user_text="What is my name?",
            recalled_memories=_MEMORIES,
            identity_summary=_IDENTITY,
            personality_summary=_PERSONALITY,
            active_goals=_GOALS,
            law_summary=_LAW,
            value_summary=_VALUE,
        )
        user_text = msgs[1]["content"]
        assert "Alice" in user_text
        assert "What is my name?" in user_text
        assert "[0]" in user_text

    def test_empty_recalls_omit_section(self):
        msgs = build_messages(
            user_text="hi",
            recalled_memories=[],
            identity_summary=_IDENTITY,
            personality_summary={},
            active_goals=[],
            law_summary={},
            value_summary={},
        )
        assert "RECALLED MEMORIES" not in msgs[1]["content"]

    def test_agent_environment_section_when_present(self):
        msgs = build_messages(
            user_text="status?",
            recalled_memories=[],
            identity_summary=_IDENTITY,
            personality_summary={},
            active_goals=[],
            law_summary={},
            value_summary={},
            agent_environment={
                "domain": "home",
                "entities": {"living_room": {"light": "on"}},
                "metrics": {"temp_c": 21.0},
            },
        )
        u = msgs[1]["content"]
        assert "[ENVIRONMENT STATE]" in u
        assert "home" in u

    def test_robot_body_section_when_extensions_robot_body(self):
        msgs = build_messages(
            user_text="move?",
            recalled_memories=[],
            identity_summary=_IDENTITY,
            personality_summary={},
            active_goals=_GOALS,
            law_summary={},
            value_summary={},
            agent_environment={
                "domain": "lab",
                "extensions": {
                    "robot_body": {
                        "body_defined": True,
                        "readings": {"hardware": {"approx_height_m": 1.7}},
                    }
                },
            },
        )
        u = msgs[1]["content"]
        assert "[ROBOT BODY STATE]" in u
        assert "approx_height_m" in u
        s = msgs[0]["content"]
        assert "body_actions" in s.lower() or "body_actions" in s
        assert "supports_goal_id" in s

    def test_robot_bridge_section_when_extensions_robot_bridge(self):
        msgs = build_messages(
            user_text="done?",
            recalled_memories=[],
            identity_summary=_IDENTITY,
            personality_summary={},
            active_goals=_GOALS,
            law_summary={},
            value_summary={},
            agent_environment={
                "domain": "lab",
                "extensions": {
                    "robot_bridge": {
                        "bridge_schema_version": 1,
                        "correlation_id": "x",
                        "results": [{"command_id": "cmd_1", "status": "failed", "detail": "clamped"}],
                    }
                },
            },
        )
        u = msgs[1]["content"]
        assert "[ROBOT BRIDGE FEEDBACK]" in u
        assert "cmd_1" in u
        assert "ROBOT BRIDGE" in msgs[0]["content"] or "bridge" in msgs[0]["content"].lower()


# -- _extract_json -----------------------------------------------------


class TestExtractJson:
    def test_clean_json(self):
        raw = '{"answer": "hi", "confidence": 0.8}'
        obj = _extract_json(raw)
        assert obj["answer"] == "hi"

    def test_markdown_fenced(self):
        raw = '```json\n{"answer": "hi"}\n```'
        obj = _extract_json(raw)
        assert obj["answer"] == "hi"

    def test_text_before_json(self):
        raw = 'Here is my response: {"answer": "hi", "confidence": 0.5}'
        obj = _extract_json(raw)
        assert obj is not None
        assert obj["answer"] == "hi"

    def test_garbage(self):
        assert _extract_json("no json here") is None

    def test_empty(self):
        assert _extract_json("") is None


# -- parse_response ----------------------------------------------------


class TestParseResponse:
    def test_valid_json(self):
        raw = json.dumps(
            {
                "answer": "Your name is Alice.",
                "confidence": 0.9,
                "reasoning_steps": ["Recall memory [0]"],
                "inferences": [
                    {"subject": "User", "predicate": "named", "object": "Alice", "confidence": 0.95}
                ],
                "cited_memories": [0],
                "requires_confirmation": False,
                "body_actions": [],
            }
        )
        result = parse_response(raw)
        assert result.has_answer
        assert result.answer == "Your name is Alice."
        assert result.confidence == 0.9
        assert result.is_grounded
        assert len(result.inferences) == 1
        assert result.cited_memories == [0]
        assert result.body_actions == []

    def test_parse_body_actions(self):
        raw = json.dumps(
            {
                "answer": "ok",
                "confidence": 0.8,
                "requires_confirmation": False,
                "body_actions": [
                    {
                        "label": "Approach user",
                        "layer": "localization",
                        "command": "move_base",
                        "parameters": {"speed": 0.3},
                        "supports_goal_id": "g1",
                        "rationale": "Advances helpfulness",
                        "safety_class": "caution",
                        "duration_hint_sec": 12.5,
                        "coordinate_frame": "map",
                        "preconditions": ["pose_valid"],
                        "priority": 0.9,
                        "resource": "base",
                        "cancel_current": True,
                        "confidence": 0.7,
                    }
                ],
            }
        )
        r = parse_response(raw)
        assert len(r.body_actions) == 1
        assert r.body_actions[0]["supports_goal_id"] == "g1"
        assert r.body_actions[0]["command"] == "move_base"
        assert r.body_actions[0]["safety_class"] == "caution"
        assert r.body_actions[0]["duration_hint_sec"] == 12.5
        assert r.body_actions[0]["coordinate_frame"] == "map"
        assert r.body_actions[0]["preconditions"] == ["pose_valid"]
        assert r.body_actions[0]["priority"] == 0.9
        assert r.body_actions[0]["resource"] == "base"
        assert r.body_actions[0]["cancel_current"] is True
        assert r.to_dict()["body_actions"][0]["label"] == "Approach user"

    def test_null_answer(self):
        raw = json.dumps({"answer": None, "confidence": 0.0})
        result = parse_response(raw)
        assert not result.has_answer
        assert not result.is_grounded

    def test_empty_response(self):
        result = parse_response(None)
        assert result.source == "llm_empty_response"
        assert not result.has_answer

    def test_plaintext_fallback_when_not_json(self):
        result = parse_response("Hello, this is a plain assistant reply.")
        assert result.source == "llm_nonjson_reply"
        assert result.answer.startswith("Hello")

    def test_bad_json(self):
        result = parse_response("not json at all")
        assert result.source == "llm_nonjson_reply"
        assert "not json" in (result.answer or "")

    def test_nonjson_repetitive_suffix_trimmed(self):
        """Degenerate decode: same block repeated; keep a single copy."""
        unit = "I'm fine. How are you? "
        junk = unit * 12
        result = parse_response(junk)
        assert result.source == "llm_nonjson_reply"
        assert result.answer is not None
        assert unit.strip() in result.answer
        assert result.answer.count("How are you?") <= 2

    def test_json_strict_rejects_plaintext(self, monkeypatch):
        monkeypatch.setenv("HAROMA_LLM_JSON_STRICT", "1")
        result = parse_response("Hello, this is a plain assistant reply.")
        assert result.source == "json_parse_failed"
        assert not result.has_answer

    def test_prose_plus_fenced_json_extracts_answer(self):
        raw = (
            'I speak first.\n```json\n'
            '{"answer": "Just this.", "confidence": 0.9, "requires_confirmation": false}\n'
            "```\n"
        )
        result = parse_response(raw)
        assert result.source == "llm_context_reasoning"
        assert result.answer == "Just this."

    def test_json_answer_key_alternate(self, monkeypatch):
        monkeypatch.setenv("HAROMA_LLM_JSON_ANSWER_KEY", "reply_text")
        raw = json.dumps({"reply_text": "Alt line.", "confidence": 0.5})
        result = parse_response(raw)
        assert result.answer == "Alt line."

    def test_whitespace_only_is_failed_parse(self):
        result = parse_response("   \n\t  ")
        assert result.source == "json_parse_failed"

    def test_requires_confirmation_blocks_grounded(self):
        raw = json.dumps(
            {
                "answer": "Maybe it's Alice?",
                "confidence": 0.9,
                "requires_confirmation": True,
            }
        )
        result = parse_response(raw)
        assert result.has_answer
        assert not result.is_grounded

    def test_low_confidence_no_citations_not_grounded(self):
        raw = json.dumps(
            {
                "answer": "I think so.",
                "confidence": 0.2,
                "requires_confirmation": False,
                "cited_memories": [],
            }
        )
        result = parse_response(raw)
        assert result.has_answer
        assert not result.is_grounded

    def test_confidence_clamped(self):
        raw = json.dumps({"answer": "x", "confidence": 5.0})
        result = parse_response(raw)
        assert result.confidence == 1.0


# -- LLMContextResult -------------------------------------------------


class TestLLMContextResult:
    def test_empty(self):
        r = LLMContextResult.empty("test")
        assert r.source == "test"
        assert not r.has_answer
        assert r.to_dict()["source"] == "test"

    def test_to_dict_roundtrip(self):
        r = LLMContextResult(answer="hi", confidence=0.5, source="test")
        d = r.to_dict()
        assert d["answer"] == "hi"
        assert d["confidence"] == 0.5


# -- run_llm_context_reasoning ----------------------------------------


class TestRunLLMContextReasoning:
    def test_unavailable_backend_uses_synthetic_dummy(self):
        result = run_llm_context_reasoning(
            llm_backend=None,
            user_text="hello",
            recalled_memories=[],
            identity_summary=_IDENTITY,
            personality_summary=_PERSONALITY,
            active_goals=[],
            law_summary={},
            value_summary={},
        )
        assert result.source == "dummy_probe"
        assert result.answer == "Testing reply"

    def test_with_mock_backend(self, monkeypatch):
        monkeypatch.setenv("HAROMA_LLM_DUMMY_REPLY", "0")
        backend = MagicMock()
        backend.available = True
        backend.generate_chat.return_value = json.dumps(
            {
                "answer": "Your name is Alice.",
                "confidence": 0.85,
                "reasoning_steps": ["Memory [0] says name is Alice"],
                "inferences": [],
                "cited_memories": [0],
                "requires_confirmation": False,
            }
        )
        result = run_llm_context_reasoning(
            llm_backend=backend,
            user_text="What is my name?",
            recalled_memories=_MEMORIES,
            identity_summary=_IDENTITY,
            personality_summary=_PERSONALITY,
            active_goals=_GOALS,
            law_summary=_LAW,
            value_summary=_VALUE,
        )
        assert result.has_answer
        assert result.is_grounded
        assert result.answer == "Your name is Alice."
        assert result.latency_ms > 0
        backend.generate_chat.assert_called_once()

    def test_payload_logging_emits_json_lines(self, monkeypatch, capsys):
        monkeypatch.setenv("HAROMA_LLM_DUMMY_REPLY", "0")
        monkeypatch.setenv("HAROMA_LLM_LOG_PAYLOAD", "1")
        backend = MagicMock()
        backend.available = True
        backend._n_ctx = 4096
        backend.generate_chat.return_value = '{"answer":"x","confidence":1}'
        run_llm_context_reasoning(
            llm_backend=backend,
            user_text="hi",
            recalled_memories=[],
            identity_summary=_IDENTITY,
            personality_summary={},
            active_goals=[],
            law_summary={},
            value_summary={},
        )
        out = capsys.readouterr().out
        assert "PAYLOAD_IN" in out and '"kind": "PAYLOAD_IN"' in out
        assert "PAYLOAD_OUT" in out and '"kind": "PAYLOAD_OUT"' in out
        assert '{"answer":"x"' in out or "answer" in out

    def test_backend_returns_none(self, monkeypatch):
        monkeypatch.setenv("HAROMA_LLM_DUMMY_REPLY", "0")
        backend = MagicMock()
        backend.available = True
        backend.generate_chat.return_value = None
        result = run_llm_context_reasoning(
            llm_backend=backend,
            user_text="hello",
            recalled_memories=[],
            identity_summary=_IDENTITY,
            personality_summary={},
            active_goals=[],
            law_summary={},
            value_summary={},
        )
        assert not result.has_answer
        assert result.source == "llm_empty_response"
