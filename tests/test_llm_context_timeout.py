"""Packed-context LLM call must not block HTTP forever."""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.persona_agent import PersonaAgent
from mind.cognitive_contracts import (
    CHAT_RESPONSE_UNKNOWN,
    normalize_http_chat_response,
    packed_messages_stats,
    run_llm_context_reasoning,
    truncate_chat_at_end_marker,
)


class _SlowBackend:
    available = True

    def generate_chat(self, *args, **kwargs):
        time.sleep(2.0)
        return '{"answer":"late","confidence":0.9}'


def test_generate_chat_respects_timeout(monkeypatch):
    monkeypatch.setenv("HAROMA_LLM_CONTEXT_TIMEOUT_SEC", "0.35")
    r = run_llm_context_reasoning(
        llm_backend=_SlowBackend(),
        user_text="hi",
        recalled_memories=[],
        identity_summary={"essence_name": "X", "vessel": "Y"},
        personality_summary={},
        active_goals=[],
        law_summary={},
        value_summary={},
        knowledge_triples=[],
        discourse_context="",
        nlu_result=None,
        memory_forest_seed="",
        llm_centric=False,
        max_tokens=64,
        temperature=0.3,
    )
    assert r.source == "llm_timeout"
    assert not r.has_answer


def test_chat_visible_response_uses_timeout_copy_not_unknown():
    pa = PersonaAgent.__new__(PersonaAgent)
    out = PersonaAgent._chat_visible_response(
        pa,
        {"text": None},
        llm_context={"source": "llm_timeout"},
    )
    assert "don't know" not in out.lower()
    assert "time limit" in out.lower() or "local model" in out.lower()


def test_chat_visible_response_falls_back_to_llm_context_answer():
    pa = PersonaAgent.__new__(PersonaAgent)
    out = PersonaAgent._chat_visible_response(
        pa,
        {"text": "", "strategy": "inquire"},
        llm_context={
            "source": "llm_context_reasoning",
            "answer": "Hello from the packed LLM.",
        },
    )
    assert out == "Hello from the packed LLM."
    assert "don't know" not in out.lower()


def test_chat_visible_response_json_parse_failed_not_unknown():
    pa = PersonaAgent.__new__(PersonaAgent)
    out = PersonaAgent._chat_visible_response(
        pa,
        {"text": None},
        llm_context={"source": "json_parse_failed"},
    )
    assert "don't know" not in out.lower()
    assert "parse" in out.lower() or "json" in out.lower()


def test_chat_visible_response_llm_empty_not_unknown():
    pa = PersonaAgent.__new__(PersonaAgent)
    out = PersonaAgent._chat_visible_response(
        pa,
        {"text": None},
        llm_context={"source": "llm_empty_response"},
    )
    assert "don't know" not in out.lower()
    assert "no reply" in out.lower() or "reply text" in out.lower()


def test_normalize_http_chat_response_falls_back_to_llm_context_answer():
    r = normalize_http_chat_response(
        {
            "response": "",
            "llm_context": {
                "answer": "Recovered from payload",
                "source": "llm_context_reasoning",
            },
        }
    )
    assert r["response"] == "Recovered from payload"
    r2 = normalize_http_chat_response({"response": None, "llm_context": {}})
    assert r2["response"] == CHAT_RESPONSE_UNKNOWN


def test_truncate_chat_at_end_marker_strips_suffix():
    raw = (
        'I am fine, thank you. How are you? [END_OF_TEXT] The user asks "Hi". '
        '```json\n{"answer": "x"}\n```'
    )
    assert truncate_chat_at_end_marker(raw) == "I am fine, thank you. How are you?"


def test_normalize_http_chat_response_truncates_at_end_marker():
    r = normalize_http_chat_response(
        {"response": "Hello [END_OF_TEXT] extra narration", "llm_context": {}}
    )
    assert r["response"] == "Hello"


def test_normalize_http_empty_response_respects_llm_source():
    """Same semantics as resolve_chat_visible_text (not only answer fallback)."""
    r = normalize_http_chat_response(
        {
            "response": "",
            "llm_context": {"source": "llm_timeout", "answer": ""},
        }
    )
    assert "don't know" not in r["response"].lower()
    assert "time limit" in r["response"].lower() or "local model" in r["response"].lower()


def test_chat_visible_response_truncates_action_text_at_end_marker():
    pa = PersonaAgent.__new__(PersonaAgent)
    out = PersonaAgent._chat_visible_response(
        pa,
        {"text": "Hi there [END_OF_TEXT] The user said hello."},
        llm_context={},
    )
    assert out == "Hi there"


def test_unlimited_context_uses_max_generate_cap(monkeypatch):
    """HAROMA_LLM_CONTEXT_TIMEOUT_SEC=0 still caps native decode via HAROMA_LLM_MAX_GENERATE_SEC."""
    monkeypatch.setenv("HAROMA_LLM_CONTEXT_TIMEOUT_SEC", "0")
    monkeypatch.setenv("HAROMA_LLM_MAX_GENERATE_SEC", "0.2")

    class _Slow:
        available = True

        def generate_chat(self, *args, **kwargs):
            time.sleep(1.0)
            return '{"answer":"late"}'

    r = run_llm_context_reasoning(
        llm_backend=_Slow(),
        user_text="hi",
        recalled_memories=[],
        identity_summary={"essence_name": "X", "vessel": "Y"},
        personality_summary={},
        active_goals=[],
        law_summary={},
        value_summary={},
        knowledge_triples=[],
        discourse_context="",
        nlu_result=None,
        memory_forest_seed="",
        llm_centric=False,
        max_tokens=64,
        temperature=0.3,
    )
    assert r.source == "llm_timeout"


def test_generate_chat_no_timeout_when_disabled(monkeypatch):
    monkeypatch.setenv("HAROMA_LLM_CONTEXT_TIMEOUT_SEC", "0")
    monkeypatch.setenv("HAROMA_LLM_MAX_GENERATE_SEC", "0")
    r = run_llm_context_reasoning(
        llm_backend=_SlowBackend(),
        user_text="hi",
        recalled_memories=[],
        identity_summary={"essence_name": "X", "vessel": "Y"},
        personality_summary={},
        active_goals=[],
        law_summary={},
        value_summary={},
        knowledge_triples=[],
        discourse_context="",
        nlu_result=None,
        memory_forest_seed="",
        llm_centric=False,
        max_tokens=64,
        temperature=0.3,
    )
    assert r.source == "llm_context_reasoning"
    assert r.answer == "late"


class _NoCallBackend:
    """Fails if generate_chat is invoked (dummy mode must return earlier)."""

    available = True
    _n_ctx = 4096

    def generate_chat(self, *args, **kwargs):
        raise AssertionError("generate_chat must not run in dummy mode")


def test_dummy_reply_skips_generate_chat_and_reports_stats(monkeypatch):
    monkeypatch.setenv("HAROMA_LLM_DUMMY_REPLY", "1")
    r = run_llm_context_reasoning(
        llm_backend=_NoCallBackend(),
        user_text="hello probe",
        recalled_memories=[{"content": "x" * 100, "tags": []}],
        identity_summary={"essence_name": "Test", "vessel": "V"},
        personality_summary={"openness": 0.5},
        active_goals=[],
        law_summary={},
        value_summary={},
        knowledge_triples=[],
        discourse_context="",
        nlu_result=None,
        memory_forest_seed="",
        llm_centric=False,
        max_tokens=64,
        temperature=0.3,
    )
    assert r.source == "dummy_probe"
    assert r.has_answer
    assert "Skipped generate_chat" in (r.answer or "")
    assert "chars" in (r.answer or "").lower()
    assert "chars_by_role" in (r.answer or "")
    assert r.prompt_info is not None
    assert r.prompt_info.get("n_ctx_allocated") == 4096
    assert "prompt_info" in r.to_dict()


def test_packed_messages_stats_sums_roles():
    msgs = [
        {"role": "system", "content": "ab"},
        {"role": "user", "content": "cdé"},
    ]
    s = packed_messages_stats(msgs)
    assert s["message_count"] == 2
    assert s["total_chars"] == 5
    assert s["total_utf8_bytes"] == len("ab".encode()) + len("cdé".encode())
    assert s["chars_by_role"]["system"] == 2
    assert s["chars_by_role"]["user"] == 3
    assert s["est_tokens_approx"] >= 1


class _PlainChatBackend:
    available = True
    _n_ctx = 8192

    def generate_chat(self, messages, max_tokens=256, temperature=0.3, **kwargs):
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        return "plain assistant reply"


def test_chat_only_sends_user_only_and_returns_plain(monkeypatch):
    monkeypatch.setenv("HAROMA_LLM_CHAT_ONLY", "1")
    monkeypatch.setenv("HAROMA_LLM_CONTEXT_TIMEOUT_SEC", "5")
    r = run_llm_context_reasoning(
        llm_backend=_PlainChatBackend(),
        user_text="hello",
        recalled_memories=[{"content": "SHOULD_NOT_APPEAR", "tags": []}],
        identity_summary={"essence_name": "X", "soul": {"big": "json"}},
        personality_summary={},
        active_goals=[],
        law_summary={},
        value_summary={},
        knowledge_triples=[],
        discourse_context="SHOULD_NOT_APPEAR",
        nlu_result=None,
        memory_forest_seed="",
        llm_centric=False,
        max_tokens=64,
        temperature=0.3,
    )
    assert r.source == "chat_only"
    assert r.answer == "plain assistant reply"
    assert r.prompt_info is not None
    assert r.prompt_info["chat_only"] is True
    assert r.prompt_info["n_ctx_allocated"] == 8192
    assert r.prompt_info["prompt_chars"] == len("hello")


def test_prompt_info_on_packed_when_env(monkeypatch):
    monkeypatch.setenv("HAROMA_LLM_PROMPT_INFO", "1")
    monkeypatch.setenv("HAROMA_LLM_CONTEXT_TIMEOUT_SEC", "5")
    monkeypatch.delenv("HAROMA_LLM_CHAT_ONLY", raising=False)

    class _JsonBackend:
        available = True
        _n_ctx = 4096

        def generate_chat(self, *a, **k):
            return '{"answer":"y","confidence":1}'

    r = run_llm_context_reasoning(
        llm_backend=_JsonBackend(),
        user_text="hi",
        recalled_memories=[],
        identity_summary={"essence_name": "Z"},
        personality_summary={},
        active_goals=[],
        law_summary={},
        value_summary={},
        knowledge_triples=[],
        discourse_context="",
        nlu_result=None,
        memory_forest_seed="",
        llm_centric=False,
        max_tokens=64,
        temperature=0.3,
    )
    assert r.answer == "y"
    assert r.prompt_info is not None
    assert r.prompt_info["n_ctx_allocated"] == 4096
    assert r.prompt_info["chat_only"] is False
