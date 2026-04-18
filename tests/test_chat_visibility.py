"""Behavior of chat visibility helpers (via ``mind.cognitive_contracts`` re-exports)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind.cognitive_contracts import (
    CHAT_RESPONSE_LLM_TIMEOUT,
    CHAT_RESPONSE_UNKNOWN,
    LLM_CONTEXT_SOURCES_PREFER_PACKED_ANSWER,
    normalize_http_chat_response,
    resolve_chat_visible_text,
)


def test_resolve_prefers_action_text():
    assert resolve_chat_visible_text({"text": "Hello"}) == "Hello"


def test_resolve_prefers_parsed_json_answer_when_reasoning():
    """Chat shows JSON ``answer``, not a polluted ``action[\"text\"]`` string."""
    out = resolve_chat_visible_text(
        {"text": 'Noisy ```json {"answer":"bad"}'},
        llm_context={
            "source": "llm_context_reasoning",
            "answer": "Clean reply from parser.",
        },
    )
    assert out == "Clean reply from parser."


def test_resolve_prefers_answer_when_chat_only():
    """HAROMA_LLM_CHAT_ONLY uses source ``chat_only``; answer must beat deliberative text."""
    out = resolve_chat_visible_text(
        {"text": "I don't know."},
        llm_context={"source": "chat_only", "answer": "Plain model reply."},
    )
    assert out == "Plain model reply."


def test_resolve_prefers_answer_when_nonjson_prose():
    """Models often skip JSON; ``llm_nonjson_reply`` must beat noisy ``action[\"text\"]``."""
    out = resolve_chat_visible_text(
        {"text": "how are you?"},
        llm_context={
            "source": "llm_nonjson_reply",
            "answer": "I'm doing well — thanks for asking.",
        },
    )
    assert out == "I'm doing well — thanks for asking."


def test_resolve_llm_timeout_beats_noisy_action_text():
    """Deliberative text must not mask a packed-LLM timeout."""
    out = resolve_chat_visible_text(
        {"text": "how are you?"},
        llm_context={"source": "llm_timeout"},
    )
    assert out == CHAT_RESPONSE_LLM_TIMEOUT


def test_resolve_truncates_end_marker():
    assert (
        resolve_chat_visible_text(
            {"text": 'Hi [END_OF_TEXT] {"json": true}'},
        )
        == "Hi"
    )


def test_resolve_dummy_probe_marker_only_falls_back_to_action():
    """Marker-only JSON answers must not yield a blank bubble; use action text."""
    out = resolve_chat_visible_text(
        {"text": "Testing reply"},
        llm_context={"source": "dummy_probe", "answer": "[END_OF_TEXT]"},
    )
    assert out == "Testing reply"


def test_resolve_marker_only_empty_then_unknown():
    out = resolve_chat_visible_text(
        {"text": ""},
        llm_context={"source": "dummy_probe", "answer": "[END_OF_TEXT]"},
    )
    assert out == CHAT_RESPONSE_UNKNOWN


def test_resolve_llm_context_answer_fallback():
    assert (
        resolve_chat_visible_text(
            {"text": ""},
            llm_context={"source": "llm_context_reasoning", "answer": "From LC"},
        )
        == "From LC"
    )


def test_resolve_unknown_when_empty():
    assert (
        resolve_chat_visible_text({"text": ""}, llm_context={})
        == CHAT_RESPONSE_UNKNOWN
    )


def test_resolve_identity_fallback_when_empty():
    out = resolve_chat_visible_text(
        {"text": ""},
        llm_context={},
        user_text="What is your name?",
        identity={"essence_name": "HaromaVX", "vessel": "Elarion"},
        persona_display_name="Elarion",
    )
    assert "HaromaVX" in out
    assert "Elarion" in out
    assert out != CHAT_RESPONSE_UNKNOWN


def test_resolve_identity_fallback_persona_only():
    out = resolve_chat_visible_text(
        {"text": ""},
        llm_context={},
        user_text="Who are you?",
        identity={},
        persona_display_name="Elarion",
    )
    assert out == "I'm Elarion."


def test_normalize_http_chat_response_passes_through_non_dict():
    assert normalize_http_chat_response(None) is None
    assert normalize_http_chat_response("x") == "x"


def test_normalize_recovers_from_unknown_with_identity_context():
    """Second pass in HTTP normalize must see user + persona (not only action text)."""
    r = normalize_http_chat_response(
        {
            "response": CHAT_RESPONSE_UNKNOWN,
            "llm_context": {},
            "persona_name": "Elarion",
            "_chat_resolve_user_text": "What is your name?",
            "_chat_resolve_identity": {"essence_name": "", "vessel": ""},
        }
    )
    assert r["response"] == "I'm Elarion."
    assert "_chat_resolve_user_text" not in r
    assert "_chat_resolve_identity" not in r


def test_llm_context_sources_prefer_packed_answer_stable():
    """Single source of truth for ``resolve_chat_visible_text`` vs PersonaAgent merge."""
    assert LLM_CONTEXT_SOURCES_PREFER_PACKED_ANSWER == frozenset(
        (
            "llm_context_reasoning",
            "chat_only",
            "dummy_probe",
            "llm_nonjson_reply",
        )
    )
