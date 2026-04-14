"""Behavior of chat visibility helpers (via ``mind.cognitive_contracts`` re-exports)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind.cognitive_contracts import (
    CHAT_RESPONSE_UNKNOWN,
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


def test_resolve_truncates_end_marker():
    assert (
        resolve_chat_visible_text(
            {"text": 'Hi [END_OF_TEXT] {"json": true}'},
        )
        == "Hi"
    )


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


def test_normalize_http_chat_response_passes_through_non_dict():
    assert normalize_http_chat_response(None) is None
    assert normalize_http_chat_response("x") == "x"
