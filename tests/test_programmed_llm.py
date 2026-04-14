"""Programmed (rule-based) LLM backend — no weights."""

import json

import pytest

from engine.LLMBackend import LLMBackend
from mind.cognitive_contracts import parse_response
from engine.ProgrammedLLM import ProgrammedLLMResponder


@pytest.fixture
def responder():
    return ProgrammedLLMResponder()


def test_programmed_name_question(responder):
    raw = responder.generate_chat(
        messages=[
            {
                "role": "system",
                "content": "You are Elarion, phase=stable.\n[PERSONA]\nYou are Elarion.",
            },
            {"role": "user", "content": "[RECALLED MEMORIES]\n\nWhat is your name?"},
        ],
    )
    assert raw
    r = parse_response(raw)
    assert r.source == "llm_context_reasoning"
    assert r.answer
    assert "stub" in r.answer.lower() or "programmed" in r.answer.lower()
    assert r.requires_confirmation


def test_backend_programmed_flag():
    be = LLMBackend(use_programmed=True)
    assert be.available
    assert be.backend_type == "programmed"
    assert "programmed" in be.model_name
    js = be.generate_chat(
        messages=[{"role": "user", "content": "Hello!"}],
    )
    obj = json.loads(js)
    assert obj.get("answer")


def test_programmed_no_math_routing():
    be = LLMBackend(use_programmed=True)
    js = be.generate_chat(
        messages=[{"role": "user", "content": "What is 3+4?"}],
    )
    r = parse_response(js)
    assert r.answer
    assert "7" not in (r.answer or "")
    assert r.requires_confirmation
