"""Tests for :mod:`mind.packed_llm_action`."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind.cognitive_contracts import merge_packed_llm_answer_into_action as merge_cc
from mind.packed_llm_action import build_llm_centric_action, merge_packed_llm_answer_into_action


def test_merge_noop_when_not_primary_path():
    action = {"text": "delib", "strategy": "x"}
    lc = {"source": "llm_context_reasoning", "answer": "model"}
    out = merge_cc(action, lc, llm_primary_path=False, llm_centric=False)
    assert out is action
    assert out["text"] == "delib"


def test_merge_noop_when_llm_centric():
    action = {"text": "direct", "strategy": "llm_centric"}
    lc = {"source": "llm_context_reasoning", "answer": "ignored"}
    out = merge_cc(action, lc, llm_primary_path=True, llm_centric=True)
    assert out["text"] == "direct"


def test_merge_copies_answer_on_primary_packed_path():
    action = {"text": "user echo?", "strategy": "respond"}
    lc = {"source": "llm_nonjson_reply", "answer": "I'm well, thanks."}
    out = merge_packed_llm_answer_into_action(
        action, lc, llm_primary_path=True, llm_centric=False
    )
    assert out["text"] == "I'm well, thanks."
    assert out["strategy"] == "llm_context"
    assert action["text"] == "user echo?"


def test_merge_identity_same_as_barrel():
    assert merge_cc is merge_packed_llm_answer_into_action


def test_build_llm_centric_action_conversational():
    a = build_llm_centric_action(
        llm_answer="Hello",
        llm_context_result={"confidence": 0.88},
        is_in_conversation=True,
    )
    assert a["text"] == "Hello"
    assert a["action_type"] == "respond"
    assert a["strategy"] == "llm_context"
    assert a["confidence"] == 0.88
    assert a["deliberation"]["winner"] == "llm_centric"


def test_build_llm_centric_action_reflect_when_not_in_conv():
    a = build_llm_centric_action(
        llm_answer="x",
        llm_context_result={},
        is_in_conversation=False,
    )
    assert a["action_type"] == "reflect"
