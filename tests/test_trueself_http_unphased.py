"""TrueSelf HTTP /chat bypasses phased cognitive scheduling when enabled."""

from __future__ import annotations

import sys

import pytest

_REPO = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from agents.message_bus import Message
from agents.persona_agent import PersonaAgent
from agents.trueself_agent import TrueSelfAgent


def test_unphased_trueself_http_traced_conversant_gate(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_TRUESELF_HTTP_CHAT_UNPHASED", "1")
    ts = TrueSelfAgent.__new__(TrueSelfAgent)
    msg = Message(
        sender_id="input",
        channel="input",
        content={"text": "hello"},
        message_type="input",
        metadata={"cognitive_trace_id": "trace-1"},
    )
    assert ts._unphased_trueself_http_traced_conversant(msg, "conversant") is True
    assert ts._unphased_trueself_http_traced_conversant(msg, "observer") is False

    msg2 = Message(
        sender_id="input",
        channel="input",
        content={"text": "hello"},
        message_type="input",
        metadata={},
    )
    assert ts._unphased_trueself_http_traced_conversant(msg2, "conversant") is False

    monkeypatch.setenv("HAROMA_TRUESELF_HTTP_CHAT_UNPHASED", "0")
    assert ts._unphased_trueself_http_traced_conversant(msg, "conversant") is False


def test_unphased_delegated_specialist_http_traced_conversant(monkeypatch: pytest.MonkeyPatch):
    """Specialists receiving ``trueself_delegate`` use the same unphased gate as TrueSelf."""
    monkeypatch.setenv("HAROMA_TRUESELF_HTTP_CHAT_UNPHASED", "1")
    pa = PersonaAgent.__new__(PersonaAgent)
    msg = Message(
        sender_id="trueself",
        channel="trueself_delegate",
        content={"text": "hello"},
        message_type="delegation",
        metadata={"cognitive_trace_id": "trace-deleg"},
    )
    assert pa._unphased_trueself_http_traced_conversant(msg, "conversant") is True
    msg_no_trace = Message(
        sender_id="trueself",
        channel="trueself_delegate",
        content={"text": "hello"},
        message_type="delegation",
        metadata={},
    )
    assert pa._unphased_trueself_http_traced_conversant(msg_no_trace, "conversant") is False
