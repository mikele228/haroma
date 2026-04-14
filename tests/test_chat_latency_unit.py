"""Unit tests for agents.chat_latency (no BootAgent / conftest)."""

from __future__ import annotations

import time

import pytest

from agents.chat_latency import (
    trace_attach_to_payload,
    trace_init,
    trace_log_requested,
    trace_requested,
    trace_span,
)


def test_trace_span_and_attach():
    slot: dict = {"event": None, "result": None}
    trace_init(slot, log_to_console=False)
    trace_span(slot, "phase_a")
    time.sleep(0.005)
    trace_span(slot, "phase_b")
    payload: dict = {}
    trace_attach_to_payload(slot, payload)

    assert "latency_trace" in payload
    lt = payload["latency_trace"]
    assert lt["total_ms"] >= 0
    assert len(lt["spans"]) == 3
    names = [s["phase"] for s in lt["spans"]]
    assert names[0] == "phase_a"
    assert names[1] == "phase_b"
    assert names[2] == "response_finalize"
    assert lt["spans"][1]["ms"] >= 0


def test_trace_noop_without_init():
    slot: dict = {}
    trace_span(slot, "x")
    payload: dict = {}
    trace_attach_to_payload(slot, payload)
    assert "latency_trace" not in payload


@pytest.mark.parametrize(
    "env_val,expected",
    [
        ("1", True),
        ("true", True),
        ("yes", True),
        ("0", False),
    ],
)
def test_trace_requested_env(monkeypatch, env_val, expected):
    monkeypatch.setenv("HAROMA_CHAT_TRACE", env_val)
    assert trace_requested() is expected


def test_trace_requested_unset(monkeypatch):
    monkeypatch.delenv("HAROMA_CHAT_TRACE", raising=False)
    assert trace_requested() is False


def test_trace_log_requested_env(monkeypatch):
    monkeypatch.setenv("HAROMA_CHAT_TRACE_LOG", "1")
    assert trace_log_requested() is True
