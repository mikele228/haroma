"""mind.chat_priority.input_pipeline_busy"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

_REPO = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def test_input_pipeline_busy_http_inflight():
    from mind.chat_priority import input_pipeline_busy

    s = SimpleNamespace(http_chat_inflight=1, _input_agent_ref=None)
    assert input_pipeline_busy(s, None) is True


def test_input_pipeline_busy_input_agent_queues(monkeypatch: pytest.MonkeyPatch):
    from mind.chat_priority import input_pipeline_busy

    monkeypatch.delenv("HAROMA_CHAT_INPUT_PRIORITY", raising=False)

    class _IA:
        def buffer_stats(self):
            return {
                "text_pending": 0,
                "text_priority_pending": 2,
                "sensor_pending": 0,
            }

    s = SimpleNamespace(http_chat_inflight=0, _input_agent_ref=_IA())
    assert input_pipeline_busy(s, None) is True


def test_input_pipeline_busy_via_boot_agent(monkeypatch: pytest.MonkeyPatch):
    from mind.chat_priority import input_pipeline_busy

    class _IA:
        def buffer_stats(self):
            return {"text_pending": 0, "text_priority_pending": 0, "sensor_pending": 1}

    boot = SimpleNamespace(input_agent=_IA())
    s = SimpleNamespace(http_chat_inflight=0, _input_agent_ref=None)
    assert input_pipeline_busy(s, boot) is True


def test_input_pipeline_yield_busy_ignores_sensor_only():
    from mind.chat_priority import input_pipeline_yield_busy

    class _IA:
        def buffer_stats(self):
            return {"text_pending": 0, "text_priority_pending": 0, "sensor_pending": 99}

    s = SimpleNamespace(http_chat_inflight=0, _input_agent_ref=_IA())
    assert input_pipeline_yield_busy(s, None) is False


def test_input_pipeline_yield_busy_http_inflight():
    from mind.chat_priority import input_pipeline_yield_busy

    class _IA:
        def buffer_stats(self):
            return {"text_pending": 0, "text_priority_pending": 0, "sensor_pending": 99}

    s = SimpleNamespace(http_chat_inflight=1, _input_agent_ref=_IA())
    assert input_pipeline_yield_busy(s, None) is True


def test_input_pipeline_busy_idle():
    from mind.chat_priority import input_pipeline_busy

    class _IA:
        def buffer_stats(self):
            return {"text_pending": 0, "text_priority_pending": 0, "sensor_pending": 0}

    s = SimpleNamespace(http_chat_inflight=0, _input_agent_ref=_IA())
    assert input_pipeline_busy(s, None) is False


def test_input_pipeline_busy_buffer_stats_failure_is_busy():
    """Unreadable queue stats must not report idle (would starve input priority)."""

    from mind.chat_priority import input_pipeline_busy

    class _IA:
        def buffer_stats(self):
            raise RuntimeError("simulated")

    s = SimpleNamespace(http_chat_inflight=0, _input_agent_ref=_IA())
    assert input_pipeline_busy(s, None) is True


def test_chat_input_priority_defer_on_pipeline_exception(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_CHAT_INPUT_PRIORITY", "1")

    from mind import chat_priority as cp

    def _boom(*_a, **_k):
        raise RuntimeError("simulated")

    monkeypatch.setattr(cp, "input_pipeline_busy", _boom)
    assert cp.chat_input_priority_defer_non_user(SimpleNamespace(), None) is True
