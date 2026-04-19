"""Tests for :func:`mind.chat_priority.background_input_active`."""

from __future__ import annotations

import os
from unittest import mock

from mind.chat_priority import background_input_active


class _IAStub:
    def __init__(self, stats, peek=None):
        self._stats = stats
        self._peek = peek

    def buffer_stats(self):
        return dict(self._stats)

    def peek_sensor_queue(self, limit=64):
        if self._peek is None:
            return []
        return list(self._peek)[:limit]


class _SharedStub:
    def __init__(self, http=0, ia=None, env=None):
        self.http_chat_inflight = http
        self._input_agent_ref = ia
        self._env = env or {}

    def get_agent_environment_snapshot(self):
        return dict(self._env)


def test_active_when_http_chat_inflight():
    s = _SharedStub(http=1)
    assert background_input_active(s, None) is True


def test_active_when_text_pending():
    s = _SharedStub(ia=_IAStub({"text_pending": 2, "text_priority_pending": 0, "sensor_pending": 0}))
    assert background_input_active(s, None) is True


def test_inactive_when_only_sensor_pending():
    s = _SharedStub(
        ia=_IAStub({"text_pending": 0, "text_priority_pending": 0, "sensor_pending": 9}),
    )
    assert background_input_active(s, None) is False


def test_sensor_pending_active_when_env_legacy():
    s = _SharedStub(
        ia=_IAStub({"text_pending": 0, "text_priority_pending": 0, "sensor_pending": 1}),
    )
    with mock.patch.dict(os.environ, {"HAROMA_BG_ACTIVE_INPUT_INCLUDES_SENSOR_PENDING": "1"}):
        assert background_input_active(s, None) is True


def test_active_when_agent_environment_alerts():
    s = _SharedStub(
        env={
            "alerts": [{"kind": "collision"}],
            "metrics": {},
            "extensions": {},
        }
    )
    assert background_input_active(s, None) is True


def test_active_when_speech_in_sensor_queue():
    ia = _IAStub(
        {"text_pending": 0, "text_priority_pending": 0, "sensor_pending": 1},
        peek=[
            {"channel": "audio", "data": {"is_speech_likely": True, "rms_level": 0.001}},
        ],
    )
    s = _SharedStub(ia=ia)
    assert background_input_active(s, None) is True


def test_boot_agent_fallback_for_input_agent():
    ia = _IAStub({"text_pending": 1, "text_priority_pending": 0, "sensor_pending": 0})
    s = _SharedStub(ia=None)
    ba = mock.Mock()
    ba.input_agent = ia
    assert background_input_active(s, ba) is True
