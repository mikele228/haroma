"""mind.structured_log — optional JSON stderr lines."""

from __future__ import annotations

import sys

import pytest

_REPO = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def test_log_event_noop_when_disabled(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.delenv("HAROMA_STRUCTURED_LOG", raising=False)
    from mind.structured_log import log_event

    log_event("should_not_emit", x=1)
    assert capsys.readouterr().err == ""


def test_log_event_writes_json_when_enabled(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("HAROMA_STRUCTURED_LOG", "1")
    from mind.structured_log import log_event

    log_event("unit_evt", path="/chat", method="POST")
    err = capsys.readouterr().err
    assert "unit_evt" in err
    assert "/chat" in err
    assert '"event"' in err or "event" in err
