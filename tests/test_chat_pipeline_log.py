"""mind.chat_pipeline_log — timing state and helpers."""

from __future__ import annotations

import os
import sys

import pytest

_REPO = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def test_full_alias_enables_timing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_CHAT_PIPELINE_LOG", "full")
    monkeypatch.delenv("HAROMA_CHAT_PIPELINE_TIMING", raising=False)
    import importlib

    import mind.chat_pipeline_log as cpl

    importlib.reload(cpl)
    assert cpl.chat_pipeline_log_enabled() is True
    assert cpl.pipeline_timing_enabled() is True


def test_trace_end_allows_fresh_timing_same_trace_id(monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setenv("HAROMA_CHAT_PIPELINE_LOG", "1")
    monkeypatch.setenv("HAROMA_CHAT_PIPELINE_TIMING", "1")
    import importlib

    import mind.chat_pipeline_log as cpl

    importlib.reload(cpl)

    tid = "t_test_reuse"
    cpl.log_chat_pipeline("first", trace_id=tid)
    cpl.pipeline_trace_end(tid)
    cpl.log_chat_pipeline("after_end", trace_id=tid)
    out = capsys.readouterr().out
    last = [ln for ln in out.splitlines() if "after_end" in ln][-1]
    assert "seg=0.00ms cum=0.00ms" in last
