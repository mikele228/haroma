"""BackgroundCadence + RuntimeSignals wiring."""

from __future__ import annotations

import sys
import time

import pytest

_REPO = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from agents.background_cadence import BackgroundCadence
from agents.runtime_signals import RuntimeSignals
from engine.WebLearnCrawler import WebLearnCrawler


class _Sh:
    def __init__(self):
        self._inflight = 0
        import threading

        self._http_chat_lock = threading.Lock()
        self.signals = RuntimeSignals(self)

    @property
    def http_chat_inflight(self):
        with self._http_chat_lock:
            return self._inflight

    def http_chat_begin(self, depth=None):
        with self._http_chat_lock:
            self._inflight += 1
            self.signals.append_depth_under_http_lock(depth)

    def http_chat_end(self):
        with self._http_chat_lock:
            if self._inflight > 0:
                self._inflight -= 1
                self.signals.pop_depth_under_http_lock()


def test_cadence_training_when_no_chat(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_BG_DEFER_TRAINING_ON_HTTP_CHAT", "1")
    monkeypatch.delenv("HAROMA_BG_DEFER_TRAINING_CAP_SEC", raising=False)
    s = _Sh()
    c = BackgroundCadence(s)
    assert c.should_run_training_now() is True


def test_cadence_defers_when_chat_inflight(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_BG_DEFER_TRAINING_ON_HTTP_CHAT", "1")
    monkeypatch.delenv("HAROMA_BG_DEFER_TRAINING_CAP_SEC", raising=False)
    s = _Sh()
    s.http_chat_begin("normal")
    c = BackgroundCadence(s)
    assert c.should_run_training_now() is False
    s.http_chat_end()
    assert c.should_run_training_now() is True


def test_cadence_cap_bypasses_defer(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_BG_DEFER_TRAINING_ON_HTTP_CHAT", "1")
    monkeypatch.setenv("HAROMA_BG_DEFER_TRAINING_CAP_SEC", "0.05")
    s = _Sh()
    s.signals.last_background_training_at = time.time() - 1.0
    s.http_chat_begin("normal")
    c = BackgroundCadence(s)
    assert c.should_run_training_now() is True


def test_should_run_web_learn():
    s = _Sh()
    c = BackgroundCadence(s)
    wl = WebLearnCrawler({"enabled": True, "every_n_ticks": 3, "seed_urls": [], "allowed_hosts": []})
    assert c.should_run_web_learn(0, wl) is False
    assert c.should_run_web_learn(3, wl) is True
