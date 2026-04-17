"""Tests for mind.lock_budget and SharedResources neural lock timing."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from mind.cognitive_observability import CognitiveMetrics
from mind.lock_budget import (
    report_shared_lock_hold,
    report_shared_lock_section,
    shared_lock_budget_sec,
)


def test_shared_lock_budget_sec_default():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert shared_lock_budget_sec() == 1.0


def test_shared_lock_budget_sec_disabled():
    with mock.patch.dict(os.environ, {"HAROMA_SHARED_LOCK_BUDGET_SEC": "0"}):
        assert shared_lock_budget_sec() is None


def test_report_shared_lock_hold_noop_when_under_budget():
    with mock.patch.dict(
        os.environ,
        {"HAROMA_SHARED_LOCK_BUDGET_SEC": "2.0", "HAROMA_SHARED_LOCK_BUDGET_MODE": "assert"},
    ):
        report_shared_lock_hold("test", 0.5)  # does not raise


def test_report_shared_lock_hold_assert_over_budget():
    with mock.patch.dict(
        os.environ,
        {"HAROMA_SHARED_LOCK_BUDGET_SEC": "0.01", "HAROMA_SHARED_LOCK_BUDGET_MODE": "assert"},
    ):
        with pytest.raises(AssertionError, match="SharedLockBudget"):
            report_shared_lock_hold("test", 1.0)


def test_budget_uses_hold_not_wait_for_assert():
    """Long wait with short hold must not assert (hold is what we budget)."""
    with mock.patch.dict(
        os.environ,
        {"HAROMA_SHARED_LOCK_BUDGET_SEC": "0.5", "HAROMA_SHARED_LOCK_BUDGET_MODE": "assert"},
    ):
        report_shared_lock_section(
            "test",
            wait_sec=99.0,
            hold_sec=0.01,
            cognitive_metrics=None,
        )


def test_cognitive_metrics_increments_on_over_budget():
    m = CognitiveMetrics()
    with mock.patch.dict(os.environ, {"HAROMA_SHARED_LOCK_BUDGET_SEC": "0.1"}):
        report_shared_lock_section(
            "neural_read",
            wait_sec=0.0,
            hold_sec=1.0,
            cognitive_metrics=m,
        )
    assert m.shared_lock_over_budget["neural_read"] == 1


def test_neural_sync_reports_hold(monkeypatch, capsys):
    from agents.shared_resources import SharedResources

    monkeypatch.setenv("HAROMA_SHARED_LOCK_BUDGET_SEC", "0")
    monkeypatch.setenv("HAROMA_SHARED_LOCK_BUDGET_MODE", "warn")

    sh = SharedResources()
    sh.initialize = lambda: None  # type: ignore[attr-defined]

    with sh.neural_sync():
        pass
    out = capsys.readouterr().out
    assert "SharedLockBudget" not in out


def test_persona_neural_section_instruments_once(monkeypatch):
    """persona_neural_section should not nest neural_sync (single timing report)."""
    from agents.shared_resources import SharedResources

    calls: list[str] = []

    def fake_section(
        name: str,
        *,
        wait_sec: float,
        hold_sec: float,
        cognitive_metrics=None,
    ) -> None:
        calls.append(name)

    monkeypatch.setenv("HAROMA_SHARED_LOCK_BUDGET_SEC", "0")
    monkeypatch.setattr(
        "agents.shared_resources.report_shared_lock_section",
        fake_section,
    )

    sh = SharedResources()
    sh.initialize = lambda: None  # type: ignore[attr-defined]

    with sh.persona_neural_section():
        pass
    assert calls == ["persona_neural_section"]


def test_background_train_round_robin(monkeypatch):
    from types import SimpleNamespace

    from agents.background_agent import BackgroundAgent
    from agents.message_bus import MessageBus

    monkeypatch.setenv("HAROMA_BG_MAX_TRAIN_MODULES_PER_TICK", "1")

    train_calls: list[str] = []

    def _fake_map(_s):
        return [
            ("a", lambda: train_calls.append("a") or 0.1),
            ("b", lambda: train_calls.append("b") or 0.1),
        ]

    monkeypatch.setattr(
        "agents.background_agent.build_background_train_map",
        _fake_map,
    )

    class _TS:
        def should_train(self, _name: str) -> bool:
            return True

        def record_loss(self, *_a, **_k) -> None:
            pass

    shared = SimpleNamespace(
        training_scheduler=_TS(),
        signals=mock.Mock(),
        agent_config={"background": {}},
    )

    from contextlib import contextmanager

    @contextmanager
    def _ntrain():
        yield

    shared.neural_train_sync = _ntrain  # type: ignore[attr-defined]

    bus = MessageBus()
    bg = BackgroundAgent(shared, bus, boot_agent=None, tick_interval=5.0)
    bg._cadence.should_run_training_now = mock.Mock(return_value=True)  # type: ignore[method-assign]

    bg._run_training()
    assert train_calls == ["a"]
    assert bg._bg_train_cursor == 1

    bg._run_training()
    assert train_calls == ["a", "b"]
    assert bg._bg_train_cursor == 0


def test_persona_merge_queue_schedule_and_drain():
    """Avoid full PersonaAgent.__init__ (heavy imports); exercise queue API only."""
    from agents.persona_agent import PersonaAgent

    merged: list[int] = []

    pa = PersonaAgent.__new__(PersonaAgent)
    pa.agent_id = "tmerge"
    pa._persona_merge_queue = []
    pa._persona_merge_lock = __import__("threading").Lock()

    PersonaAgent._schedule_persona_merge(pa, lambda: merged.append(1))
    PersonaAgent._drain_persona_merge_queue(pa)
    assert merged == [1]


def test_schedule_persona_merge_rejects_wrong_thread():
    import threading

    from agents.persona_agent import PersonaAgent

    pa = PersonaAgent.__new__(PersonaAgent)
    pa.agent_id = "tx"
    pa._persona_merge_queue = []
    pa._persona_merge_lock = threading.Lock()
    # Simulate an agent thread that is never the current thread (not started).
    pa._thread = threading.Thread(name="fake_persona_agent")

    with pytest.raises(RuntimeError, match="persona agent thread"):
        PersonaAgent._schedule_persona_merge(pa, lambda: None)
