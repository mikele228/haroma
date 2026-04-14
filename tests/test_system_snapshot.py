"""Unit tests for mind.system_snapshot (no Flask server)."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

_REPO = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mind.system_snapshot import build_http_status_payload, _resolve_http_inflight


def _minimal_boot(shared: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        shared=shared,
        stats=lambda: {"boot": 1},
        bus=SimpleNamespace(stats=lambda: {"bus": 2}),
        background_agent=None,
    )


def test_payload_core_keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_BG_DEFER_TRAINING_ON_HTTP_CHAT", "1")
    monkeypatch.delenv("HAROMA_BG_DEFER_TRAINING_CAP_SEC", raising=False)

    class _Sig:
            def snapshot(self):
                return {
                    "http_chat_inflight": 0,
                    "http_chat_depth_stack": [],
                    "last_background_training_at": 0.0,
                    "last_background_training_had_effect": False,
                }

    shared = SimpleNamespace(
        cycle_count=7,
        memory=SimpleNamespace(count_nodes=lambda: 3),
        organ_registry=SimpleNamespace(summary=lambda: {"o": 1}),
        symbolic_queue=SimpleNamespace(stats=lambda: {"q": 1}),
        fingerprint_engine=SimpleNamespace(stats=lambda: {"f": 1}),
        reconciliation=SimpleNamespace(stats=lambda: {"r": 1}),
        http_chat_inflight=0,
        agent_environment_status=lambda: {},
        signals=_Sig(),
    )
    boot = _minimal_boot(shared)
    reg = SimpleNamespace(pending_count=lambda: 0, _ttl_sec=30.0)
    out = build_http_status_payload(boot, None, reg)
    assert out["architecture"] == "multi-agent-v2"
    assert isinstance(out.get("health"), dict)
    assert out["health"].get("process") == "up"
    assert out["health"].get("llm_ready") is False
    assert isinstance(out.get("embodiment_readiness"), dict)
    assert out["lab_run_id"] is None
    assert out["cycle_count"] == 7
    assert out["memory_nodes"] == 3
    assert out["chat_async_pending"] == 0
    assert out["chat_async_ttl_sec"] == 30.0
    assert out["http_chat_inflight"] == 0
    assert out["bg_training_defer_enabled"] is True
    assert out["bg_training_deferred"] is False
    assert out["bg_training_defer_cap_sec"] is None
    assert out["runtime_signals"]["http_chat_inflight"] == 0
    assert "status_build_notes" not in out


def test_defer_fields_when_inflight(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_BG_DEFER_TRAINING_ON_HTTP_CHAT", "1")

    class _Sig:
            def snapshot(self):
                return {
                    "http_chat_inflight": 2,
                    "http_chat_depth_stack": ["normal"],
                    "last_background_training_at": 1.0,
                    "last_background_training_had_effect": True,
                }

    shared = SimpleNamespace(
        cycle_count=0,
        memory=SimpleNamespace(count_nodes=lambda: 0),
        organ_registry=SimpleNamespace(summary=lambda: {}),
        symbolic_queue=SimpleNamespace(stats=lambda: {}),
        fingerprint_engine=SimpleNamespace(stats=lambda: {}),
        reconciliation=SimpleNamespace(stats=lambda: {}),
        http_chat_inflight=2,
        agent_environment_status=lambda: {},
        signals=_Sig(),
    )
    boot = _minimal_boot(shared)
    out = build_http_status_payload(boot, None, None)
    assert out["http_chat_inflight"] == 2
    assert out["bg_training_deferred"] is True


def test_inflight_mismatch_emits_note(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HAROMA_BG_DEFER_TRAINING_CAP_SEC", raising=False)

    class _Sig:
        def snapshot(self):
            return {"http_chat_inflight": 1, "http_chat_depth_stack": []}

    shared = SimpleNamespace(
        cycle_count=0,
        memory=SimpleNamespace(count_nodes=lambda: 0),
        organ_registry=SimpleNamespace(summary=lambda: {}),
        symbolic_queue=SimpleNamespace(stats=lambda: {}),
        fingerprint_engine=SimpleNamespace(stats=lambda: {}),
        reconciliation=SimpleNamespace(stats=lambda: {}),
        http_chat_inflight=9,
        agent_environment_status=lambda: {},
        signals=_Sig(),
    )
    boot = _minimal_boot(shared)
    out = build_http_status_payload(boot, None, None)
    assert out["http_chat_inflight"] == 1
    notes = out.get("status_build_notes") or []
    assert any("mismatch" in n for n in notes)


def test_resolve_http_inflight_helpers():
    notes: list[str] = []
    sh = SimpleNamespace(http_chat_inflight=3)
    assert _resolve_http_inflight(sh, {"http_chat_inflight": 3}, notes) == 3
    assert notes == []

    notes.clear()
    sh2 = SimpleNamespace(http_chat_inflight=1)
    assert _resolve_http_inflight(sh2, {"http_chat_inflight": 2}, notes) == 2
    assert any("mismatch" in n for n in notes)
