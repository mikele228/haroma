"""Opt-in packed LLM for :class:`~mind.control.ElarionController`."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind.packed_llm_controller_bridge import (
    controller_packed_llm_enabled,
    run_packed_llm_phase_for_elarion_controller,
)


def test_controller_packed_llm_enabled_env(monkeypatch):
    monkeypatch.delenv("HAROMA_CONTROLLER_PACKED_LLM", raising=False)
    assert controller_packed_llm_enabled() is False
    monkeypatch.setenv("HAROMA_CONTROLLER_PACKED_LLM", "1")
    assert controller_packed_llm_enabled() is True


def test_run_packed_llm_skips_when_env_off(monkeypatch):
    monkeypatch.delenv("HAROMA_CONTROLLER_PACKED_LLM", raising=False)
    ctrl = MagicMock()
    episode = MagicMock()
    episode.recalled_memories = []
    out = run_packed_llm_phase_for_elarion_controller(
        ctrl,
        episode,
        role="observer",
        content="hi",
        text="hi",
        has_external=True,
        gate_reasoning=True,
        reasoning_result={},
        appraisal_result={},
        active_goals=[],
        identity_summary={},
        nlu_result={},
        knowledge_summary={},
        cycle_id=1,
    )
    assert out.get("source") == "skipped"


def test_run_packed_llm_invokes_phase_when_env_on(monkeypatch):
    monkeypatch.setenv("HAROMA_CONTROLLER_PACKED_LLM", "1")
    monkeypatch.setenv("HAROMA_LLM_CENTRIC", "1")

    ctrl = MagicMock()
    ctrl.knowledge = MagicMock()
    ctrl.law = MagicMock()
    ctrl.law.summarize.return_value = {}
    ctrl.value = MagicMock()
    ctrl.value.summarize.return_value = {}
    ctrl.conversation = MagicMock()
    ctrl.conversation.is_in_conversation.return_value = False
    ctrl.memory = MagicMock()
    ctrl.memory.build_seed_context = MagicMock(return_value="")
    ctrl.llm_backend = None

    episode = MagicMock()
    episode.recalled_memories = []
    episode.drives = {}
    episode.affect = {}

    called = {}

    def _fake_invoke(**kwargs):
        called["kwargs"] = kwargs
        return {"source": "dummy_probe", "answer": "ok"}

    monkeypatch.setattr(
        "mind.packed_llm_controller_bridge.invoke_run_llm_context_reasoning_phase",
        _fake_invoke,
    )

    out = run_packed_llm_phase_for_elarion_controller(
        ctrl,
        episode,
        role="observer",
        content="hello",
        text="hello",
        has_external=True,
        gate_reasoning=True,
        reasoning_result={"steps": []},
        appraisal_result={},
        active_goals=[],
        identity_summary={"essence_name": "Test"},
        nlu_result={},
        knowledge_summary={"entity_count": 0},
        cycle_id=3,
    )
    assert out["answer"] == "ok"
    assert called["kwargs"]["pl"].path.llm_ctx_enabled is True
    assert called["kwargs"]["pl"].user_text == "hello"
