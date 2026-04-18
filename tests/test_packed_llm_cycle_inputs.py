"""Tests for :mod:`mind.packed_llm_cycle_inputs`."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind.packed_llm_cycle_inputs import build_packed_llm_cycle_inputs


def test_build_produces_skipped_llm_when_no_user_text(monkeypatch):
    monkeypatch.setenv("HAROMA_LLM_CHAT_ONLY", "0")
    monkeypatch.delenv("HAROMA_LLM_DUMMY_REPLY", raising=False)
    out = build_packed_llm_cycle_inputs(
        text="",
        content="   \n",
        llm_centric=False,
        chat_llm_primary=False,
        role="conversant",
        has_external=True,
        is_internal=False,
        trueself_agent=False,
        user_or_traced_turn=True,
        gate_reasoning=True,
        over_budget=False,
        deliberative_flag=False,
        recalled_memories=[],
        reasoning_result={},
        appraisal_result={},
        organic_skip_threshold=0.9,
        knowledge_summary={},
        knowledge_graph=MagicMock(),
        nlu_result=None,
    )
    assert out.user_message is False
    assert out.path.llm_ctx_enabled is False
    assert out.kg_triples is None


def test_defer_bind_when_deliberative_and_llm_on(monkeypatch):
    monkeypatch.setenv("HAROMA_LLM_CHAT_ONLY", "0")
    monkeypatch.delenv("HAROMA_LLM_DUMMY_REPLY", raising=False)
    out = build_packed_llm_cycle_inputs(
        text="hi",
        content="",
        llm_centric=True,
        chat_llm_primary=False,
        role="conversant",
        has_external=True,
        is_internal=False,
        trueself_agent=False,
        user_or_traced_turn=True,
        gate_reasoning=True,
        over_budget=False,
        deliberative_flag=True,
        recalled_memories=[],
        reasoning_result={},
        appraisal_result={},
        organic_skip_threshold=0.9,
        knowledge_summary={},
        knowledge_graph=MagicMock(),
        nlu_result=None,
    )
    assert out.path.llm_ctx_enabled is True
    assert out.defer_episode_bind is True
