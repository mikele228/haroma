"""Tests for :mod:`mind.packed_llm_reasoning_kwargs`."""

from __future__ import annotations

import inspect
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind.cycle_flow import run_llm_context_reasoning_phase
from mind.packed_llm_reasoning_kwargs import build_persona_packed_llm_reasoning_phase_kwargs


def test_builder_keys_match_run_llm_context_reasoning_phase():
    sig = inspect.signature(run_llm_context_reasoning_phase)
    param_names = set(sig.parameters.keys())
    kw = build_persona_packed_llm_reasoning_phase_kwargs(
        enabled=False,
        llm_backend=None,
        user_text="",
        recalled_memories=[],
        identity_summary={},
        personality_summary={},
        active_goals=[],
        law_summary={},
        value_summary={},
        knowledge_triples=None,
        discourse_context="",
        nlu_result=None,
        memory_forest_seed="",
        llm_centric=False,
        episode=None,
        memory_forest=None,
        trace_label="13.2b.llm_context_reasoning",
        timeout_override=None,
        deliberative=False,
        agent_state_json="",
        bind_episode=True,
    )
    assert set(kw.keys()) == param_names, (
        f"kwargs/phase drift: only in phase {param_names - set(kw)} "
        f"only in builder {set(kw) - param_names}"
    )


def test_skipped_call_via_unpack(monkeypatch):
    monkeypatch.delenv("HAROMA_LLM_DUMMY_REPLY", raising=False)
    out = run_llm_context_reasoning_phase(
        **build_persona_packed_llm_reasoning_phase_kwargs(
            enabled=False,
            llm_backend=None,
            user_text="x",
            recalled_memories=[],
            identity_summary={},
            personality_summary={},
            active_goals=[],
            law_summary={},
            value_summary={},
            knowledge_triples=None,
            discourse_context="",
            nlu_result=None,
            memory_forest_seed="",
            llm_centric=False,
            episode=None,
            memory_forest=None,
            trace_label="t",
            timeout_override=None,
            deliberative=False,
            agent_state_json="",
            bind_episode=True,
        )
    )
    assert out.get("source") == "skipped"
