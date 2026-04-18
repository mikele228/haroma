"""``run_llm_context_reasoning_phase`` skipped path respects ``bind_episode``."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind.cycle_flow import TRACE_LABEL_PERSONA_PACKED_LLM, run_llm_context_reasoning_phase


def test_skipped_does_not_bind_when_bind_episode_false():
    binds = []

    class Ep:
        def bind_llm_context(self, d):
            binds.append(dict(d))

    ep = Ep()
    out = run_llm_context_reasoning_phase(
        enabled=False,
        llm_backend=None,
        user_text="hi",
        recalled_memories=[],
        identity_summary={},
        personality_summary={},
        active_goals=[],
        law_summary={},
        value_summary={},
        episode=ep,
        bind_episode=False,
    )
    assert out.get("source") == "skipped"
    assert binds == []


def test_trace_label_constant_matches_persona_default():
    assert TRACE_LABEL_PERSONA_PACKED_LLM == "13.2b.llm_context_reasoning"


def test_skipped_binds_when_bind_episode_true():
    binds = []

    class Ep:
        def bind_llm_context(self, d):
            binds.append(dict(d))

    ep = Ep()
    run_llm_context_reasoning_phase(
        enabled=False,
        llm_backend=None,
        user_text="hi",
        recalled_memories=[],
        identity_summary={},
        personality_summary={},
        active_goals=[],
        law_summary={},
        value_summary={},
        episode=ep,
        bind_episode=True,
    )
    assert len(binds) == 1
    assert binds[0].get("source") == "skipped"
