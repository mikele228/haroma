"""``mind.prompt_packaging`` re-exports match ``engine.LLMContextReasoner``."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine.LLMContextReasoner as _eng
import mind.prompt_packaging as pp


def test_reexports_are_engine_identity():
    assert pp.run_llm_context_reasoning is _eng.run_llm_context_reasoning
    assert pp.build_messages is _eng.build_messages
    assert pp.packed_messages_stats is _eng.packed_messages_stats
    assert pp.parse_response is _eng.parse_response
    assert pp.LLMContextResult is _eng.LLMContextResult
    assert pp.packed_llm_timeout_seconds is _eng._llm_context_timeout_seconds
