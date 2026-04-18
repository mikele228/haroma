"""Tests for :mod:`mind.packed_llm_kg`."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind.packed_llm_kg import kg_triples_for_packed_llm_prompt


def test_returns_none_when_llm_ctx_disabled():
    assert (
        kg_triples_for_packed_llm_prompt(
            llm_ctx_enabled=False,
            skip_full_pack_messages=False,
            knowledge_summary={},
            knowledge_graph=MagicMock(),
            nlu_result={"entities": [{"text": "x"}]},
        )
        is None
    )


def test_returns_none_when_skip_full_pack():
    assert (
        kg_triples_for_packed_llm_prompt(
            llm_ctx_enabled=True,
            skip_full_pack_messages=True,
            knowledge_summary={},
            knowledge_graph=MagicMock(),
            nlu_result=None,
        )
        is None
    )


@patch("engine.LanguageComposer.LanguageComposer.select_relevant_triples", return_value=[("s", "p", "o")])
def test_calls_select_relevant_triples(_mock_sel):
    kg = MagicMock()
    out = kg_triples_for_packed_llm_prompt(
        llm_ctx_enabled=True,
        skip_full_pack_messages=False,
        knowledge_summary={"top_entities": ["A"]},
        knowledge_graph=kg,
        nlu_result={"entities": [{"text": "n1"}]},
    )
    assert out == [("s", "p", "o")]
