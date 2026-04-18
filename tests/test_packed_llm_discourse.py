"""Tests for :mod:`mind.packed_llm_discourse`."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind.packed_llm_discourse import (
    discourse_context_for_packed_llm,
    merge_first_encounter_discourse_hint,
)


def test_noop_when_gate_off():
    assert merge_first_encounter_discourse_hint("ctx", False) == "ctx"


def test_hint_only_when_base_empty():
    out = merge_first_encounter_discourse_hint("", True)
    assert "First exchange" in out
    assert " | " not in out


def test_appends_with_separator_when_base_nonempty():
    out = merge_first_encounter_discourse_hint("hello", True)
    assert out.startswith("hello | ")
    assert "First exchange" in out


def test_discourse_context_uses_conversation_summary():
    conv = MagicMock()
    conv.is_in_conversation.return_value = True
    conv.get_context_summary.return_value = "thread"
    out = discourse_context_for_packed_llm(
        conv,
        cycle_id=3,
        speaker_key="sk",
        session_uid=True,
        first_encounter_asks_name=False,
    )
    assert out == "thread"
    conv.is_in_conversation.assert_called_once_with(3)
    conv.get_context_summary.assert_called_once_with("sk")


def test_discourse_context_session_uid_false_passes_none_key():
    conv = MagicMock()
    conv.is_in_conversation.return_value = True
    discourse_context_for_packed_llm(
        conv,
        cycle_id=1,
        speaker_key="sk",
        session_uid=False,
        first_encounter_asks_name=False,
    )
    conv.get_context_summary.assert_called_once_with(None)


def test_discourse_context_not_in_conversation_empty_base():
    conv = MagicMock()
    conv.is_in_conversation.return_value = False
    out = discourse_context_for_packed_llm(
        conv,
        cycle_id=1,
        speaker_key="sk",
        session_uid=True,
        first_encounter_asks_name=True,
    )
    assert "First exchange" in out
    conv.get_context_summary.assert_not_called()
