"""Tests for :mod:`mind.packed_llm_centric`."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind.packed_llm_centric import llm_centric_enabled_for_persona_cycle


def test_off_when_env_disabled(monkeypatch):
    monkeypatch.setenv("HAROMA_LLM_CENTRIC", "0")
    assert (
        llm_centric_enabled_for_persona_cycle(
            is_internal=False,
            trueself_agent=False,
            user_or_traced_turn=True,
            goal_board=None,
        )
        is False
    )


def test_off_when_internal(monkeypatch):
    monkeypatch.setenv("HAROMA_LLM_CENTRIC", "1")
    assert (
        llm_centric_enabled_for_persona_cycle(
            is_internal=True,
            trueself_agent=False,
            user_or_traced_turn=True,
            goal_board=None,
        )
        is False
    )


def test_off_when_mandate_active(monkeypatch):
    monkeypatch.setenv("HAROMA_LLM_CENTRIC", "1")
    gb = MagicMock()
    gb.has_active_mandate.return_value = True
    assert (
        llm_centric_enabled_for_persona_cycle(
            is_internal=False,
            trueself_agent=False,
            user_or_traced_turn=True,
            goal_board=gb,
        )
        is False
    )


def test_trueself_requires_traced_turn(monkeypatch):
    monkeypatch.setenv("HAROMA_LLM_CENTRIC", "1")
    assert (
        llm_centric_enabled_for_persona_cycle(
            is_internal=False,
            trueself_agent=True,
            user_or_traced_turn=False,
            goal_board=None,
        )
        is False
    )


def test_on_when_configured(monkeypatch):
    monkeypatch.setenv("HAROMA_LLM_CENTRIC", "1")
    assert (
        llm_centric_enabled_for_persona_cycle(
            is_internal=False,
            trueself_agent=False,
            user_or_traced_turn=True,
            goal_board=None,
        )
        is True
    )
