"""Tests for :mod:`mind.packed_llm_inputs`."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind.packed_llm_inputs import should_skip_full_pack_messages_for_llm


def test_skip_when_chat_only(monkeypatch):
    monkeypatch.setenv("HAROMA_LLM_CHAT_ONLY", "1")
    monkeypatch.delenv("HAROMA_LLM_DUMMY_REPLY", raising=False)
    assert should_skip_full_pack_messages_for_llm() is True


def test_skip_when_dummy_without_full_pack(monkeypatch):
    monkeypatch.delenv("HAROMA_LLM_CHAT_ONLY", raising=False)
    monkeypatch.setenv("HAROMA_LLM_DUMMY_REPLY", "1")
    monkeypatch.setenv("HAROMA_LLM_DUMMY_FULL_PACK", "0")
    assert should_skip_full_pack_messages_for_llm() is True


def test_no_skip_when_full_pack_dummy(monkeypatch):
    monkeypatch.delenv("HAROMA_LLM_CHAT_ONLY", raising=False)
    monkeypatch.setenv("HAROMA_LLM_DUMMY_REPLY", "1")
    monkeypatch.setenv("HAROMA_LLM_DUMMY_FULL_PACK", "1")
    assert should_skip_full_pack_messages_for_llm() is False
