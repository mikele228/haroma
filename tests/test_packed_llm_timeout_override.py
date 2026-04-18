"""Tests for :mod:`mind.packed_llm_timeout_override`."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind.packed_llm_timeout_override import packed_llm_generate_chat_timeout_override


def test_fast_llm_timeout_sec_wins(monkeypatch):
    monkeypatch.delenv("HAROMA_FAST_LLM_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("HAROMA_FAST_LLM_DEFAULT_TIMEOUT_SEC", raising=False)
    monkeypatch.setenv("HAROMA_FAST_LLM_TIMEOUT_SEC", "45")
    assert packed_llm_generate_chat_timeout_override(apply_true_self_user_chat_cap=False) == 45.0


def test_default_timeout_when_fast_unset(monkeypatch):
    monkeypatch.delenv("HAROMA_FAST_LLM_TIMEOUT_SEC", raising=False)
    monkeypatch.setenv("HAROMA_FAST_LLM_DEFAULT_TIMEOUT_SEC", "200")
    assert packed_llm_generate_chat_timeout_override(apply_true_self_user_chat_cap=False) == 200.0


def test_true_self_cap_min_with_fast_default(monkeypatch):
    monkeypatch.delenv("HAROMA_FAST_LLM_TIMEOUT_SEC", raising=False)
    monkeypatch.setenv("HAROMA_FAST_LLM_DEFAULT_TIMEOUT_SEC", "600")
    monkeypatch.setenv("HAROMA_TRUESELF_USER_CHAT_LLM_TIMEOUT_SEC", "8")
    assert packed_llm_generate_chat_timeout_override(apply_true_self_user_chat_cap=True) == 8.0


def test_true_self_ignored_when_cap_disabled(monkeypatch):
    monkeypatch.delenv("HAROMA_FAST_LLM_TIMEOUT_SEC", raising=False)
    monkeypatch.setenv("HAROMA_FAST_LLM_DEFAULT_TIMEOUT_SEC", "600")
    monkeypatch.setenv("HAROMA_TRUESELF_USER_CHAT_LLM_TIMEOUT_SEC", "8")
    assert packed_llm_generate_chat_timeout_override(apply_true_self_user_chat_cap=False) == 600.0


def test_invalid_fast_sec_falls_back_then_true_self(monkeypatch):
    monkeypatch.setenv("HAROMA_FAST_LLM_TIMEOUT_SEC", "not-a-number")
    monkeypatch.setenv("HAROMA_FAST_LLM_DEFAULT_TIMEOUT_SEC", "100")
    monkeypatch.setenv("HAROMA_TRUESELF_USER_CHAT_LLM_TIMEOUT_SEC", "8")
    assert packed_llm_generate_chat_timeout_override(apply_true_self_user_chat_cap=True) == 8.0
