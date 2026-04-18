"""Tests for :mod:`mind.packed_llm_pipeline_log`."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind.packed_llm_pipeline_log import packed_llm_before_llm_log_detail


def test_log_detail_includes_agent_and_dummy_env(monkeypatch):
    monkeypatch.setenv("HAROMA_LLM_DUMMY_REPLY", "1")
    s = packed_llm_before_llm_log_detail(
        agent_id="primary",
        role="conversant",
        llm_ctx_enabled=True,
    )
    assert "agent=primary" in s
    assert "role=conversant" in s
    assert "llm_enabled=True" in s
    assert "dummy_env=True" in s
    assert "HAROMA_LLM_DUMMY_REPLY=" in s
