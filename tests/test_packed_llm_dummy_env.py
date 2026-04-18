"""mind.packed_llm_dummy_env — HAROMA_LLM_DUMMY_REPLY truthy alignment."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind.packed_llm_dummy_env import (
    packed_llm_dummy_probe_active,
    packed_llm_dummy_reply_raw,
    synthetic_llm_dummy_reply_env,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("TRUE", True),
        ("0", False),
        ("", False),
    ],
)
def test_synthetic_env(monkeypatch, raw, expected):
    if raw == "":
        monkeypatch.delenv("HAROMA_LLM_DUMMY_REPLY", raising=False)
    else:
        monkeypatch.setenv("HAROMA_LLM_DUMMY_REPLY", raw)
    assert synthetic_llm_dummy_reply_env() is expected
    assert packed_llm_dummy_probe_active() is expected


def test_packed_llm_dummy_reply_raw(monkeypatch):
    monkeypatch.setenv("HAROMA_LLM_DUMMY_REPLY", "maybe")
    assert packed_llm_dummy_reply_raw() == "maybe"
