"""Boot-time local LLM warmup (no GGUF load in CI)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.LLMBackend import LLMBackend


def test_warmup_skips_when_no_local_model():
    be = LLMBackend(use_programmed=True)
    assert be.warmup_local_inference() is False


def test_warmup_respects_disable_env(monkeypatch):
    monkeypatch.setenv("HAROMA_LLM_WARMUP", "0")
    be = LLMBackend.__new__(LLMBackend)
    be._model = object()
    assert LLMBackend.warmup_local_inference(be) is False
