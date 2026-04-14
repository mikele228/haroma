"""Smoke tests for LLM training hooks (record / train_reward_model) and query (generate_chat).

Keeps ``use_programmed=True`` so no GGUF/API is required."""

from __future__ import annotations

import json
import os


def test_programmed_backend_generate_chat_returns_text():
    from engine.LLMBackend import LLMBackend

    b = LLMBackend(use_programmed=True)
    out = b.generate_chat([{"role": "user", "content": "Hello."}], max_tokens=64)
    assert isinstance(out, str)
    assert len(out.strip()) > 0


def test_record_outcome_train_reward_no_crash_programmed():
    from engine.LLMBackend import LLMBackend

    b = LLMBackend(use_programmed=True)
    b.record_outcome("prompt text", "response text", 0.55)
    loss = b.train_reward_model()
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_record_outcome_writes_rllib_jsonl_when_enabled(monkeypatch, tmp_path):
    path = tmp_path / "tr.jsonl"
    monkeypatch.setenv("HAROMA_RLLIB_LOG_TRANSITIONS", "1")
    monkeypatch.setenv("HAROMA_RLLIB_TRANSITIONS_PATH", str(path))

    from engine.LLMBackend import LLMBackend

    b = LLMBackend(use_programmed=True)
    b.record_outcome("user asks", "bot says", 0.8, alignment_metadata=None)
    assert path.is_file()
    line = path.read_text(encoding="utf-8").strip()
    row = json.loads(line)
    assert row["reward"] == 0.8
    assert "user asks" in row["obs"]
    assert "bot says" in row["action"]


def test_stats_include_training_hook_keys():
    from engine.LLMBackend import LLMBackend

    b = LLMBackend(use_programmed=True)
    st = b.stats()
    assert "vw_trainer" in st
    assert "rllib_logger" in st
    assert "reward_model" in st


def test_env_chat_llm_primary_helper_importable():
    """Guard: ActionLoop chat-primary toggle used by HTTP path."""
    from core.ActionLoop import _env_chat_llm_primary

    os.environ.pop("HAROMA_CHAT_LLM_PRIMARY", None)
    assert _env_chat_llm_primary() is True
