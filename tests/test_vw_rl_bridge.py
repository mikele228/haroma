"""Tests for optional Vowpal Wabbit / RLlib transition logging (no heavy deps required)."""

from __future__ import annotations

import json
import os
import tempfile


def test_rllib_logger_writes_jsonl(monkeypatch):
    from mind.training.vw_rl_bridge import RLlibTransitionLogger

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "t.jsonl")
        monkeypatch.setenv("HAROMA_RLLIB_LOG_TRANSITIONS", "1")
        monkeypatch.setenv("HAROMA_RLLIB_TRANSITIONS_PATH", path)
        log = RLlibTransitionLogger()
        assert log.enabled
        log.record("hello", "world", 0.7, metadata={"x": 1})
        with open(path, encoding="utf-8") as f:
            line = f.readline()
        row = json.loads(line)
        assert row["reward"] == 0.7
        assert row["obs"] == "hello"
        assert row["action"] == "world"
        assert row["done"] is True


def test_rllib_logger_basename_only_path_no_makedirs_crash(monkeypatch, tmp_path):
    """``HAROMA_RLLIB_TRANSITIONS_PATH=foo.jsonl`` has empty dirname; must not call os.makedirs('')."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HAROMA_RLLIB_LOG_TRANSITIONS", "1")
    monkeypatch.setenv("HAROMA_RLLIB_TRANSITIONS_PATH", "bandit.jsonl")
    from mind.training.vw_rl_bridge import RLlibTransitionLogger

    log = RLlibTransitionLogger()
    log.record("a", "b", 0.5)
    out = tmp_path / "bandit.jsonl"
    assert out.is_file()
    assert json.loads(out.read_text(encoding="utf-8"))["reward"] == 0.5


def test_vw_trainer_disabled_by_default():
    from mind.training.vw_rl_bridge import VowpalWabbitRewardTrainer

    t = VowpalWabbitRewardTrainer()
    assert not t.available


def test_vw_trainer_record_noop_when_unavailable(monkeypatch):
    monkeypatch.delenv("HAROMA_VW_REWARD", raising=False)
    from mind.training.vw_rl_bridge import VowpalWabbitRewardTrainer

    t = VowpalWabbitRewardTrainer()
    t.record("a", "b", 0.5)
    assert t.train_step() == 0.0
