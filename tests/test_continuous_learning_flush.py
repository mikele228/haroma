"""BackgroundAgent continuous-learning flush configuration (no full boot)."""

from __future__ import annotations

from unittest.mock import MagicMock

from agents.background_agent import BackgroundAgent
from agents.message_bus import MessageBus


def test_finetune_flush_interval_from_env(monkeypatch):
    monkeypatch.setenv("HAROMA_FINETUNE_FLUSH_TICKS", "7")
    monkeypatch.setenv("HAROMA_MEMORY_TRAINING_EXPORT_TICKS", "0")
    mock_shared = MagicMock()
    mock_shared.agent_config = {"background": {}}
    bus = MessageBus()
    bg = BackgroundAgent(mock_shared, bus, boot_agent=None, tick_interval=5.0)
    assert bg._finetune_flush_every == 7
    assert bg._memory_train_export_every == 0


def test_finetune_flush_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("HAROMA_FINETUNE_FLUSH_TICKS", raising=False)
    monkeypatch.delenv("HAROMA_MEMORY_TRAINING_EXPORT_TICKS", raising=False)
    mock_shared = MagicMock()
    mock_shared.agent_config = {"background": {}}
    bus = MessageBus()
    bg = BackgroundAgent(mock_shared, bus, boot_agent=None, tick_interval=5.0)
    assert bg._finetune_flush_every == 12
