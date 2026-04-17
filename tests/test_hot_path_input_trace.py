"""Hot path: InputAgent assigns a cognitive trace id per push_text."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_REPO = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
import sys

if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


@pytest.fixture
def input_agent_minimal():
    from agents.input_agent import InputAgent
    from agents.message_bus import MessageBus

    bus = MessageBus()
    shared = SimpleNamespace(
        cycle_count=0,
        cognitive_metrics=MagicMock(),
        encoder=None,
        neural_sync=lambda: __import__("contextlib").nullcontext(),
    )
    ia = InputAgent(shared=shared, bus=bus, tick_interval=1.0)
    ia._boot_agent = SimpleNamespace(trueself_agent=MagicMock())
    return ia, shared


def test_push_text_non_normal_depth_normalized_queues_with_trace(input_agent_minimal):
    """Unknown ``depth`` values are normalized to ``normal``."""
    ia, shared = input_agent_minimal

    slot = ia.push_text("hello", source="user", depth="fast")

    assert "cognitive_trace_id" in slot
    assert len(slot["cognitive_trace_id"]) == 20
    shared.cognitive_metrics.on_chat_turn_started.assert_called()
    with ia._lock:
        assert len(ia._text_queue_priority) >= 1
        item = ia._text_queue_priority[-1]
    assert item.get("cognitive_trace_id") == slot["cognitive_trace_id"]
    assert item.get("depth") == "normal"
    assert item.get("channel") == "chat"


def test_push_text_normal_queues_item_with_trace(input_agent_minimal):
    ia, _shared = input_agent_minimal
    slot = ia.push_text("hello", source="user", depth="normal")
    assert isinstance(slot.get("event"), type(threading.Event()))
    with ia._lock:
        assert len(ia._text_queue_priority) >= 1
        item = ia._text_queue_priority[-1]
    assert item.get("cognitive_trace_id") == slot["cognitive_trace_id"]
    assert item.get("channel") == "chat"
