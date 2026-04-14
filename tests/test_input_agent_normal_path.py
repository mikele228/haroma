"""InputAgent queue behavior for depth=normal (priority + drain order)."""

from unittest.mock import MagicMock

from agents.input_agent import InputAgent
from agents.message_bus import MessageBus


def _minimal_shared():
    s = MagicMock()
    s.cycle_count = 1
    s.encoder = None
    s.perception = None
    return s


def test_normal_user_goes_to_priority_queue():
    bus = MessageBus(dead_letter_timeout_ms=2000)
    ia = InputAgent(_minimal_shared(), bus, tick_interval=1.0)
    slot = ia.push_text("a", source="user", depth="normal")
    with ia._lock:
        assert len(ia._text_queue_priority) == 1
        assert len(ia._text_queue) == 0
    texts, sensors = ia._drain_all()
    assert len(texts) == 1
    assert texts[0]["content"] == "a"
    assert not sensors
    assert not slot["event"].is_set()


def test_priority_drains_before_fifo():
    bus = MessageBus(dead_letter_timeout_ms=2000)
    ia = InputAgent(_minimal_shared(), bus, tick_interval=1.0)
    ia.push_text("later", source="internal", depth="normal")
    ia.push_text("first", source="user", depth="normal")
    texts, _ = ia._drain_all()
    assert [t["content"] for t in texts] == ["first", "later"]
