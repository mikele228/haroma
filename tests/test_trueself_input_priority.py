"""TrueSelf input batch ordering: HTTP-traced turns before sensor backlog."""

from __future__ import annotations

import sys

_REPO = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from agents.message_bus import Message
from mind.trueself_input_priority import (
    is_sensor_pulse_no_chat_text,
    partition_trueself_input_for_tick,
    prioritize_trueself_input_messages,
)


def _msg(mid: str, **meta):
    return Message(
        sender_id="input",
        channel="x",
        content="",
        metadata=meta,
        message_id=mid,
    )


def test_prioritize_traced_before_untraced_preserves_fifo_within_tier():
    a = _msg("a")  # sensor-only
    b = _msg("b")  # sensor-only
    c = _msg("c", cognitive_trace_id="trace_http")
    out = prioritize_trueself_input_messages([a, b, c])
    assert [m.message_id for m in out] == ["c", "a", "b"]


def test_prioritize_slot_without_trace_before_plain_sensor():
    slot = _msg("s", _response_slot={"event": object()})
    plain = _msg("p")
    traced = _msg("t", cognitive_trace_id="z")
    out = prioritize_trueself_input_messages([plain, slot, traced])
    assert [m.message_id for m in out] == ["t", "s", "p"]


def test_single_message_unchanged():
    m = _msg("only")
    assert prioritize_trueself_input_messages([m]) == [m]


def test_prioritize_trace_only_on_response_slot():
    """Match :func:`trace_id_from_message` — trace may live only on ``_response_slot``."""
    slot = {"cognitive_trace_id": "slot_only", "event": object()}
    traced = Message(
        sender_id="input",
        channel="input",
        content={"text": "hi", "source": "user"},
        metadata={"_response_slot": slot},
        message_id="slottr",
    )
    sensor = _msg("sens")
    out = prioritize_trueself_input_messages([sensor, traced])
    assert [m.message_id for m in out] == ["slottr", "sens"]


def test_prioritize_user_text_without_top_level_trace():
    user = Message(
        sender_id="input",
        channel="input",
        content={"text": "how are you?", "source": "user"},
        metadata={},
        message_id="u1",
    )
    vision = Message(
        sender_id="input",
        channel="input",
        content={"text": "Multimodal tick", "source": "sensor"},
        metadata={},
        message_id="v1",
    )
    out = prioritize_trueself_input_messages([vision, user])
    assert [m.message_id for m in out] == ["u1", "v1"]


def test_partition_defers_sensor_when_user_present():
    user = Message(
        sender_id="input",
        channel="input",
        content={"text": "hi", "source": "user"},
        metadata={},
        message_id="u1",
    )
    s1 = Message(
        sender_id="input",
        channel="input",
        content={"text": "", "source": "sensor"},
        metadata={},
        message_id="s1",
    )
    s2 = Message(
        sender_id="input",
        channel="input",
        content={"text": "", "source": "sensor"},
        metadata={},
        message_id="s2",
    )
    now, later = partition_trueself_input_for_tick([s1, user, s2])
    assert [m.message_id for m in now] == ["u1"]
    assert [m.message_id for m in later] == ["s1", "s2"]


def test_is_sensor_pulse_no_chat_text():
    pulse = Message(
        sender_id="input",
        channel="input",
        content={
            "text": "",
            "source": "sensor",
            "sensor_data": {"vision": [{"x": 1}]},
        },
        message_type="input",
        message_id="pulse1",
    )
    assert is_sensor_pulse_no_chat_text(pulse) is True
    user = Message(
        sender_id="input",
        channel="input",
        content={"text": "hi", "source": "user"},
        metadata={},
        message_id="u1",
    )
    assert is_sensor_pulse_no_chat_text(user) is False


def test_partition_all_sensor_no_split():
    a = Message(
        sender_id="input",
        channel="input",
        content={"text": "", "source": "sensor"},
        metadata={},
        message_id="a",
    )
    b = Message(
        sender_id="input",
        channel="input",
        content={"text": "", "source": "sensor"},
        metadata={},
        message_id="b",
    )
    now, later = partition_trueself_input_for_tick([a, b])
    assert later == []
    assert [m.message_id for m in now] == ["a", "b"]
