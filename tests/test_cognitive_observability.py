"""Unit tests for mind.cognitive_observability and chat_latency trace attachment."""

from mind.cognitive_observability import (
    CognitiveMetrics,
    append_cognitive_trace_to_payload,
    new_trace_id,
)
from agents import chat_latency


def test_new_trace_id_shape():
    t = new_trace_id()
    assert len(t) == 20
    assert t == t.lower()


def test_cognitive_metrics_snapshot():
    m = CognitiveMetrics(sample_cap=10)
    m.on_chat_turn_started()
    m.record_route("fast_trueself")
    m.record_route("delegate")
    m.observe_llm_wait_ms(120.0)
    m.observe_persona_cycle_ms(450.0)
    m.observe_input_queue_depth(1, 2)
    m.record_persistence_save_sec(0.05)
    snap = m.snapshot()
    assert snap["chat_turns"] == 1
    assert snap["routes"]["fast_trueself"] == 1
    assert snap["routes"]["delegate"] == 1
    assert snap["input_queue_depth_last"] == 3
    assert snap["persistence_save"]["count"] == 1
    assert snap["llm_wait_ms"]["count"] >= 1


def test_append_cognitive_trace_to_payload():
    p: dict = {"response": "hi"}
    append_cognitive_trace_to_payload(p, trace_id="abc", route="fast:trueself")
    assert p["cognitive_trace_id"] == "abc"
    assert p["cognitive_route"] == "fast:trueself"


def test_trace_attach_adds_cognitive_without_latency_trace():
    """``trace_attach_to_payload`` must always attach trace/route when present."""
    slot = {
        "cognitive_trace_id": "tid123",
        "_cognitive_route": "normal:trueself",
        # trace_latency disabled: no _trace_enabled
    }
    payload: dict = {"response": "ok", "cycle": 1}
    chat_latency.trace_attach_to_payload(slot, payload)
    assert payload.get("cognitive_trace_id") == "tid123"
    assert payload.get("cognitive_route") == "normal:trueself"
    assert "latency_trace" not in payload
