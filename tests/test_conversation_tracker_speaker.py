"""ConversationTracker speaker scoping (multi-user /chat)."""

from core.ConversationTracker import ConversationTracker


def test_get_recent_filters_by_speaker():
    ct = ConversationTracker(max_history=50)
    ct.record_input("hi a", speaker="user:alice", cycle_id=1)
    ct.record_input("hi b", speaker="user:bob", cycle_id=2)
    r_all = ct.get_recent(10)
    assert len(r_all) == 2
    r_alice = ct.get_recent(10, speaker="user:alice")
    assert len(r_alice) == 1
    assert r_alice[0].content == "hi a"


def test_get_context_summary_no_fallback_to_other_speaker():
    ct = ConversationTracker(max_history=50)
    ct.record_input("only bob", speaker="user:bob", cycle_id=1)
    # No alice turns — must not echo bob's line.
    assert ct.get_context_summary(speaker="user:alice") == ""


def test_store_discourse_snapshot_attaches_to_recorded_turn():
    """Snapshot must run after ``record_input`` for the same ``cycle_id``."""
    ct = ConversationTracker(max_history=50)
    ct.record_input("hello", speaker="user:u1", cycle_id=7, tags=[])
    ct.store_discourse_snapshot(7, {"topics": ["weather"], "open_questions": []})
    with ct._lock:
        last = ct.history[-1]
    assert getattr(last, "_discourse", None) is not None
    assert "weather" in str(last._discourse.get("topics", []))


def test_get_context_summary_omits_global_topic_when_scoped():
    ct = ConversationTracker(max_history=50)
    ct.record_input("x", speaker="user:alice", cycle_id=1, tags=["t1"])
    ct.record_input("y", speaker="user:bob", cycle_id=2, tags=["t2"])
    full = ct.get_context_summary(None)
    assert "Current topic" in full or len(full) > 0
    scoped = ct.get_context_summary(speaker="user:alice")
    assert "Current topic:" not in scoped
