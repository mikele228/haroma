"""Tests for mind.user_identity."""

from mind.user_identity import sanitize_user_id, speaker_key, user_tag


def test_sanitize_user_id_none_and_empty():
    assert sanitize_user_id(None) is None
    assert sanitize_user_id("") is None
    assert sanitize_user_id("   ") is None


def test_sanitize_user_id_strips_unsafe():
    assert sanitize_user_id("alice/bob") == "alice_bob"
    assert sanitize_user_id("a" * 200) is not None
    assert len(sanitize_user_id("a" * 200) or "") <= 128


def test_user_tag_and_speaker_key():
    assert user_tag("alice") == "user:alice"
    assert user_tag(None) is None
    assert speaker_key("conversant", None) == "conversant"
    assert speaker_key("conversant", "bob") == "user:bob"
