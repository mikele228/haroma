"""HTTP /chat wait vs packed LLM caps."""

import os


def test_http_wait_normal_covers_llm_cap(monkeypatch):
    monkeypatch.delenv("HAROMA_CHAT_TIMEOUT", raising=False)
    monkeypatch.setenv("HAROMA_LLM_CONTEXT_TIMEOUT_SEC", "360")
    from mind.elarion_server_v2 import _http_chat_wait_sec

    w = _http_chat_wait_sec("normal")
    assert w >= 480
    assert w >= 360 + 120


def test_http_wait_normal_when_llm_unlimited(monkeypatch):
    monkeypatch.delenv("HAROMA_CHAT_TIMEOUT", raising=False)
    monkeypatch.setenv("HAROMA_LLM_CONTEXT_TIMEOUT_SEC", "0")
    from mind.elarion_server_v2 import _http_chat_wait_sec

    assert _http_chat_wait_sec("normal") == 600


def test_invalid_haroma_chat_timeout_falls_back_to_llm_margin(monkeypatch):
    monkeypatch.setenv("HAROMA_CHAT_TIMEOUT", "not_a_number")
    monkeypatch.setenv("HAROMA_LLM_CONTEXT_TIMEOUT_SEC", "360")
    from mind.elarion_server_v2 import _http_chat_wait_sec

    w = _http_chat_wait_sec("normal")
    assert w >= 540


def test_http_wait_legacy_fast_arg_same_as_normal(monkeypatch):
    """``depth`` is legacy; ``fast`` and ``normal`` use the same HTTP wait."""
    monkeypatch.delenv("HAROMA_CHAT_TIMEOUT", raising=False)
    monkeypatch.setenv("HAROMA_FAST_LLM_DEFAULT_TIMEOUT_SEC", "90")
    monkeypatch.setenv("HAROMA_LLM_CONTEXT_TIMEOUT_SEC", "360")
    from mind.elarion_server_v2 import _http_chat_wait_sec

    assert _http_chat_wait_sec("fast") == _http_chat_wait_sec("normal")


def test_json_bool_does_not_treat_string_false_as_true():
    """Flask/JSON may leave flags as strings; ``bool(\"false\")`` is True in Python."""
    from utils.coerce_bool import json_bool

    assert json_bool("false", False) is False
    assert json_bool("true", False) is True
    assert json_bool("0", True) is False
    assert json_bool(0, True) is False
    assert json_bool(None, True) is True
