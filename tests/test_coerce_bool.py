"""Tests for ``utils.coerce_bool``."""

from utils.coerce_bool import env_flag, json_bool


def test_json_bool():
    assert json_bool("false", default=True) is False
    assert json_bool("true", default=False) is True


def test_env_flag_missing_uses_default(monkeypatch):
    monkeypatch.delenv("HAROMA_TEST_FLAG_XX", raising=False)
    assert env_flag("HAROMA_TEST_FLAG_XX", default=False) is False
    assert env_flag("HAROMA_TEST_FLAG_XX", default=True) is True


def test_env_flag_set(monkeypatch):
    monkeypatch.setenv("HAROMA_TEST_FLAG_YY", "1")
    assert env_flag("HAROMA_TEST_FLAG_YY", default=False) is True
    monkeypatch.setenv("HAROMA_TEST_FLAG_YY", "0")
    assert env_flag("HAROMA_TEST_FLAG_YY", default=True) is False
