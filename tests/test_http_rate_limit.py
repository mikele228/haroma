"""Unit tests for mind.http_rate_limit (no BootAgent)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from mind.http_rate_limit import (
    check_rate_limit,
    clear_rate_limit_state_for_tests,
    get_rate_limit_per_minute,
    rate_limit_per_minute,
)


@pytest.fixture(autouse=True)
def _clear_buckets():
    clear_rate_limit_state_for_tests()
    yield
    clear_rate_limit_state_for_tests()


def _req(method: str, path: str, ip: str = "198.51.100.1"):
    return SimpleNamespace(method=method, path=path, remote_addr=ip, headers={})


def test_post_limit_respected(monkeypatch):
    monkeypatch.setenv("HAROMA_HTTP_RATE_LIMIT_PER_MIN", "2")
    monkeypatch.delenv("HAROMA_HTTP_GET_RATE_LIMIT_PER_MIN", raising=False)
    assert rate_limit_per_minute() == 2
    assert check_rate_limit(_req("POST", "/chat")) is None
    assert check_rate_limit(_req("POST", "/chat")) is None
    out = check_rate_limit(_req("POST", "/chat"))
    assert out is not None
    assert out[1] == 429


def test_get_chat_result_limit_separate(monkeypatch):
    monkeypatch.delenv("HAROMA_HTTP_RATE_LIMIT_PER_MIN", raising=False)
    monkeypatch.setenv("HAROMA_HTTP_GET_RATE_LIMIT_PER_MIN", "2")
    assert get_rate_limit_per_minute() == 2
    assert check_rate_limit(_req("GET", "/chat/result")) is None
    assert check_rate_limit(_req("GET", "/chat/result")) is None
    out = check_rate_limit(_req("GET", "/chat/result"))
    assert out is not None and out[1] == 429


def test_get_status_unlimited(monkeypatch):
    monkeypatch.setenv("HAROMA_HTTP_GET_RATE_LIMIT_PER_MIN", "1")
    assert check_rate_limit(_req("GET", "/status")) is None
