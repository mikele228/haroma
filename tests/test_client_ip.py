"""Unit tests for mind.client_ip (no BootAgent / torch)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mind.client_ip import (
    clear_trusted_proxy_cache_for_tests,
    direct_peer_trusted_for_xff,
    get_effective_client_ip,
)


@pytest.fixture(autouse=True)
def _clear_ip_cache():
    clear_trusted_proxy_cache_for_tests()
    yield
    clear_trusted_proxy_cache_for_tests()


def _req(remote: str, xff: str | None = None):
    h = {}
    if xff is not None:
        h["X-Forwarded-For"] = xff
    return SimpleNamespace(remote_addr=remote, headers=h)


def test_default_uses_remote_addr(monkeypatch):
    monkeypatch.delenv("HAROMA_HTTP_USE_X_FORWARDED_FOR", raising=False)
    assert get_effective_client_ip(_req("203.0.113.5")) == "203.0.113.5"


def test_xff_ignored_when_env_off(monkeypatch):
    monkeypatch.delenv("HAROMA_HTTP_USE_X_FORWARDED_FOR", raising=False)
    r = _req("127.0.0.1", "198.51.100.2, 10.0.0.1")
    assert get_effective_client_ip(r) == "127.0.0.1"


def test_xff_from_loopback_when_enabled(monkeypatch):
    monkeypatch.setenv("HAROMA_HTTP_USE_X_FORWARDED_FOR", "1")
    r = _req("127.0.0.1", "198.51.100.2, 10.0.0.1")
    assert get_effective_client_ip(r) == "198.51.100.2"


def test_xff_spoof_ignored_when_peer_untrusted(monkeypatch):
    monkeypatch.setenv("HAROMA_HTTP_USE_X_FORWARDED_FOR", "1")
    r = _req("203.0.113.99", "198.51.100.2")
    assert get_effective_client_ip(r) == "203.0.113.99"


def test_xff_from_trusted_cidr(monkeypatch):
    monkeypatch.setenv("HAROMA_HTTP_USE_X_FORWARDED_FOR", "1")
    monkeypatch.setenv("HAROMA_HTTP_TRUSTED_PROXIES", "10.0.0.0/8")
    r = _req("10.1.2.3", "198.51.100.7")
    assert get_effective_client_ip(r) == "198.51.100.7"


def test_direct_peer_trusted_loopback():
    assert direct_peer_trusted_for_xff("127.0.0.1") is True
    assert direct_peer_trusted_for_xff("203.0.113.1") is False
