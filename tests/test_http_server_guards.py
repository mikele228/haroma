"""mind.http_server_guards — optional bearer token."""

from __future__ import annotations

import sys

import pytest

_REPO = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mind import http_server_guards as g


class _Req:
    def __init__(self, path: str, headers: dict):
        self.path = path
        self.headers = headers


def test_no_secret_allows_all(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HAROMA_HTTP_BEARER_TOKEN", raising=False)
    r = _Req("/teach", {})
    assert g.verify_http_request_bearer(r) is None


def test_secret_requires_bearer(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_HTTP_BEARER_TOKEN", "secret-token")
    r = _Req("/teach", {})
    out = g.verify_http_request_bearer(r)
    assert out is not None
    assert out[1] == 401


def test_valid_bearer_ok(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_HTTP_BEARER_TOKEN", "secret-token")
    r = _Req("/teach", {"Authorization": "Bearer secret-token"})
    assert g.verify_http_request_bearer(r) is None


def test_x_haroma_token_ok(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_HTTP_BEARER_TOKEN", "abc")
    r = _Req("/save", {"X-Haroma-Token": "abc"})
    assert g.verify_http_request_bearer(r) is None


def test_wrong_token_403(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_HTTP_BEARER_TOKEN", "good")
    r = _Req("/agent/environment", {"Authorization": "Bearer bad"})
    out = g.verify_http_request_bearer(r)
    assert out is not None
    assert out[1] == 403


def test_unlisted_path_not_checked(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_HTTP_BEARER_TOKEN", "x")
    r = _Req("/status", {})
    assert g.verify_http_request_bearer(r) is None


def test_custom_protect_paths_empty_uses_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_HTTP_BEARER_TOKEN", "x")
    monkeypatch.setenv("HAROMA_HTTP_PROTECT_PATHS", "")
    # empty string -> default set in module (non-empty raw falls through - check code)
    # Our code: if not raw -> default; if raw empty after strip -> default
    assert "/teach" in g.protected_path_set()


def test_custom_protect_paths_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_HTTP_BEARER_TOKEN", "x")
    monkeypatch.setenv("HAROMA_HTTP_PROTECT_PATHS", "/custom")
    r = _Req("/custom", {})
    out = g.verify_http_request_bearer(r)
    assert out is not None and out[1] == 401
    assert g.verify_http_request_bearer(_Req("/teach", {"Authorization": "Bearer x"})) is None
