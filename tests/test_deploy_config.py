"""mind.deploy_config — dotenv loading and listen helpers."""

from __future__ import annotations

import os
import sys

import pytest

_REPO = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def test_http_listen_port_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HAROMA_HTTP_PORT", raising=False)
    from mind import deploy_config as dc

    assert dc.http_listen_port() == 8193


def test_http_listen_port_invalid_falls_back(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_HTTP_PORT", "not-a-port")
    from mind import deploy_config as dc

    assert dc.http_listen_port() == 8193


def test_load_dotenv_sets_keys(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HAROMA_HTTP_PORT", raising=False)
    p = tmp_path / ".env"
    p.write_text(
        "# comment\nHAROMA_HTTP_PORT=9000\nHAROMA_BIND_HOST=127.0.0.1\n",
        encoding="utf-8",
    )
    import mind.deploy_config as dc

    monkeypatch.setattr(dc, "dotenv_path", lambda: str(p))
    n = dc.load_dotenv(override=True)
    assert n >= 2
    assert os.environ.get("HAROMA_HTTP_PORT") == "9000"
    assert dc.http_listen_port() == 9000
