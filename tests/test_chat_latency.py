"""
Chat response time: optional integration check (server on localhost:8193).

CI / default pytest: skipped if nothing is listening.
Manual: start Elarion, then  pytest tests/test_chat_latency.py -m integration

For full benchmarks use:  python scripts/benchmark_chat_response_time.py
"""

from __future__ import annotations

import time

import pytest

try:
    import requests
except ImportError:
    requests = None  # type: ignore


BASE = "http://127.0.0.1:8193"


def _server_up() -> bool:
    if requests is None:
        return False
    try:
        r = requests.get(f"{BASE}/status", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.integration
def test_chat_response_time_when_server_running():
    """Soft check: /chat returns within a generous budget when server is up."""
    if requests is None:
        pytest.skip("requests not installed")
    if not _server_up():
        pytest.skip("Haroma server not reachable at " + BASE)

    # First reply can be very slow (local GGUF); use large timeout, loose assert.
    t0 = time.perf_counter()
    r = requests.post(
        f"{BASE}/chat",
        json={"message": "ping", "depth": "normal"},
        timeout=(10.0, 600.0),
    )
    elapsed = time.perf_counter() - t0
    assert r.status_code == 200, r.text[:200]
    data = r.json()
    assert "response" in data
    assert elapsed < 900.0, f"/chat took {elapsed:.1f}s (adjust env/timeouts if legitimately slow)"


@pytest.mark.integration
def test_chat_uses_depth_not_mode():
    """Regression: API expects ``depth``, not ``mode`` (wrong key yields slow/default path)."""
    if requests is None or not _server_up():
        pytest.skip("server unavailable")

    r = requests.post(
        f"{BASE}/chat",
        json={"message": "hi", "depth": "normal"},
        timeout=(10.0, 120.0),
    )
    assert r.status_code == 200
