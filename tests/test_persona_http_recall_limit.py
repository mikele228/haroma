"""HTTP-traced /chat recall_fast limit matches env cap (no legacy hard cap at 8)."""

from __future__ import annotations

import sys

_REPO = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from agents.persona_agent import _http_traced_recall_fast_limit


def test_http_traced_recall_fast_limit_respects_twelve():
    assert _http_traced_recall_fast_limit(12) == 12


def test_http_traced_recall_fast_limit_minimum_one():
    assert _http_traced_recall_fast_limit(0) == 1
    assert _http_traced_recall_fast_limit(-3) == 1


def test_http_traced_recall_fast_limit_small_positive():
    assert _http_traced_recall_fast_limit(3) == 3
