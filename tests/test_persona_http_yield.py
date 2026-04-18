"""mind.persona_http_yield — HTTP /chat yield helpers for PersonaAgent."""

from __future__ import annotations

import sys

_REPO = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from types import SimpleNamespace

from mind.persona_http_yield import (
    defer_inner_cycle_before_neural,
    http_chat_inflight_positive,
    inner_relay_should_requeue,
    internal_treat_as_over_budget,
    skip_semantic_recall_for_internal,
)


def test_http_chat_inflight_positive():
    assert http_chat_inflight_positive(SimpleNamespace(http_chat_inflight=1)) is True
    assert http_chat_inflight_positive(SimpleNamespace(http_chat_inflight=0)) is False


def test_inner_relay_should_requeue():
    assert inner_relay_should_requeue("inner_dialogue") is True
    assert inner_relay_should_requeue("dialogue_reply") is True
    assert inner_relay_should_requeue("input") is False


def test_defer_inner_cycle_before_neural():
    assert defer_inner_cycle_before_neural(True, SimpleNamespace(http_chat_inflight=1)) is True
    assert defer_inner_cycle_before_neural(False, SimpleNamespace(http_chat_inflight=1)) is False
    assert defer_inner_cycle_before_neural(True, SimpleNamespace(http_chat_inflight=0)) is False


def test_skip_semantic_recall_matches_internal():
    assert skip_semantic_recall_for_internal(True, SimpleNamespace(http_chat_inflight=1)) is True
    assert skip_semantic_recall_for_internal(True, SimpleNamespace(http_chat_inflight=0)) is False


def test_internal_treat_as_over_budget():
    assert internal_treat_as_over_budget(True, SimpleNamespace(http_chat_inflight=2)) is True
    assert internal_treat_as_over_budget(False, SimpleNamespace(http_chat_inflight=2)) is False
