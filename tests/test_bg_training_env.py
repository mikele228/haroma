"""mind.bg_training_env — defer policy and cap parsing."""

from __future__ import annotations

import sys

import pytest

_REPO = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mind.bg_training_env import (
    bg_training_defer_cap_effective_seconds,
    bg_training_defer_cap_sec,
    defer_training_on_http_chat,
    defer_training_on_input_pipeline,
)


def test_defer_default_true(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HAROMA_BG_DEFER_TRAINING_ON_HTTP_CHAT", raising=False)
    assert defer_training_on_http_chat() is True


def test_defer_explicit_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_BG_DEFER_TRAINING_ON_HTTP_CHAT", "0")
    assert defer_training_on_http_chat() is False


def test_defer_input_pipeline_env_overrides_legacy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_BG_DEFER_TRAINING_ON_HTTP_CHAT", "1")
    monkeypatch.setenv("HAROMA_BG_DEFER_TRAINING_ON_INPUT_PIPELINE", "0")
    assert defer_training_on_input_pipeline() is False
    assert defer_training_on_http_chat() is False


def test_defer_legacy_when_new_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HAROMA_BG_DEFER_TRAINING_ON_INPUT_PIPELINE", raising=False)
    monkeypatch.setenv("HAROMA_BG_DEFER_TRAINING_ON_HTTP_CHAT", "0")
    assert defer_training_on_input_pipeline() is False


def test_cap_unset_is_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HAROMA_BG_DEFER_TRAINING_CAP_SEC", raising=False)
    assert bg_training_defer_cap_sec() is None
    assert bg_training_defer_cap_effective_seconds() == 0.0


def test_cap_explicit_zero(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_BG_DEFER_TRAINING_CAP_SEC", "0")
    assert bg_training_defer_cap_sec() == 0.0
    assert bg_training_defer_cap_effective_seconds() == 0.0


def test_cap_positive(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_BG_DEFER_TRAINING_CAP_SEC", "12.5")
    assert bg_training_defer_cap_sec() == 12.5
    assert bg_training_defer_cap_effective_seconds() == 12.5


def test_cap_invalid_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HAROMA_BG_DEFER_TRAINING_CAP_SEC", "not_a_float")
    assert bg_training_defer_cap_sec() is None
    assert bg_training_defer_cap_effective_seconds() == 0.0
