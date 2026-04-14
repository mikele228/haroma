"""Unit tests for teaching-turn detection and web_learn recall merge."""

from __future__ import annotations

import pytest

from core.chat_recall_policy import (
    is_factual_heuristic_turn,
    is_teaching_turn,
    merge_web_learn_tail,
    should_merge_web_learn,
)


class _FakeNode:
    __slots__ = ("moment_id",)

    def __init__(self, moment_id: str) -> None:
        self.moment_id = moment_id


class _FakeForest:
    def __init__(self, web_nodes: list[_FakeNode]) -> None:
        self._web = web_nodes

    def get_nodes(self, tree_name: str, branch_name: str) -> list[_FakeNode]:
        return list(self._web)


@pytest.mark.parametrize(
    "text,nlu,expect",
    [
        ("Teach me about rays", {}, True),
        ("Explain wave interference", {}, True),
        ("ok thanks", {}, False),
        ("Right?", {"intent": "utterance"}, False),
        ("What is consciousness?", {"intent": "utterance"}, True),
    ],
)
def test_is_teaching_turn(text: str, nlu: dict, expect: bool) -> None:
    assert is_teaching_turn(text, nlu) is expect


def test_merge_web_learn_tail_prefix_and_cap() -> None:
    a, b, c = _FakeNode("wa"), _FakeNode("wb"), _FakeNode("wc")
    forest = _FakeForest([a, b, c])
    existing = [_FakeNode("mem1")]
    out = merge_web_learn_tail(forest, existing, recall_limit=3, max_inject=3)
    ids = [n.moment_id for n in out]
    assert ids == ["wc", "wb", "wa"]


@pytest.mark.parametrize(
    "text,nlu,expect",
    [
        ("Who discovered DNA?", {}, True),
        ("Paris is nice", {"entities": [{"x": 1}]}, True),
        ("hello", {}, False),
    ],
)
def test_is_factual_heuristic_turn(text: str, nlu: dict, expect: bool) -> None:
    assert is_factual_heuristic_turn(text, nlu) is expect


def test_should_merge_teaching_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAROMA_WEB_LEARN_INJECT_MODE", "teaching_only")
    assert should_merge_web_learn("Who won?", {}, teaching=False) is False
    assert should_merge_web_learn("Teach me X", {}, teaching=True) is True


def test_should_merge_factual_heuristic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAROMA_WEB_LEARN_INJECT_MODE", "factual_heuristic")
    assert should_merge_web_learn("Who won the race?", {}, teaching=False) is True
    assert should_merge_web_learn("ok", {}, teaching=False) is False


def test_should_merge_always(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAROMA_WEB_LEARN_INJECT_MODE", "always")
    assert should_merge_web_learn("hi", {}, teaching=False) is True


def test_merge_web_learn_tail_dedup_with_recalled() -> None:
    a, b, c = _FakeNode("wa"), _FakeNode("wb"), _FakeNode("wc")
    forest = _FakeForest([a, b, c])
    existing = [_FakeNode("wb")]
    out = merge_web_learn_tail(forest, existing, recall_limit=10, max_inject=3)
    ids = [n.moment_id for n in out]
    assert ids == ["wc", "wa", "wb"]
