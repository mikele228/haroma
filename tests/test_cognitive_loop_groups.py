"""mind.cognitive_loop_groups — env helpers and parallel prep."""

from __future__ import annotations

import sys

import pytest

_REPO = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def test_phased_cycle_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HAROMA_PERSONA_PHASED_CYCLE", raising=False)
    import importlib

    import mind.cognitive_loop_groups as clg

    importlib.reload(clg)
    assert clg.cognitive_phases_enabled() is False

    monkeypatch.setenv("HAROMA_PERSONA_PHASED_CYCLE", "1")
    importlib.reload(clg)
    assert clg.cognitive_phases_enabled() is True
    assert clg.phase_steps_per_invocation() >= 1


def test_run_parallel_prep_sequential_by_default():
    from mind.cognitive_loop_groups import run_parallel_prep

    out = run_parallel_prep([("a", lambda: 1), ("b", lambda: 2)])
    assert len(out) == 2
    vals = {k: v for k, v in out}
    assert vals["a"] == 1
    assert vals["b"] == 2
