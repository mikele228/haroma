"""Behavior of ``merge_deliberative_into_llm_context`` (via ``mind.cognitive_contracts``)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mind.cognitive_contracts import (
    complete_deferred_deliberative_llm_context,
    merge_deliberative_into_llm_context,
)


def test_merge_skips_when_deliberative_flag_false():
    d = {"candidate_actions": [{"x": 1}], "source": "x"}
    merge_deliberative_into_llm_context(
        d,
        deliberative_flag=False,
        current_values={},
        active_goals=[],
        drive_state=None,
        episode_affect=None,
        emotion_summary=None,
        symbolic_law=None,
    )
    assert "deliberative_scores" not in d


def test_merge_skips_when_no_candidates():
    d = {"source": "x"}
    merge_deliberative_into_llm_context(
        d,
        deliberative_flag=True,
        current_values={},
        active_goals=[],
        drive_state=None,
        episode_affect=None,
        emotion_summary=None,
        symbolic_law=None,
    )
    assert "deliberative_scores" not in d


def test_complete_deferred_binds_episode():
    bound = []

    class Ep:
        affect = {"x": 1}
        symbolic_law = None

        def bind_llm_context(self, d):
            bound.append(d)

    payload = {"source": "packed", "candidate_actions": []}
    complete_deferred_deliberative_llm_context(
        payload,
        deliberative_flag=True,
        val_mgr=None,
        episode=Ep(),
        active_goals=[],
        drive_state=None,
        emotion_summary=None,
    )
    assert bound == [payload]
