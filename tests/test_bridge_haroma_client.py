"""bridge/ stub executor + feedback shape (no live Haroma)."""

from __future__ import annotations

import sys

_REPO = __import__("os").path.dirname(__import__("os").path.dirname(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from bridge.stub_executor import feedback_block_from_results, simulate_command_results
from mind.robot_execution_contract import build_executor_command_batch, normalize_feedback_payload


def test_stub_executor_produces_normalizable_feedback():
    batch = build_executor_command_batch(
        [{"label": "a", "command": "noop"}],
        correlation_id="test-corr-uuid",
    )
    results = simulate_command_results(batch)
    fb = feedback_block_from_results(batch, results)
    norm, err = normalize_feedback_payload(fb)
    assert err is None
    assert norm["correlation_id"]
    assert len(norm["results"]) == 1
    assert norm["results"][0]["status"] == "completed"


def test_feedback_empty_batch_normalizes_to_empty_results():
    batch = build_executor_command_batch([], correlation_id="empty")
    results = simulate_command_results(batch)
    assert results == []
    fb = feedback_block_from_results(batch, results)
    norm, err = normalize_feedback_payload(fb)
    assert err is None
    assert norm["results"] == []
