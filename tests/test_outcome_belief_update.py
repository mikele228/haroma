"""Tests for outcome-grounded belief updates."""

from unittest.mock import MagicMock

from mind import outcome_belief_update as obu


def test_merge_dedupes_across_sources():
    r = {"inferences": [{"subject": "A", "predicate": "p", "object": "B", "confidence": 0.6}]}
    llm = {"inferences": [{"subject": "A", "predicate": "p", "object": "B", "confidence": 0.9}]}
    m = obu._merge_inference_rows(r, llm, max_rows=10)
    assert len(m) == 1


def test_apply_writes_belief_nodes(monkeypatch):
    monkeypatch.setenv("HAROMA_OUTCOME_BELIEF_UPDATE", "1")
    added = []
    mem = MagicMock()

    def _add(tree, branch, node):
        added.append((tree, branch, node.content[:40]))

    mem.add_node = _add
    n = obu.apply_outcome_grounded_belief_updates(
        memory=mem,
        outcome={"score": 0.85, "lesson": "ok"},
        reasoning_result={
            "inferences": [
                {"subject": "X", "predicate": "likes", "object": "Y", "confidence": 0.5},
            ],
        },
        llm_context={},
        cycle_id=7,
        branch_name="conversant",
        agent_id="p1",
    )
    assert n == 1
    assert added[0][0] == "belief_tree"
    assert added[0][1] == "conversant"
    assert "outcome 0.85" in added[0][2] or "outcome 0.85" in added[0][2].lower()


def test_deliberative_multiplier_extremes():
    assert obu.deliberative_belief_outcome_multiplier(0.5) == 1.0
    assert obu.deliberative_belief_outcome_multiplier(1.0) > 1.0
    assert obu.deliberative_belief_outcome_multiplier(0.0) < 1.0
