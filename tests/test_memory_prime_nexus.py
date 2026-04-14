"""Memory prime / vector / Nexus bridge smoke tests."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.Memory import MemoryForest, MemoryNode
from core.NexusMemoryBridge import ingest_nexus_fact
from core.ActionLoop import _build_memory_context_snippets


def test_memory_node_vector_roundtrip():
    n = MemoryNode(content="hello", tags=["t"], vector=[0.1, 0.2, 0.3])
    d = n.to_dict()
    assert "vector" in d
    n2 = MemoryNode.from_dict(d)
    assert n2.vector is not None
    assert n2.vector.shape == (3,)
    assert abs(float(n2.vector[0]) - 0.1) < 1e-5


def test_large_vector_omitted_in_to_dict():
    n = MemoryNode(content="big")
    n.set_vector(np.ones(70000))
    d = n.to_dict()
    assert d.get("vector_omitted") is True


def test_prime_recall_boost():
    mf = MemoryForest(encoder=None)
    a = MemoryNode(
        content="alpha unrelated zyxwvuts",
        tags=["episodic"],
        confidence=0.9,
    )
    b = MemoryNode(
        content="beta unrelated zyxwvuts",
        tags=["episodic", "prime"],
        confidence=0.9,
    )
    mf.add_node("thought_tree", "common", a)
    mf.add_node("thought_tree", "common", b)
    out = mf.recall(query_text="zyxwvuts unrelated", limit=4)
    ids = [n.moment_id for n in out]
    assert b.moment_id in ids
    assert ids[0] == b.moment_id


def test_merge_recall_with_prime_reserves_fast_path():
    mf = MemoryForest(encoder=None)
    p = ingest_nexus_fact(
        mf,
        "curated fact about quotas",
        nexus_uri="https://nexus.example.com/r1",
    )
    base = MemoryNode(content="noise", tags=["x"])
    merged = mf.merge_recall_with_prime([base], limit=2, fast_cycle=True)
    assert len(merged) == 2
    assert merged[0].moment_id == p.moment_id


def test_recall_by_vector():
    mf = MemoryForest(encoder=None)
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    good = MemoryNode(content="g", tags=["has"])
    good.set_vector([0.99, 0.01, 0.0])
    bad = MemoryNode(content="b", tags=["has"])
    bad.set_vector([0.0, 1.0, 0.0])
    mf.add_node("learn_tree", "common", good)
    mf.add_node("learn_tree", "common", bad)
    hits = mf.recall_by_vector(q, limit=2)
    assert hits[0].moment_id == good.moment_id


def test_build_memory_context_snippets():
    s = _build_memory_context_snippets(
        {
            "recalled_memories": [
                {"content": "memory snippet alpha content"},
                {"content": "memory snippet beta content"},
            ],
            "_llm_supplement": ["supplement line for context wiring test"],
        }
    )
    assert "supplement line" in s
    assert "memory snippet alpha" in s
