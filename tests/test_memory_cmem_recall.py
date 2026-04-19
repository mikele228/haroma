"""cmem tree filtering on MemoryForest.recall / merge_recall_with_prime."""

from __future__ import annotations

import sys

import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.Memory import MemoryForest, MemoryNode, SemanticIndex


def test_recall_tree_names_filters_to_cmem():
    mf = MemoryForest(encoder=None)
    n_th = MemoryNode(content="alpha beta gamma thought_line", tags=["episodic"])
    n_cm = MemoryNode(content="alpha beta gamma cmem_line", tags=["cmem", "consolidated"])
    mf.add_node("thought_tree", "analyst", n_th)
    mf.add_node("cmem", "analyst", n_cm)
    out = mf.recall(
        query_text="alpha beta gamma",
        limit=10,
        tree_names=frozenset({"cmem"}),
        tree_recall_max_probe=500,
    )
    assert out
    assert all(getattr(n, "tree", None) == "cmem" for n in out)
    ids = {n.moment_id for n in out}
    assert n_cm.moment_id in ids
    assert n_th.moment_id not in ids


def test_merge_recall_prepend_prime_false_skips_prime():
    mf = MemoryForest(encoder=None)
    n = MemoryNode(content="only recalled", tags=[])
    merged = mf.merge_recall_with_prime([n], limit=6, prepend_prime=False)
    assert len(merged) == 1
    assert merged[0].moment_id == n.moment_id


def test_recall_cmem_calls_tree_dense_when_global_index_unready(monkeypatch):
    """``recall`` must use ``query_dense_only_for_trees`` for cmem-only even when the main dense index is not ready."""
    mf = MemoryForest(encoder=None)
    si = mf.semantic_index
    if not si._dense_available or si._sbert is None:
        pytest.skip("SBERT not available in this environment")

    calls: list[int] = []

    def fake_qdft(self, text, limit, allowed_trees, max_probe):
        calls.append(1)
        return []

    monkeypatch.setattr(SemanticIndex, "query_dense_only_for_trees", fake_qdft)
    with si._lock:
        si._dense_vectors = None
        si._dense_indexed = 0
        si._faiss_index = None
        si._dirty = True
    mf.recall(query_text="hello", limit=5, tree_names=frozenset({"cmem"}))
    assert calls == [1]


def test_cmem_only_short_circuits_to_subindex(monkeypatch):
    """cmem-only ``query_dense_only_for_trees`` must not gate on the global dense matrix."""
    mf = MemoryForest(encoder=None)
    si = mf.semantic_index
    calls: list[tuple[str, int]] = []

    def fake(self, text: str, limit: int):
        calls.append((text, limit))
        return []

    monkeypatch.setattr(SemanticIndex, "_query_cmem_dense_subindex", fake)
    with si._lock:
        si._dense_vectors = None
        si._dense_indexed = 0
        si._faiss_index = None
    si.query_dense_only_for_trees("q", 5, frozenset({"cmem"}), 100)
    assert calls == [("q", 5)]


def test_cmem_only_dense_uses_subindex_without_global_dense():
    """When SBERT is available, cmem-only dense query works without the main FAISS index."""
    mf = MemoryForest(encoder=None)
    si = mf.semantic_index
    if not si._dense_available or si._sbert is None:
        pytest.skip("SBERT not available in this environment")
    n_cm = MemoryNode(content="unique cmem phrase xyz789subidx", tags=["t1"])
    mf.add_node("cmem", "b", n_cm)
    with si._lock:
        si._dense_vectors = None
        si._dense_indexed = 0
        si._faiss_index = None
    out = si.query_dense_only_for_trees(
        "unique cmem phrase",
        limit=5,
        allowed_trees=frozenset({"cmem"}),
        max_probe=100,
    )
    assert out
    ids = {n.moment_id for _, n in out}
    assert n_cm.moment_id in ids
    st = si.stats()
    assert st.get("cmem_subindex_nodes", 0) >= 1
    assert st.get("cmem_subindex_dirty") is False
