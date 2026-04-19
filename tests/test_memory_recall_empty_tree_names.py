"""recall(..., tree_names=frozenset()) returns []."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.Memory import MemoryForest, MemoryNode


def test_recall_empty_tree_names_returns_empty():
    mf = MemoryForest(encoder=None)
    mf.add_node("cmem", "a", MemoryNode(content="hello world", tags=[]))
    assert mf.recall(query_text="hello", limit=5, tree_names=frozenset()) == []
