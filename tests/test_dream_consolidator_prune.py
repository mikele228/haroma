"""DreamConsolidator — prune only dream_tree; episodic nodes remain."""

import time

from core.DreamConsolidator import DreamConsolidator
from core.Memory import MemoryForest, MemoryNode


def test_prune_only_dream_tree_keeps_thought_tree():
    mf = MemoryForest(encoder=None)
    dc = DreamConsolidator(mf, encoder=None)
    old_ts = time.time() - 3600.0

    keep = MemoryNode(content="episodic keep", confidence=0.1, tags=["experience"])
    keep.timestamp = old_ts
    mf.add_node("thought_tree", "analyst", keep)

    drop = MemoryNode(content="dream fog", confidence=0.1, tags=["dream"])
    drop.timestamp = old_ts
    mf.add_node("dream_tree", "dreamer", drop)

    assert dc._prune() == 1
    assert mf.get_node_by_id(keep.moment_id) is not None
    assert mf.get_node_by_id(drop.moment_id) is None
