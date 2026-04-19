"""Materialize curated facts into MemoryForest for Nexus-style graphs.

Writes to ``belief_tree`` / ``nexus_sync`` with ``prime`` and ``nexus`` tags so
``MemoryForest.merge_recall_with_prime`` surfaces them under budget.
"""

from __future__ import annotations

from typing import Any, List, Optional, TYPE_CHECKING

from core.Memory import MemoryNode

if TYPE_CHECKING:
    from core.Memory import MemoryForest


def nexus_uri_tag(uri: str) -> str:
    u = str(uri).strip()[:512]
    return f"nexus_id:{u}" if u else "nexus_id:unknown"


def ingest_nexus_fact(
    memory: "MemoryForest",
    content: str,
    *,
    nexus_uri: str,
    vector: Optional[Any] = None,
    extra_tags: Optional[List[str]] = None,
    confidence: float = 0.95,
    emotion: Optional[str] = None,
) -> MemoryNode:
    """Add a prime memory row suitable for Blue Brain Nexus–sourced data."""
    tags = ["prime", "nexus", nexus_uri_tag(nexus_uri)]
    if extra_tags:
        tags.extend(str(t) for t in extra_tags if t)
    node = MemoryNode(
        content=str(content).strip()[:4000],
        emotion=emotion,
        confidence=confidence,
        tags=tags,
    )
    if vector is not None:
        try:
            node.set_vector(vector)
        except (TypeError, ValueError):
            pass
    memory.add_node("belief_tree", "nexus_sync", node)
    return node
