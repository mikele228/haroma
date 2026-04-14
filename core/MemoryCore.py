"""MemoryCore — thin facade over MemoryForest providing the X7 multi-agent API.

All state still lives in MemoryForest; this layer adds agent-branch semantics:
 * Each symbolic tree has a ``common`` branch for the reconciled view.
 * Per-agent branches are named ``agent:<agent_id>``.
 * ``commit_agent_tree`` atomically replaces ``common`` with reconciled nodes
   and clears the agent branches that were merged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.Memory import MemoryForest, MemoryNode


COMMON_BRANCH = "common"
AGENT_PREFIX = "agent:"

RECONCILABLE_TREES = [
    "belief_tree",
    "goal_tree",
    "value_tree",
    "emotion_tree",
    "dream_tree",
    "encounter_tree",
    "thought_tree",
    "action_tree",
]


class MemoryCore:
    """High-level multi-agent memory API wrapping :class:`MemoryForest`."""

    def __init__(self, forest: MemoryForest):
        self._forest = forest

    @property
    def forest(self) -> MemoryForest:
        return self._forest

    # ── read helpers ─────────────────────────────────────────────────

    def get_context(self, tree_name: str, agent_id: Optional[str] = None) -> List[MemoryNode]:
        """Return merged common + agent branch nodes (or just common)."""
        nodes = list(self._forest.get_nodes(tree_name, COMMON_BRANCH))
        if agent_id:
            branch = f"{AGENT_PREFIX}{agent_id}"
            nodes.extend(self._forest.get_nodes(tree_name, branch))
        return nodes

    def list_agent_branches(self, tree_name: str) -> List[str]:
        """Return branch names that start with ``agent:`` for *tree_name*."""
        tree = self._forest.get_tree(tree_name)
        if not tree:
            return []
        return [b for b in tree.branches if b.startswith(AGENT_PREFIX)]

    def get_identity(self) -> List[MemoryNode]:
        """Read-only aggregate of identity_tree common + soul branches."""
        tree = self._forest.get_tree("identity_tree")
        if not tree:
            return []
        with self._forest._lock:
            branches_snap = list(tree.branches.values())
        nodes: List[MemoryNode] = []
        for branch in branches_snap:
            nodes.extend(branch.nodes)
        return nodes

    # ── write helpers ────────────────────────────────────────────────

    def set_context(
        self,
        tree_name: str,
        content: str,
        *,
        agent_id: Optional[str] = None,
        emotion: Optional[str] = None,
        confidence: float = 1.0,
        tags: Optional[List[str]] = None,
    ) -> MemoryNode:
        """Write a MemoryNode to the common or agent branch."""
        branch = f"{AGENT_PREFIX}{agent_id}" if agent_id else COMMON_BRANCH
        node = MemoryNode(
            content=content,
            emotion=emotion,
            confidence=confidence,
            tags=tags or [],
        )
        self._forest.add_node(tree_name, branch, node)
        return node

    def commit_agent_tree(self, tree_name: str, new_common_nodes: List[MemoryNode]) -> int:
        """Atomically replace ``common`` with reconciled nodes and clear agent branches.

        Holds the forest lock for the entire operation so readers never see
        a partially-committed state.

        Returns the number of agent-branch nodes removed.
        """
        with self._forest._lock:
            removed = 0

            agent_branches = self.list_agent_branches(tree_name)
            for ab in agent_branches:
                removed += self._clear_agent_branch_locked(tree_name, ab)

            old_common = list(self._forest.get_nodes(tree_name, COMMON_BRANCH))
            for node in old_common:
                self._forest.remove_node(node.moment_id)

            for node in new_common_nodes:
                self._forest.add_node(tree_name, COMMON_BRANCH, node)

            return removed

    def clear_agent_branch(self, tree_name: str, branch_name: str) -> int:
        """Remove all nodes from a specific branch and delete it. Returns count removed."""
        with self._forest._lock:
            return self._clear_agent_branch_locked(tree_name, branch_name)

    def _clear_agent_branch_locked(self, tree_name: str, branch_name: str) -> int:
        """Inner clear -- caller MUST already hold ``self._forest._lock``."""
        nodes = list(self._forest.get_nodes(tree_name, branch_name))
        count = 0
        for node in nodes:
            if self._forest.remove_node(node.moment_id):
                count += 1
        tree = self._forest.get_tree(tree_name)
        if tree and branch_name in tree.branches:
            del tree.branches[branch_name]
        return count

    # ── introspection ────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Per-tree summary of common/agent node counts."""
        out: Dict[str, Any] = {}
        for tname in RECONCILABLE_TREES:
            tree = self._forest.get_tree(tname)
            if not tree:
                out[tname] = {"common": 0, "agents": {}}
                continue
            with self._forest._lock:
                branches_snapshot = dict(tree.branches)
            common_branch = branches_snapshot.get(COMMON_BRANCH)
            common_count = len(common_branch.nodes) if common_branch else 0
            agents: Dict[str, int] = {}
            for bname, branch in branches_snapshot.items():
                if bname.startswith(AGENT_PREFIX):
                    agents[bname] = len(branch.nodes)
            out[tname] = {"common": common_count, "agents": agents}
        return out
