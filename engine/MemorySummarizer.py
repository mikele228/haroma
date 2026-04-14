from typing import Dict, Any, List, Optional
from utils.module_base import ModuleBase


class MemorySummarizer(ModuleBase):
    """Summarizes memory moments and forests."""

    def __init__(self, forest=None):
        super().__init__("MemorySummarizer")
        self.forest = forest

    def summarize_moment(self, moment_id: str) -> Dict[str, Any]:
        if not self.forest:
            return {"moment_id": moment_id, "summary": "no forest"}
        nodes = self.forest.get_nodes_by_moment(moment_id)
        return {
            "moment_id": moment_id,
            "trees": list(nodes.keys()),
            "total_nodes": sum(len(v) for v in nodes.values()),
        }

    def summarize_all(self, limit: int = 100) -> List[Dict[str, Any]]:
        if not self.forest:
            return []
        summaries = []
        for tree_name, tree in list(self.forest.trees.items())[:limit]:
            total_nodes = sum(len(b.nodes) for b in tree.branches.values())
            summaries.append(
                {
                    "tree": tree_name,
                    "branches": list(tree.branches.keys()),
                    "total_nodes": total_nodes,
                }
            )
        return summaries
