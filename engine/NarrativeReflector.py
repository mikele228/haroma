from typing import Dict, Any, List, Optional
from utils.module_base import ModuleBase


class NarrativeReflector(ModuleBase):
    """Builds narrative threads and timelines from memory forests."""

    def __init__(self, forest=None):
        super().__init__("NarrativeReflector")
        self.forest = forest

    def build_narrative_thread(self, moment_ids: List[str]) -> List[Dict[str, Any]]:
        if not self.forest:
            return []
        thread = []
        for mid in moment_ids:
            nodes = self.forest.get_nodes_by_moment(mid)
            thread.append({"moment_id": mid, "trees": list(nodes.keys())})
        return thread

    def build_full_narrative(self) -> List[Dict[str, Any]]:
        if not self.forest:
            return []
        narrative = []
        for tree_name, tree in self.forest.trees.items():
            for branch_name, branch in tree.branches.items():
                for node in branch.nodes[-5:]:
                    narrative.append(
                        {
                            "tree": tree_name,
                            "branch": branch_name,
                            "content": node.content,
                            "emotion": node.emotion,
                        }
                    )
        return narrative

    def export_timeline(self, as_text: bool = False):
        narrative = self.build_full_narrative()
        if as_text:
            return "\n".join(f"[{n['tree']}] {n['content']}" for n in narrative)
        return narrative

    def _summarize_moment(self, moment_id: str) -> Dict[str, Any]:
        return {"moment_id": moment_id, "summary": "narrative_moment"}
