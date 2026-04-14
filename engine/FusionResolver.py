from typing import Dict, Any, List
from utils.module_base import ModuleBase


class FusionResolver(ModuleBase):
    """Fuses symbolic data structures and resolves conflicts between clusters."""

    _MAX_HISTORY = 500

    def __init__(self):
        super().__init__("FusionResolver")
        self.history: List[Dict[str, Any]] = []

    def fuse(self, a: Dict[str, Any], b: Dict[str, Any], mode: str = "merge") -> Dict[str, Any]:
        if mode == "merge":
            merged = {**a, **b}
        elif mode == "priority_a":
            merged = {**b, **a}
        else:
            merged = {**a, **b}
        self.history.append({"mode": mode, "keys_a": list(a.keys()), "keys_b": list(b.keys())})
        if len(self.history) > self._MAX_HISTORY:
            self.history = self.history[-self._MAX_HISTORY :]
        return merged

    def resolve_conflicts(self, cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not cluster:
            return {}
        result = {}
        for item in cluster:
            result.update(item)
        return result

    def summarize(self) -> Dict[str, Any]:
        return {"fusions": len(self.history)}

    def reset(self):
        self.history.clear()
