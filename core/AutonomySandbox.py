"""
AutonomySandbox — allow-listed, budget-limited tools for self-initiated steps.

Phase 5: autonomous code can only invoke named tools here (expand allowlist
explicitly). Prevents unbounded side effects from background ticks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

from core.cognitive_null import is_cognitive_null


class AutonomySandbox:
    """Small tool surface for autonomy; all calls consume session budget."""

    def __init__(self, session_budget: int = 200):
        self.allowlist: Set[str] = {
            "echo",
            "memory_node_count",
            "top_goal_snippet",
        }
        self.session_budget = max(1, int(session_budget))
        self.used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "used": self.used,
            "session_budget": self.session_budget,
            "allowlist": sorted(self.allowlist),
        }

    def from_dict(self, data: Dict[str, Any]):
        try:
            self.used = int(data.get("used", 0))
        except (TypeError, ValueError):
            self.used = 0
        try:
            self.session_budget = max(1, int(data.get("session_budget", 200)))
        except (TypeError, ValueError):
            self.session_budget = 200
        al = data.get("allowlist")
        if isinstance(al, list):
            self.allowlist = set(str(x) for x in al)

    def try_execute(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
        context: Optional[Any] = None,
    ) -> Dict[str, Any]:
        params = params or {}
        if name not in self.allowlist:
            return {"ok": False, "error": "not_allowlisted"}
        if self.used >= self.session_budget:
            return {"ok": False, "error": "session_budget_exhausted"}
        self.used += 1

        if name == "echo":
            return {"ok": True, "result": str(params.get("text", ""))}

        if name == "memory_node_count":
            mem = getattr(context, "memory", None) if context else None
            if mem is None or is_cognitive_null(mem):
                return {"ok": False, "error": "no_memory"}
            try:
                n = mem.count_nodes()
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "result": n}

        if name == "top_goal_snippet":
            goal = getattr(context, "goal", None) if context else None
            if goal is None or is_cognitive_null(goal):
                return {"ok": False, "error": "no_goal"}
            try:
                eng = goal.engine
                pri = eng.prioritize() if hasattr(eng, "prioritize") else []
                for gid in pri[:1]:
                    g = eng.goals.get(gid)
                    if isinstance(g, dict):
                        d = str(g.get("description", ""))[:200]
                    else:
                        d = str(getattr(g, "description", "") or "")[:200]
                    if d.strip():
                        return {"ok": True, "result": d}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "result": ""}

        return {"ok": False, "error": "no_handler"}
