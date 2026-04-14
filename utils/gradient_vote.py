# === plugin_gradient_vote.py ===
# Plugin to choose the best thought based on gradient field from harmonized thoughts

from typing import List, Dict, Any


class GradientVote:
    def vote(self, thoughts: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not thoughts:
            return {}

        best = None
        max_gradient = float("-inf")

        for t in thoughts:
            g = t.get("gradient", 0.0)
            if isinstance(g, dict):
                g = g.get("score", 0.0)
            try:
                g = float(g)
            except (TypeError, ValueError):
                g = 0.0
            if g > max_gradient:
                max_gradient = g
                best = t

        return best or thoughts[0]
