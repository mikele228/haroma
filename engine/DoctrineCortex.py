from typing import Dict, List, Any, Optional
import time
from utils.module_base import ModuleBase
from core.KnowledgeBase import (
    DoctrineSynthesizer,
    DoctrineConflictDetector,
    DoctrineStabilityEvaluator,
)
from core.SymbolicUtils import SymbolUtils


class DoctrineCortex(ModuleBase):
    """
    Tier MAX++ Engine: Unified doctrine processor.
    Synthesizes symbolic doctrines, manages value vectors,
    detects cross-doctrine conflict, and reinforces symbolic consistency.
    """

    def __init__(self, memory_tree: Optional[Any] = None, agent_id: Optional[str] = None):
        super().__init__("DoctrineCortex")
        self.agent_id = agent_id
        self.memory_tree = memory_tree

        # Sub-engines
        self.synth = DoctrineSynthesizer()
        self.conflict = DoctrineConflictDetector()
        self.stability = DoctrineStabilityEvaluator()
        self.utils = SymbolUtils()

        # Persistent store
        self.doctrines: Dict[str, Dict[str, Any]] = {}
        self.values: Dict[str, float] = {}
        self.conflict_log: List[Dict[str, Any]] = []
        self._max_doctrines = 200
        self._max_conflict_log = 500

    def run_cycle(
        self,
        identity_history: List[str],
        emotion_history: List[str],
        goal_tags: List[str],
        narrative_themes: List[str],
        doctrine_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        doctrine = self.synth.synthesize(
            identity_history, emotion_history, goal_tags, narrative_themes
        )
        conflicts = self.conflict.detect(doctrine)
        stability_score = self.stability.evaluate(doctrine_history)

        # Store doctrine
        timestamp = time.time()
        doctrine_id = f"doctrine_{int(timestamp)}"
        self.doctrines[doctrine_id] = {
            "values": doctrine.get("values", {}),
            "summary": doctrine.get("summary", "auto_generated"),
            "timestamp": timestamp,
        }
        if len(self.doctrines) > self._max_doctrines:
            oldest = sorted(self.doctrines, key=lambda k: self.doctrines[k].get("timestamp", 0))
            for k in oldest[: len(self.doctrines) - self._max_doctrines]:
                del self.doctrines[k]

        return {
            "agent_id": self.agent_id,
            "current_doctrine": doctrine,
            "conflicts_detected": conflicts,
            "stability_score": stability_score,
            "doctrine_id": doctrine_id,
        }

    def compare_doctrines(self, doctrine_a: str, doctrine_b: str) -> Dict[str, Any]:
        a_values = self.doctrines.get(doctrine_a, {}).get("values", {})
        b_values = self.doctrines.get(doctrine_b, {}).get("values", {})
        score = self.utils.value_vector_distance(a_values, b_values)

        entry = {
            "doctrine_a": doctrine_a,
            "doctrine_b": doctrine_b,
            "conflict_score": round(score, 3),
            "timestamp": time.time(),
        }
        self.conflict_log.append(entry)
        if len(self.conflict_log) > self._max_conflict_log:
            self.conflict_log = self.conflict_log[-self._max_conflict_log :]
        return entry

    def define_doctrine(self, name: str, value_weights: Dict[str, float], summary: str = ""):
        """Define a named doctrine with explicit value weights."""
        self.doctrines[name] = {
            "values": dict(value_weights),
            "summary": summary or name,
            "timestamp": time.time(),
        }
        if len(self.doctrines) > self._max_doctrines:
            oldest = sorted(self.doctrines, key=lambda k: self.doctrines[k].get("timestamp", 0))
            for k in oldest[: len(self.doctrines) - self._max_doctrines]:
                del self.doctrines[k]

    def evaluate_conflict(self, doctrine_a: str, doctrine_b: str) -> Dict[str, Any]:
        """Evaluate conflict between two named doctrines."""
        return self.compare_doctrines(doctrine_a, doctrine_b)

    def reinforce_value(self, key: str, weight: float):
        self.values[key] = round(min(1.0, max(0.0, weight)), 3)

    def summarize(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "doctrine_count": len(self.doctrines),
            "value_keys": list(self.values.keys()),
            "values": dict(self.values),
            "recent_conflict": self.conflict_log[-1] if self.conflict_log else None,
            "conflict_log_size": len(self.conflict_log),
        }

    _MAX_PROMPT_VALUES = 20

    def summarize_for_prompt(self) -> Dict[str, Any]:
        """Compact snapshot suitable for LLM prompt injection (size-capped)."""
        vals = dict(self.values)
        if len(vals) > self._MAX_PROMPT_VALUES:
            vals = dict(sorted(vals.items(), key=lambda kv: -kv[1])[: self._MAX_PROMPT_VALUES])
        active_doctrines = []
        for name, d in sorted(
            self.doctrines.items(),
            key=lambda kv: kv[1].get("timestamp", 0),
            reverse=True,
        )[:5]:
            active_doctrines.append(
                {
                    "name": name,
                    "summary": str(d.get("summary", ""))[:120],
                    "values": d.get("values", {}),
                }
            )
        return {
            "value_keys": list(vals.keys()),
            "values": vals,
            "active_doctrines": active_doctrines,
        }

    def reset(self):
        self.values.clear()
        self.doctrines.clear()
        self.conflict_log.clear()

    def __repr__(self):
        return f"<DoctrineCortex doctrines={len(self.doctrines)} values={len(self.values)}>"
