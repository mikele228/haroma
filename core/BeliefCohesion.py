from utils.module_base import ModuleBase
from typing import Dict, Any, Optional, List


# === Tier 84 Core Module: BeliefOverlapAnalyzer (MAX UPGRADE) ===


class BeliefOverlapAnalyzer(ModuleBase):
    """
    Tier 84 Module: Analyzes symbolic belief alignment across agents.
    Detects shared beliefs (consensus) and unique contradictions per identity.
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(module_name="BeliefOverlapAnalyzer")
        self.context = context

    def analyze(self, belief_map: Dict[str, List[str]]) -> Dict[str, Any]:
        all_beliefs = [b for beliefs in belief_map.values() for b in beliefs]
        belief_counts = {b: all_beliefs.count(b) for b in set(all_beliefs)}

        shared_beliefs = sorted([b for b, count in belief_counts.items() if count > 1])

        contradictions = {
            aid: [b for b in beliefs if b not in shared_beliefs]
            for aid, beliefs in belief_map.items()
        }

        # Log belief overlap analysis
        self.signal_history.append(
            {
                "event": "belief_overlap_analyzed",
                "cycle_ts": (self.context or {}).get("cycle_ts", "unknown"),
                "shared_belief_count": len(shared_beliefs),
                "agent_count": len(belief_map),
            }
        )

        return {"shared_beliefs": shared_beliefs, "contradictions": contradictions}


from utils.module_base import ModuleBase
from typing import Dict, Any, Optional, List
from collections import Counter


# === Tier 84 Core Module: ConsensusSynthesizer (MAX UPGRADE) ===


class ConsensusSynthesizer(ModuleBase):
    """
    Tier 84 Module: Synthesizes cross-agent belief consensus by applying minimum support thresholds.
    Produces unified symbolic axioms supported by a minimum number of agents.
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(module_name="ConsensusSynthesizer")
        self.context = context

    def synthesize(self, belief_map: Dict[str, List[str]], min_support: int = 2) -> List[str]:
        all_beliefs = []
        for beliefs in belief_map.values():
            all_beliefs.extend(set(beliefs))  # Deduplicates per agent

        belief_counts = Counter(all_beliefs)
        consensus = sorted([b for b, c in belief_counts.items() if c >= min_support])

        # Log consensus synthesis
        self.signal_history.append(
            {
                "event": "belief_consensus_synthesized",
                "cycle_ts": (self.context or {}).get("cycle_ts", "unknown"),
                "threshold": min_support,
                "consensus_count": len(consensus),
                "agent_count": len(belief_map),
            }
        )

        return consensus


from utils.module_base import ModuleBase
from typing import Dict, Any, Optional, List


# === Tier 84 Core Module: DisagreementClassifier (MAX UPGRADE) ===


class DisagreementClassifier(ModuleBase):
    """
    Tier 84 Module: Identifies belief disagreements by comparing agent beliefs
    to the shared consensus pool. Outputs unresolved beliefs per agent.
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(module_name="DisagreementClassifier")
        self.context = context

    def classify(
        self, belief_map: Dict[str, List[str]], consensus: List[str]
    ) -> Dict[str, List[str]]:
        unresolved = {
            aid: [b for b in beliefs if b not in consensus] for aid, beliefs in belief_map.items()
        }

        # Log disagreement classification
        self.signal_history.append(
            {
                "event": "belief_disagreement_classified",
                "cycle_ts": (self.context or {}).get("cycle_ts", "unknown"),
                "agent_count": len(belief_map),
                "max_disagreements": max(len(v) for v in unresolved.values()) if unresolved else 0,
            }
        )

        return unresolved


from utils.module_base import ModuleBase
from typing import Dict, Any, Optional, List


class SchemaConflictResolver(ModuleBase):
    """
    Tier 90 Module: Resolves divergence between schema memory patterns and present symbolic state.
    Suggests belief deprecation, role reclassification, and contradiction synthesis.
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__("SchemaConflictResolver")
        self.context = context

    def resolve(
        self, schema: Dict[str, Any], current_beliefs: List[str], current_role: str
    ) -> List[Dict[str, Any]]:
        updates = []

        # Handle belief deprecation
        for belief in schema.get("dominant_beliefs", []):
            if belief not in current_beliefs:
                updates.append(
                    {
                        "action": "deprecate_belief",
                        "target": belief,
                        "reason": "no longer reinforced by current beliefs",
                    }
                )

        # Handle role reclassification
        for role in schema.get("frequent_roles", []):
            if role != current_role:
                updates.append(
                    {
                        "action": "reclassify_role_context",
                        "target": role,
                        "reason": "role mismatch with schema history",
                    }
                )

        # Handle contradiction integration
        for tag in schema.get("recurring_contradictions", []):
            updates.append(
                {
                    "action": "introduce_synthesis_rule",
                    "target": tag,
                    "reason": "persistent contradiction requires symbolic integration",
                }
            )

        # Log internal schema resolution
        self.signal_history.append(
            {
                "event": "schema_conflict_resolved",
                "total_updates": len(updates),
                "context_role": current_role,
            }
        )

        return updates


from utils.module_base import ModuleBase
from typing import Dict, Any, List, Optional
import time


class BeliefDriftMonitor(ModuleBase):
    """
    MAX++ Module: Tracks the evolution of belief and goal states over time.
    Allows detection of outdated goals, changed assumptions, or belief drift.
    """

    def __init__(self):
        super().__init__(module_name=self.__class__.__name__)
        self.history: List[Dict[str, Any]] = []

    def ingest(self, entry: Dict[str, Any], kind: str) -> None:
        """
        Records a new belief or goal with timestamp and type.
        """
        snapshot = {"timestamp": time.time(), "type": kind, "logic": entry}
        self.history.append(snapshot)

    def latest(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves the most recent belief or goal of the given type.
        """
        filtered = [e for e in reversed(self.history) if kind is None or e["type"] == kind]
        return filtered[:1] if kind else filtered

    def detect_drift(self, current: Dict[str, Any], kind: str = "belief") -> bool:
        """
        Compares a new input against the most recent of same kind.
        Returns True if drift (contradiction or strong change) is detected.
        """
        recent = self.latest(kind)
        if not recent:
            return False
        last_logic = recent[0]["logic"]
        if current.get("THEN") != last_logic.get("THEN"):
            return True
        if current.get("NEGATION") != last_logic.get("NEGATION"):
            return True
        if current.get("MODALITY") != last_logic.get("MODALITY"):
            return True
        return False

    def get_narrative(self) -> List[Dict[str, Any]]:
        """
        Returns full history as a narrative timeline.
        """
        return sorted(self.history, key=lambda x: x["timestamp"])


from utils.module_base import ModuleBase
from typing import Union, Dict, List, Set


class SymbolicCohesionScorer(ModuleBase):
    def __init__(self):
        super().__init__("SymbolicCohesionScorer")

    def score(
        self,
        a: Union[Dict[str, any], List[str], Set[str]],
        b: Union[Dict[str, any], List[str], Set[str]],
    ) -> float:
        """
        Computes symbolic cohesion between two structures (dicts or sets/lists of tags).
        Returns a score between 0.0 and 1.0 based on key/tag overlap.
        """
        if not a or not b:
            return 0.0

        a_keys = set(a.keys()) if isinstance(a, dict) else set(a)
        b_keys = set(b.keys()) if isinstance(b, dict) else set(b)

        overlap = a_keys & b_keys
        union = a_keys | b_keys

        return round(len(overlap) / max(len(union), 1), 3)


from utils.module_base import ModuleBase
from typing import Dict, Any, Optional, List, Union


class CoherenceEvaluator:
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        self.context = context
        self.scorer = SymbolicCohesionScorer()

    def evaluate(self, symbolic_state: Dict[str, Any]) -> Dict[str, float]:
        # Extract symbolic layers with default fallbacks
        goal = symbolic_state.get("goal", {})
        belief = symbolic_state.get("belief", {})
        identity = symbolic_state.get("identity", [])
        narrative = symbolic_state.get("narrative", [])
        emotion = symbolic_state.get("emotion", [])

        # Score each coherence relationship
        goal_belief_score = self.scorer.score(goal, belief)
        identity_narrative_score = self.scorer.score(identity, narrative)
        emotion_identity_score = self.scorer.score(emotion, identity)

        # Compute overall coherence (mean of available scores)
        scores = [goal_belief_score, identity_narrative_score, emotion_identity_score]
        valid_scores = [s for s in scores if s is not None]
        overall_score = round(sum(valid_scores) / len(valid_scores), 3) if valid_scores else 0.0

        return {
            "goal_belief": goal_belief_score,
            "identity_narrative": identity_narrative_score,
            "emotion_identity": emotion_identity_score,
            "overall": overall_score,
        }


from utils.module_base import ModuleBase
from typing import Dict, Any, List, Set, Optional
import time


class BeliefReflector(ModuleBase):
    """
    Tier MAX+ Composite: Unified belief engine.
    Integrates tracking, clarification, expansion, contradiction resolution, and affect binding.
    """

    def __init__(self, context_engine: Optional[Any] = None):
        super().__init__("BeliefReflector")
        self.context_engine = context_engine
        self.load_state()

        self.tags: Set[str] = set(self.state.get("tags", []))
        self.emotional_trace: List[Any] = self.state.get("emotional_trace", [])
        self.history: List[Dict[str, Any]] = self.state.get("history", [])
        self.beliefs: Dict[str, Dict[str, Any]] = self.state.get("beliefs", {})
        self.conflicts: List[Dict[str, Any]] = self.state.get("conflicts", [])
        self.threads: Dict[str, List[str]] = self.state.get("threads", {})
        self.schema: Dict[str, List[str]] = self.state.get("schemas", {})

    def resolve_belief_conflict(self, belief_a, belief_b):
        if self.context_engine:
            return self.context_engine.context_truth_resolver_handler(
                {"belief_a": belief_a, "belief_b": belief_b}
            )
        return {"resolution": "unknown", "reason": "no context engine"}

    def ingest(self, input_data: Dict[str, Any]) -> None:
        self.history.append(input_data)
        tags = input_data.get("tags", [])
        self.tags.update(tags)
        if "emotion" in input_data:
            self.emotional_trace.append(input_data["emotion"])
        for k, v in input_data.items():
            if k != "tags":
                self.state[k] = v
        self._persist()

    def add_belief(self, key: str, content: str, confidence: float = 0.7):
        self.beliefs[key] = {
            "content": content,
            "confidence": confidence,
            "layer": content.count("because"),
            "rationale": "auto",
        }
        self.state["beliefs"] = self.beliefs

    def clarify_belief(self, key: str) -> Dict[str, Any]:
        belief = self.beliefs.get(key, {})
        return {
            "key": key,
            "confidence": belief.get("confidence"),
            "layer": belief.get("layer"),
            "tags": list(self.tags),
            "emotion_tail": self.emotional_trace[-3:],
        }

    def bind_to_thread(self, belief_key: str, thread: str):
        if thread not in self.threads:
            self.threads[thread] = []
        if belief_key not in self.threads[thread]:
            self.threads[thread].append(belief_key)
        self.state["threads"] = self.threads

    def assign_schema(self, belief_key: str, schema_name: str):
        if schema_name not in self.schema:
            self.schema[schema_name] = []
        if belief_key not in self.schema[schema_name]:
            self.schema[schema_name].append(belief_key)
        self.state["schemas"] = self.schema

    def detect_conflicts(self) -> List[Dict[str, Any]]:
        self.conflicts.clear()
        keys = list(self.beliefs.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = self.beliefs[keys[i]], self.beliefs[keys[j]]
                if a["content"] != b["content"]:
                    self.conflicts.append(
                        {"a": keys[i], "b": keys[j], "a_val": a["content"], "b_val": b["content"]}
                    )
        self.state["conflicts"] = self.conflicts
        return self.conflicts

    def summarize(self) -> Dict[str, Any]:
        return {
            "belief_count": len(self.beliefs),
            "conflicts": len(self.conflicts),
            "schemas": list(self.schema.keys()),
            "threads": len(self.threads),
            "tags": list(self.tags),
            "recent_emotion": self.emotional_trace[-3:],
        }

    def _persist(self):
        self.state["tags"] = list(self.tags)
        self.state["emotional_trace"] = self.emotional_trace
        self.state["history"] = self.history
        self.state["beliefs"] = self.beliefs
        self.state["conflicts"] = self.conflicts
        self.state["threads"] = self.threads
        self.state["schemas"] = self.schema

    def reset(self):
        self.tags.clear()
        self.emotional_trace.clear()
        self.history.clear()
        self.beliefs.clear()
        self.conflicts.clear()
        self.threads.clear()
        self.schema.clear()
        self._persist()

    def __repr__(self):
        return f"<BeliefReflector beliefs={len(self.beliefs)} tags={len(self.tags)}>"
