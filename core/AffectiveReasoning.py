from typing import Dict, List, Any
from core.Memory import MemoryForest
from collections import defaultdict


class EmotionBeliefDriftDetector:
    def __init__(self, forest: MemoryForest):
        self.forest = forest

    def analyze_drift(self, moment_ids: List[str]) -> Dict[str, Any]:
        drift_map = defaultdict(float)
        mismatched = []

        for moment_id in moment_ids:
            cross_tree_nodes = self.forest.get_nodes_by_moment(moment_id)
            emotion = None
            belief_confidence = None

            for tree_name, nodes in cross_tree_nodes.items():
                for node in nodes:
                    if tree_name == "belief_tree" and node.confidence is not None:
                        belief_confidence = node.confidence
                    if node.emotion:
                        emotion = node.emotion

            if belief_confidence is not None and emotion:
                alignment = self._emotion_confidence_alignment(emotion, belief_confidence)
                drift_map[moment_id] = alignment
                if alignment < 0.2:
                    mismatched.append(
                        {
                            "moment_id": moment_id,
                            "confidence": belief_confidence,
                            "emotion": emotion,
                            "alignment_score": round(alignment, 2),
                        }
                    )

        return {
            "drift_scores": dict(drift_map),
            "mismatches": mismatched,
            "avg_alignment": round(sum(drift_map.values()) / len(drift_map), 3)
            if drift_map
            else 0.0,
        }

    def _emotion_confidence_alignment(self, emotion: str, confidence: float) -> float:
        polarity = {
            "joy": +1.0,
            "peace": +0.8,
            "curiosity": +0.6,
            "neutral": 0.5,
            "confusion": -0.3,
            "fear": -0.7,
            "anger": -1.0,
            "sadness": -0.9,
        }
        emotional_value = polarity.get(emotion.lower(), 0.0)
        return max(0.0, 1.0 - abs(emotional_value - confidence))


from utils.module_base import ModuleBase
from typing import Dict, Any, Tuple, Optional


class EmotionStateModel(ModuleBase):
    def __init__(self):
        super().__init__("EmotionStateModel")

    def compute(
        self,
        current_emotion: str,
        current_intensity: float,
        new_emotion: str,
        boost: float,
        dt: float,
        decay_rate: float = 0.05,
    ) -> Tuple[str, float]:
        """
        Computes the next emotional state given current state, incoming emotional influence, and elapsed time.
        """
        decayed = max(current_intensity - decay_rate * dt, 0.0)

        if new_emotion == current_emotion:
            return current_emotion, min(decayed + boost, 1.0)
        elif decayed < 0.1:
            return new_emotion, min(boost, 1.0)
        else:
            return current_emotion, decayed


from utils.module_base import ModuleBase
from typing import Dict, Any


class EmotionReflectionEngine(ModuleBase):
    def __init__(self):
        super().__init__("EmotionReflectionEngine")

    def reflect(self, memory_context: Dict[str, Any]) -> Dict[str, Any]:
        tags = memory_context.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        emotional_signals = {
            "joy": ["success", "gratitude", "connection"],
            "sadness": ["loss", "failure", "loneliness"],
            "anger": ["conflict", "injustice", "violation"],
            "fear": ["threat", "uncertainty", "risk"],
            "neutral": [],
        }

        score_map = {emotion: 0 for emotion in emotional_signals}

        for tag in tags:
            tag_l = tag.lower()
            for emotion, triggers in emotional_signals.items():
                if tag_l in triggers:
                    score_map[emotion] += 1

        inferred_emotion = max(score_map, key=score_map.get)
        confidence = round(score_map[inferred_emotion] / max(1, len(tags)), 3)

        return {"inferred_emotion": inferred_emotion, "tags": tags, "confidence": confidence}


from utils.module_base import ModuleBase
from typing import Dict, Any, List


class EmotionDrivenResponseEngine(ModuleBase):
    def __init__(self):
        super().__init__("EmotionDrivenResponseEngine")

    def generate(self, emotion: str, tags: List[str]) -> str:
        emotion_templates = {
            "joy": "AffirmativeResponse",
            "sadness": "SoothingResponse",
            "anger": "BoundaryResponse",
            "fear": "CautiousResponse",
            "neutral": "StandardResponse",
        }

        emotion = emotion.lower() if isinstance(emotion, str) else "neutral"
        tags = [t.lower() for t in tags if isinstance(t, str)]

        base = emotion_templates.get(emotion, "StandardResponse")

        if "injustice" in tags or "violation" in tags:
            return "AssertiveBoundary" if emotion == "anger" else base
        if "connection" in tags and emotion == "joy":
            return "EmpatheticEcho"
        if "loss" in tags and emotion == "sadness":
            return "MemoryAnchor"
        return base


from utils.module_base import ModuleBase
from typing import List


class MythicEmotionBinder(ModuleBase):
    def __init__(self):
        super().__init__("MythicEmotionBinder")

    def bind(self, emotion: str, tags: List[str]) -> str:
        """
        Maps an emotion + symbolic tag pattern into a mythic archetype role.
        """
        emotion = emotion.lower() if isinstance(emotion, str) else "neutral"
        tags = [t.lower() for t in tags if isinstance(t, str)]

        pattern_map = {
            "joy": {"struggle": "Hero", "success": "Champion", "gratitude": "Guardian"},
            "sadness": {"loss": "Exile", "loneliness": "Wanderer", "grief": "Pilgrim"},
            "anger": {"injustice": "Rebel", "violation": "Avenger", "betrayal": "Renegade"},
            "fear": {"threat": "Watcher", "uncertainty": "Hermit", "risk": "Scout"},
            "neutral": {"analysis": "Observer", "pattern": "Scribe"},
        }

        for tag in tags:
            role = pattern_map.get(emotion, {}).get(tag)
            if role:
                return role

        return "Undefined"


from utils.module_base import ModuleBase
from typing import List


class EmotionIdentityTracer(ModuleBase):
    def __init__(self):
        super().__init__("EmotionIdentityTracer")

    def trace(self, emotion_history: List[str]) -> str:
        """
        Given a history of emotions, infer a symbolic identity trajectory.
        """
        if not isinstance(emotion_history, list) or not emotion_history:
            return "Undefined"

        normalized = [e.lower() for e in emotion_history if isinstance(e, str)]

        path_map = {
            "joy": "Seeker",
            "sadness": "Exile",
            "anger": "Rebel",
            "fear": "Hermit",
            "neutral": "Observer",
        }

        counts = {emo: normalized.count(emo) for emo in set(normalized)}
        dominant = max(counts, key=counts.get, default=None)

        return path_map.get(dominant, "Undefined")


import time
from typing import Dict, Any, List
from utils.module_base import ModuleBase


class EmotionCore(ModuleBase):
    """
    Tier MAX++ Sovereign Module:
    Unified emotion processor with symbolic modulation, emotional inertia,
    narrative peak tracking, and temporal reinforcement logic.
    """

    def __init__(self, context_engine: Optional[Any] = None):
        super().__init__("EmotionCore")
        self.context_engine = context_engine
        self.context = getattr(context_engine, "context", None)

        self.load_state()

        self.tags: set = set(self.state.get("tags", []))
        self.emotional_trace: List[Any] = self.state.get("emotional_trace", [])
        self.history: List[Dict[str, Any]] = self.state.get("history", [])
        self.state_core: Dict[str, float] = self.state.get(
            "state_core", {"joy": 0.2, "sadness": 0.1, "anger": 0.0, "fear": 0.0, "curiosity": 0.5}
        )
        self.peaks: Dict[str, float] = self.state.get("peaks", {})
        self.timestamps: Dict[str, float] = self.state.get("timestamps", {})
        self.slopes: Dict[str, float] = self.state.get("slopes", {})

        self._state_model = EmotionStateModel()
        self._reflector = EmotionReflectionEngine()
        self._generator = EmotionDrivenResponseEngine()

    def reflect(self, memory_context: Dict[str, Any]) -> Dict[str, Any]:
        return self._reflector.reflect(memory_context)

    def compute(self, current_emotion, current_intensity, new_emotion, boost, dt=1.0):
        return self._state_model.compute(current_emotion, current_intensity, new_emotion, boost, dt)

    def generate(self, emotion: str, tags: list) -> str:
        return self._generator.generate(emotion, tags)

    def map_emotion_to_intent(self, emotion_data: Dict[str, Any]) -> Optional[str]:
        if self.context_engine:
            return self.context_engine.emotion_intent_mapper_handler(emotion_data)
        return None

    def ingest(self, input_data: Dict[str, Any]) -> None:
        symbolic_tags = input_data.get("tags", [])
        self.tags.update(symbolic_tags)
        self.history.append(input_data)
        if len(self.history) > 500:
            self.history = self.history[-500:]

        if "emotion" in input_data:
            emotion = input_data["emotion"]
            intensity = input_data.get("intensity", 0.5)
            self.update_emotion(emotion, intensity, context=input_data)

        for k, v in input_data.items():
            if k not in {"tags", "emotion"}:
                self.state[k] = v
        self._persist()

    def update_emotion(
        self, emotion: str, intensity: float, context: Optional[Dict[str, Any]] = None
    ):
        now = time.time()
        val = max(0.0, min(1.0, intensity))
        prev_val = self.state_core.get(emotion, 0.0)
        delta = val - prev_val

        self.state_core[emotion] = val
        self.peaks[emotion] = max(self.peaks.get(emotion, 0.0), val)
        self.timestamps[emotion] = now
        self.slopes[emotion] = delta

        self.emotional_trace.append(
            {
                "emotion": emotion,
                "value": val,
                "delta": round(delta, 3),
                "timestamp": now,
                "context": context,
            }
        )
        if len(self.emotional_trace) > 500:
            self.emotional_trace = self.emotional_trace[-500:]

        if abs(delta) > 0.3:
            self._trigger_reflex_adjustment(emotion, delta, context)

        self._persist()

    def _trigger_reflex_adjustment(
        self, emotion: str, delta: float, context: Optional[Dict[str, Any]] = None
    ):
        if self.context_engine and hasattr(self.context_engine, "adjust_from_emotion"):
            self.context_engine.adjust_from_emotion(emotion, delta, context)

    def apply_decay(self, decay: float = 0.01):
        for emotion in self.state_core:
            self.state_core[emotion] = max(0.0, self.state_core[emotion] - decay)
        self._persist()

    def harmonize(self, gradient: Dict[str, float]) -> Dict[str, float]:
        avg = sum(self.state_core.values()) / max(1, len(self.state_core))
        return {
            "zone_limbic": round(avg, 3),
            "zone_salience": gradient.get("salience_trigger", 0.0),
        }

    def summarize(self, data: dict = {}) -> Dict[str, Any]:
        return {
            "tags": list(self.tags),
            "emotional_intensity": dict(self.state_core),
            "recent_slopes": dict(self.slopes),
            "coherence": self._estimate_coherence(),
            "trace_tail": self.emotional_trace[-3:],
            "entry_count": len(self.history),
        }

    def _estimate_coherence(self) -> float:
        return round(len(self.tags) / (len(self.history) + 1), 3)

    def get_recent_history(self, count: int = 5) -> List[Dict[str, Any]]:
        return self.history[-count:]

    def _persist(self):
        self.state["tags"] = list(self.tags)
        self.state["emotional_trace"] = self.emotional_trace
        self.state["history"] = self.history
        self.state["state_core"] = self.state_core
        self.state["peaks"] = self.peaks
        self.state["timestamps"] = self.timestamps
        self.state["slopes"] = self.slopes

    def reset(self):
        self.tags.clear()
        self.emotional_trace.clear()
        self.history.clear()
        self.state_core.clear()
        self.peaks.clear()
        self.timestamps.clear()
        self.slopes.clear()
        self._persist()

    def __repr__(self):
        if not self.state_core:
            return "<EmotionCore ∅>"
        max_e = max(self.state_core, key=lambda e: self.state_core[e])
        return f"<EmotionCore dominant={max_e} level={self.state_core[max_e]:.2f}>"


from utils.module_base import ModuleBase
from typing import Dict, Any


class EmotionIntentMapper(ModuleBase):
    """
    MAX++ Module: Maps emotional states to interpreted beliefs/goals and evaluates alignment or conflict.
    Detects whether emotion supports, contradicts, or suppresses intent.
    """

    def __init__(self):
        super().__init__(module_name="EmotionIntentMapper")
        self.emotion_map = {
            "fear": {
                "conflicts": ["attain", "initiate", "risk"],
                "reinforces": ["avoid", "prevent"],
            },
            "hope": {"reinforces": ["achieve", "attain", "start"], "conflicts": ["cease", "halt"]},
            "trust": {"reinforces": ["support", "allow", "enable"]},
            "anger": {"conflicts": ["tolerate", "permit"], "reinforces": ["block", "demand"]},
            "sadness": {"conflicts": ["attain", "achieve"], "reinforces": ["withdraw", "cease"]},
            "joy": {"reinforces": ["start", "attain", "sustain"]},
            "disgust": {"reinforces": ["remove", "stop"], "conflicts": ["tolerate"]},
        }

    def evaluate(self, emotion: str, logic_stmt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes if the emotional state supports or contradicts the logic statement.
        """
        emotion = str(emotion or "").lower()
        then_raw = str(logic_stmt.get("THEN", "") or "")
        parts = then_raw.split()
        action = parts[0].lower() if parts else ""
        negation = logic_stmt.get("NEGATION", False)

        relation = "neutral"
        confidence = 0.5

        if emotion in self.emotion_map:
            data = self.emotion_map[emotion]
            if action in data.get("reinforces", []):
                relation = "reinforces"
                confidence = 0.9 if not negation else 0.6
            elif action in data.get("conflicts", []):
                relation = "conflicts"
                confidence = 0.9 if not negation else 0.6

        return {
            "emotion": emotion,
            "logic": logic_stmt,
            "relation": relation,
            "confidence": confidence,
        }


from utils.module_base import ModuleBase
from typing import Dict, List, Set, Tuple, Any, Optional


class FuzzyAlignmentScorer(ModuleBase):
    """
    Tier MAX++ Module: Infers multi-hop logical chains between concepts.
    Enables indirect alignment or contradiction reasoning (e.g., A→B, B→C → A→C).
    """

    def __init__(self, context: Optional[Any] = None):
        super().__init__(module_name=self.__class__.__name__)
        self.context = context
        self.graph: Dict[str, Set[str]] = {
            "shutdown": {"reduce emissions"},
            "reduce emissions": {"sustainability", "improve air quality"},
            "reduce co2": {"sustainability"},
            "factory closure": {"shutdown"},
            "reduce energy": {"reduce emissions"},
            "stop production": {"shutdown"},
            "cease": {"shutdown"},
            "halt": {"shutdown"},
            "attain": {"achieve"},
            "achieve": {"goal"},
            "allow": {"enable"},
            "enable": {"proceed"},
        }

    def infer_chain(self, start: str, end: str, max_depth: int = 4) -> Tuple[bool, List[str]]:
        """
        Returns True if a conceptual path exists from start to end, with the path.
        """
        visited = set()
        path = []

        def dfs(current, depth):
            if current == end:
                path.append(current)
                return True
            if depth == 0 or current in visited:
                return False
            visited.add(current)
            for neighbor in self.graph.get(current, []):
                if dfs(neighbor, depth - 1):
                    path.append(current)
                    return True
            return False

        found = dfs(start, max_depth)
        return found, list(reversed(path)) if found else []

    def get_summary(self, term: str) -> Set[str]:
        """
        Returns all reachable nodes from the term within 3 hops.
        (Formerly expand_concept)
        """
        result = set()
        frontier = [term]
        visited = set()

        for _ in range(3):
            next_frontier = []
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                neighbors = self.graph.get(node, [])
                next_frontier.extend(neighbors)
                result.update(neighbors)
            frontier = next_frontier

        return result

    def score(self, logic_a: List[Dict[str, Any]], logic_b: List[Dict[str, Any]]) -> float:
        """Alignment in [0, 1] from pairwise action overlap and fuzzy chain reachability."""
        if not logic_a and not logic_b:
            return 1.0
        if not logic_a or not logic_b:
            return 0.0

        def _first_action(stmt: Dict[str, Any]) -> str:
            ta = (stmt.get("THEN") or "").split()
            return ta[0].lower() if ta else ""

        hits = 0.0
        total = 0
        for a in logic_a:
            act_a = _first_action(a)
            for b in logic_b:
                total += 1
                act_b = _first_action(b)
                if not act_a or not act_b:
                    continue
                if act_a == act_b:
                    hits += 1.0
                elif self.infer_chain(act_a, act_b, 3)[0] or self.infer_chain(act_b, act_a, 3)[0]:
                    hits += 0.5

        return min(1.0, hits / total if total else 0.0)
