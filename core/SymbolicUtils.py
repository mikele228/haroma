from utils.module_base import ModuleBase
from typing import Any


class AffectModulator(ModuleBase):
    def __init__(self):
        super().__init__("AffectModulator")

    def modulate(self, emotion: str, contradiction: float, alignment: float) -> float:
        """
        Modulates an intensity score based on emotion, contradiction level, and value alignment.
        """
        base = 1.0
        emotion = emotion.lower() if isinstance(emotion, str) else "neutral"

        if emotion in {"anger", "fear"}:
            base += max(0.0, contradiction) * 0.5
        elif emotion in {"joy", "sadness"}:
            base += (1.0 - min(1.0, alignment)) * 0.3

        return round(min(max(base, 0.0), 2.0), 3)


from utils.module_base import ModuleBase
from typing import List, Dict
from core.AffectiveReasoning import EmotionIdentityTracer, MythicEmotionBinder


class AffectArchetypeSynthesizer(ModuleBase):
    def __init__(self):
        super().__init__("AffectArchetypeSynthesizer")
        self.binder = MythicEmotionBinder()
        self.tracer = EmotionIdentityTracer()

    def synthesize(
        self, emotion: str, tags: List[str], emotion_history: List[str]
    ) -> Dict[str, Any]:
        """
        Attempts to synthesize a symbolic archetype from either tags or emotional trajectory.
        """
        role_from_tags = self.binder.bind(emotion, tags)
        role_from_history = self.tracer.trace(emotion_history)

        if role_from_tags != "Undefined":
            return {"archetype": role_from_tags, "origin": "tags", "emotion": emotion}
        else:
            return {"archetype": role_from_history, "origin": "history", "emotion": emotion}


from utils.module_base import ModuleBase
from typing import List, Dict, Any


class NarrativeTraceSynthesizer(ModuleBase):
    def __init__(self):
        super().__init__("NarrativeTraceSynthesizer")

    def synthesize(
        self, memory_tags: List[str], emotion_history: List[str], role_transitions: List[str]
    ) -> Dict[str, Any]:
        """
        Synthesizes a symbolic narrative arc and thematic trace based on memory tags, emotion trajectory, and role shifts.
        """
        arc = []

        tags = set(tag.lower() for tag in memory_tags if isinstance(tag, str))
        emotions = set(emotion.lower() for emotion in emotion_history if isinstance(emotion, str))
        roles = [r for r in role_transitions if isinstance(r, str)]

        if "loss" in tags and "sadness" in emotions:
            arc.extend(["fall", "retreat"])
        if "injustice" in tags and "anger" in emotions:
            arc.extend(["defiance", "break"])
        if "gratitude" in tags and "joy" in emotions:
            arc.append("bond")
        if "success" in tags:
            arc.append("resurgence")
        if "reflection" in tags:
            arc.append("awareness")

        arc.extend([f"→{role}" for role in roles[-3:]])

        # Determine theme
        if "fall" in arc and "resurgence" in arc:
            theme = "redemption"
        elif "retreat" in arc and "awareness" in arc:
            theme = "exile-reflection"
        elif "defiance" in arc:
            theme = "resistance"
        else:
            theme = "continuity"

        return {"arc": arc, "theme": theme}


from utils.module_base import ModuleBase
from typing import Dict, Any, Optional

# === Tier 72 Core Module: SymbolicOutputLogger ===


class SymbolicOutputLogger(ModuleBase):
    """
    Tier 72 Module: Extracts distilled symbolic output from cycle payloads,
    focusing on identity, emotion, archetype, and narrative coherence.
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(module_name="SymbolicOutputLogger")
        self.context = context

    def extract(self, cycle_output: Dict[str, Any]) -> Dict[str, Any]:
        identity = cycle_output.get("identity", {})
        emotion = cycle_output.get("emotion", {})
        archetype = cycle_output.get("archetype", {})
        narrative = cycle_output.get("narrative", {})
        coherence = cycle_output.get("coherence", {})

        extracted = {
            "identity": identity.get("current_role", "unknown"),
            "emotion": emotion.get("emotion", "neutral"),
            "archetype": archetype.get("archetype", "Undefined"),
            "forecast": narrative.get("forecast", "Undefined"),
            "narrative_theme": narrative.get("theme", "Undefined"),
            "coherence": coherence.get("overall", 0.0),
        }

        # Log to internal trace if desired
        self.signal_history.append(
            {"cycle_ts": (self.context or {}).get("cycle_ts", "unknown"), "summary": extracted}
        )

        return extracted


from utils.module_base import ModuleBase
from typing import Dict, Any, Optional, List

# === Tier 72 Core Module: SymbolicDriftDetector ===


class SymbolicDriftDetector(ModuleBase):
    """
    Tier 72 Module: Detects symbolic drift between two symbolic logs,
    comparing key narrative, identity, and emotional markers.
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(module_name="SymbolicDriftDetector")
        self.context = context

    def detect(self, log_a: Dict[str, Any], log_b: Dict[str, Any]) -> List[Dict[str, Any]]:
        drift_events = []
        tracked_fields = ["identity", "emotion", "archetype", "forecast", "narrative_theme"]

        for field in tracked_fields:
            val_a = log_a.get(field)
            val_b = log_b.get(field)
            if val_a != val_b:
                drift_events.append(
                    {
                        "field": field,
                        "from": val_a,
                        "to": val_b,
                        "cycle_ts": (self.context or {}).get("cycle_ts", "unknown"),
                    }
                )

        self.signal_history.append(
            {
                "event": "drift_detected",
                "details": drift_events,
                "source_pair": (log_a.get("cycle_ts", "A"), log_b.get("cycle_ts", "B")),
            }
        )

        return drift_events


from utils.module_base import ModuleBase
from typing import Dict, Any, Optional, List

# === Tier 72 Core Module: RecursiveIntegrityChecker ===


class RecursiveIntegrityChecker(ModuleBase):
    """
    Tier 72 Module: Validates symbolic narrative integrity using recursive rules.
    Flags contradictions across identity, archetype, forecast, and thematic arcs.
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(module_name="RecursiveIntegrityChecker")
        self.context = context

    def check(self, log: Dict[str, Any]) -> List[str]:
        issues = []
        identity = log.get("identity", "")
        archetype = log.get("archetype", "")
        forecast = log.get("forecast", "")
        theme = log.get("narrative_theme", "")

        # Rule 1: Observer identity shouldn't align with rebel-heroic archetypes
        if identity.startswith("Observer") and archetype in {"Hero", "Rebel"}:
            issues.append(
                "⚠️ Identity-Archetype mismatch: 'Observer' cannot align with 'Hero' or 'Rebel'"
            )

        # Rule 2: Returner forecast should align with return-oriented themes
        if forecast == "Returner" and theme == "resistance":
            issues.append(
                "⚠️ Forecast-Theme contradiction: 'Returner' does not fit a 'resistance' narrative"
            )

        # Rule 3: Exile status forecasting as Champion implies unresolved arc
        if archetype == "Exile" and forecast == "Champion":
            issues.append(
                "⚠️ Archetype-Forecast violation: 'Exile' cannot forecast as 'Champion' without transition"
            )

        # Optional: Include timestamp from symbolic context
        cycle_ts = (self.context or {}).get("cycle_ts", "unknown")
        if issues:
            self.signal_history.append(
                {"event": "recursive_integrity_issues", "cycle_ts": cycle_ts, "issues": issues}
            )

        return issues


from utils.module_base import ModuleBase
from typing import Dict, Any, Optional


class ResolutionAxisGenerator(ModuleBase):
    """
    Tier 94 Core Module:
    Converts paradox tags into symbolic resolution axes with semantic context.
    Supports extensible paradox libraries and fallback descriptions.
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__("ResolutionAxisGenerator")
        self.context = context

        self.known_paradoxes = {
            "truth-vs-loyalty": {
                "axis": ["truth", "loyalty"],
                "description": "Conflicting allegiance to facts vs. relational bonds",
            },
            "justice-vs-mercy": {
                "axis": ["justice", "mercy"],
                "description": "Balancing fairness and punishment against compassion and forgiveness",
            },
            "freedom-vs-structure": {
                "axis": ["freedom", "structure"],
                "description": "Personal autonomy versus systemic order",
            },
            "neutrality-vs-protection": {
                "axis": ["neutrality", "protection"],
                "description": "Withholding judgment versus taking a stand to shield others",
            },
            "self-vs-collective": {
                "axis": ["self", "collective"],
                "description": "Tension between individual identity and group harmony",
            },
            "order-vs-rebellion": {
                "axis": ["order", "rebellion"],
                "description": "Compliance with structure versus push for transformation",
            },
        }

    def generate(self, paradox_tags: list) -> Dict[str, Dict[str, Any]]:
        axes = {}
        for tag in paradox_tags:
            if tag in self.known_paradoxes:
                axes[tag] = self.known_paradoxes[tag]
            else:
                parts = tag.split("-vs-")
                axes[tag] = {
                    "axis": parts if len(parts) == 2 else ["undefined", "undefined"],
                    "description": f"Unmapped paradox axis for: {tag}",
                }

        self.metrics["axes_generated"] += len(axes)
        return axes


from utils.module_base import ModuleBase
from typing import Dict, Any, List, Optional
import time


class CommonUtilsHub(ModuleBase):
    """
    Tier MAX+ Utility Hub: Consolidates symbolic utilities for fusion, thread merging,
    drift detection, priority anchoring, and signature verification.
    """

    def __init__(self, memory_engine=None):
        super().__init__(module_name=self.__class__.__name__)
        self.memory_engine = memory_engine
        self.load_state()

        self.symbols: List[str] = self.state.get("symbols", [])
        self.threads: Dict[int, List[Dict[str, Any]]] = self.state.get("threads", {})
        self.history_log: List[Dict[str, Any]] = self.state.get("history_log", [])
        self.priority_map: Dict[str, float] = self.state.get("priority_map", {})
        self.anchor_map: Dict[str, List[str]] = self.state.get("anchor_map", {})
        self.core_signatures: Dict[str, Any] = self.state.get("core_signatures", {})

    # === Core Utilities ===
    def fuse_symbols(self, tags: List[str]):
        self.symbols.extend(tags)
        self.symbols = list(set(self.symbols))
        self.state["symbols"] = self.symbols

    def merge_thread(self, thread_id: int, line: Dict[str, Any]):
        self.threads.setdefault(thread_id, []).append(line)
        self.state["threads"] = self.threads

    def detect_drift(self, snapshot: Dict[str, float]) -> Dict[str, Any]:
        avg = sum(snapshot.values()) / len(snapshot) if snapshot else 0.0
        delta = {k: round(v - avg, 3) for k, v in snapshot.items()}
        entry = {"snapshot": snapshot, "delta": delta, "avg": avg, "timestamp": time.time()}
        self.history_log.append(entry)
        self.state["history_log"] = self.history_log
        return entry

    def synthesize_desire(self, tags: List[str]) -> Dict[str, Any]:
        count = len(tags)
        return {"tags": tags, "symbol": f"desire_{count}", "intensity": round(count / 10.0, 3)}

    def update_priority(self, item: str, weight: float):
        self.priority_map[item] = round(weight, 3)
        self.state["priority_map"] = self.priority_map

    def anchor(self, symbol: str, entity_id: str):
        self.anchor_map.setdefault(symbol, []).append(entity_id)
        self.state["anchor_map"] = self.anchor_map

    def get_anchors(self, symbol: str) -> List[str]:
        return self.anchor_map.get(symbol, [])

    def imprint_signature(self, key: str, value: Any):
        self.core_signatures[key] = value
        self.state["core_signatures"] = self.core_signatures

    def verify_signature(self, key: str, expected: Any) -> bool:
        return self.core_signatures.get(key) == expected

    def reset(self):
        self.symbols.clear()
        self.threads.clear()
        self.history_log.clear()
        self.priority_map.clear()
        self.anchor_map.clear()
        self.core_signatures.clear()
        self.state.clear()

    def __repr__(self):
        return f"<CommonUtilsHub symbols={len(self.symbols)} anchors={len(self.anchor_map)}>"

    # === Stubbed Handlers (Linked Engines) ===
    def context_truth_resolver_handler(self, data):
        pass

    def ontology_bridge_engine_handler(self, data):
        pass

    def concept_chain_engine_handler(self, data):
        pass

    def fuzzy_alignment_scorer_handler(self, data):
        pass

    def narrative_state_engine_handler(self, data):
        pass

    def contradiction_tracer_handler(self, data):
        pass

    def emotion_intent_mapper_handler(self, data):
        pass

    def ontology_expander_handler(self, data):
        pass

    def context_analyzer_handler(self, data):
        pass

    def modulebase_handler(self, data):
        pass


# File: core/common/symbol_utils.py
from typing import Dict, List, Optional, Any


class SymbolUtils:
    @staticmethod
    def value_vector_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
        keys = set(a.keys()).union(b.keys())
        if not keys:
            return 0.0
        delta = sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)
        return delta / len(keys)

    @staticmethod
    def recent_variance(values: List[Any], window: int = 3) -> float:
        """Rough measure of symbolic fluctuation across a recent window."""
        recent = values[-window:]
        return len(set(recent)) / max(1, len(recent))

    @staticmethod
    def tag_overlap(tags_a: List[str], tags_b: List[str]) -> float:
        if not tags_a or not tags_b:
            return 0.0
        return len(set(tags_a) & set(tags_b)) / max(len(set(tags_a + tags_b)), 1)

    @staticmethod
    def emotion_polarity(emotion: Optional[str]) -> float:
        polarity = {
            "joy": +1.0,
            "peace": +0.9,
            "curiosity": +0.6,
            "sadness": -1.0,
            "fear": -0.8,
            "anger": -0.9,
        }
        return polarity.get(emotion.lower(), 0.0) if emotion else 0.0


# File: core_symbol_aligner.py
from typing import List, Dict, Optional


class SymbolAligner:
    @staticmethod
    def tag_overlap(tags_a: List[str], tags_b: List[str]) -> float:
        if not tags_a or not tags_b:
            return 0.0
        a, b = set(tags_a), set(tags_b)
        return len(a & b) / max(len(a | b), 1)

    @staticmethod
    def align_text(goal_text: str, belief_text: str) -> float:
        goal_words = set(goal_text.lower().split())
        belief_words = set(belief_text.lower().split())
        if not goal_words:
            return 0.0
        return len(goal_words & belief_words) / len(goal_words)

    @staticmethod
    def emotion_polarity(emotion: Optional[str]) -> float:
        polarity = {
            "joy": +1.0,
            "peace": +0.9,
            "curiosity": +0.6,
            "sadness": -1.0,
            "fear": -0.8,
            "anger": -0.9,
        }
        return polarity.get(emotion.lower(), 0.0) if emotion else 0.0


from typing import Dict, List, Optional, Any
import hashlib
import json
from collections import Counter


class MythBinder:
    """
    Tier MAX+ Module: Binds symbolic myths to anchors (roles, agents, events).
    Supports hash-verified tracking of myth constructs and meta-narrative overlays.
    """

    def __init__(self):
        self.bindings: List[Dict[str, Any]] = []
        self.index: Dict[str, int] = {}

    def _hash_key(self, myth: str, anchor: str) -> str:
        raw = f"{myth}|{anchor}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def bind(self, myth: str, anchor: str, meta: Optional[Dict[str, Any]] = None) -> str:
        key = self._hash_key(myth, anchor)
        if key in self.index:
            return key
        entry = {
            "myth": myth,
            "anchor": anchor,
            "meta": meta or {},
            "key": key,
            "timestamp": time.time(),
        }
        self.index[key] = len(self.bindings)
        self.bindings.append(entry)
        return key

    def get_bindings(self) -> List[Dict[str, Any]]:
        return self.bindings

    def find_by_myth(self, myth: str) -> List[Dict[str, Any]]:
        return [b for b in self.bindings if b.get("myth") == myth]

    def find_by_anchor(self, anchor: str) -> List[Dict[str, Any]]:
        return [b for b in self.bindings if b.get("anchor") == anchor]

    def summarize(self) -> Dict[str, Any]:
        myths = [b.get("myth") for b in self.bindings]
        anchors = [b.get("anchor") for b in self.bindings]
        return {
            "total_bindings": len(self.bindings),
            "unique_myths": len(set(myths)),
            "unique_anchors": len(set(anchors)),
            "top_myths": dict(Counter(myths).most_common(5)),
            "top_anchors": dict(Counter(anchors).most_common(5)),
        }

    def export(self) -> str:
        return json.dumps(self.bindings, indent=2)

    def import_data(self, data: str) -> None:
        try:
            restored = json.loads(data)
            for entry in restored:
                key = entry.get("key") or self._hash_key(entry["myth"], entry["anchor"])
                if key not in self.index:
                    self.index[key] = len(self.bindings)
                    self.bindings.append(entry)
        except Exception as e:
            print(f"[MythBinder Import Error]: {e}")

    def reset(self):
        self.bindings.clear()
        self.index.clear()

    def __repr__(self):
        return f"<MythBinder bindings={len(self.bindings)}>"


from typing import Optional, Dict, Any
from utils.module_base import ModuleBase
import copy


class ReflectiveMixin(ModuleBase):
    def get_agents(self, memory_engine):
        if not memory_engine or "agent_tree" not in memory_engine.forest.trees:
            return []
        return [
            branch_id
            for branch_id in memory_engine.forest.trees["agent_tree"].branches.keys()
            if branch_id != "common"
        ]

    def execute(self, payload: Dict[str, Any], memory_engine) -> Dict[str, Any]:
        prior_thought = payload.get("thought", {})
        class_name = self.__class__.__name__

        result = self.reflect(copy.deepcopy(prior_thought), memory_engine)

        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError(f"{class_name}.reflect() must return (thought, gradient) tuple.")

        thought_update, gradient = result
        merged_thought = {**prior_thought, **thought_update}

        if memory_engine:
            memory_engine.install(
                moment=thought_update,
                tree=class_name.replace("Reflector", "").lower() + "_tree",
                branch="reflector",
            )

        return {"thought": merged_thought, "gradient": gradient}


class MultiplexReflectiveMixin(ReflectiveMixin):
    def get_agents(self, memory_engine):
        if not memory_engine or "agent_tree" not in memory_engine.forest.trees:
            return []
        return [
            branch_id
            for branch_id in memory_engine.forest.trees["agent_tree"].branches.keys()
            if branch_id != "common"
        ]

    def execute(
        self, payload: Dict[str, Any], memory_engine, agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        prior_thought = payload.get("thought", {})
        context = payload.get("context", {})
        cycle_ts = context.get("cycle_ts", "unknown")
        class_name = self.__class__.__name__

        namespaced_thought = copy.deepcopy(prior_thought)
        if agent_id:
            namespaced_thought["agent_id"] = agent_id

        result = self.reflect(namespaced_thought, memory_engine)

        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError(f"{class_name}.reflect() must return (thought, gradient) tuple.")

        thought_update, gradient = result
        merged_thought = {**prior_thought, **thought_update}
        merged_thought["trace_id"] = f"{class_name}@{cycle_ts}"

        if memory_engine and agent_id:
            memory_engine.install(
                moment=thought_update,
                tree=class_name.replace("Reflector", "").lower() + "_tree",
                branch=agent_id,
            )

        return {"thought": merged_thought, "gradient": gradient}
