from utils.module_base import ModuleBase
from typing import Dict, Tuple, Optional, Any


class OntologyCore(ModuleBase):
    """
    MAX Engine: Resolves conceptual relationships between terms.
    Detects synonymy, implication, opposition, or semantic similarity for deep symbolic reasoning.
    """

    def __init__(self, context: Optional[Any] = None):
        super().__init__("OntologyCore")
        self.context = context

        self.ontology: Dict[str, Dict[str, list]] = {
            "sustainability": {
                "related": ["climate", "co2", "emissions", "environment", "ecology"]
            },
            "shutdown": {"synonyms": ["cease", "terminate", "halt", "close"]},
            "startup": {"synonyms": ["initiate", "launch", "begin"]},
            "reduce": {"related": ["lower", "decrease", "cut", "minimize"]},
            "emissions": {"related": ["pollution", "co2", "waste", "climate impact"]},
            "attain": {"synonyms": ["achieve", "reach", "complete"]},
            "prevent": {
                "synonyms": ["block", "deny", "inhibit"],
                "opposes": ["allow", "permit", "enable"],
            },
            "allow": {"opposes": ["prevent", "forbid", "deny"]},
        }

    def are_related(self, term1: str, term2: str) -> Tuple[bool, str]:
        """
        Determines whether term1 and term2 are related, synonyms, or opposites.
        Returns (True/False, explanation).
        """
        term1, term2 = term1.lower(), term2.lower()

        for concept, relations in self.ontology.items():
            synonyms = relations.get("synonyms", [])
            related = relations.get("related", [])
            opposes = relations.get("opposes", [])

            all_concept_terms = [concept] + synonyms + related

            if term1 in all_concept_terms:
                if term2 in all_concept_terms:
                    return True, f"Related via concept '{concept}'"
                if term2 in opposes:
                    return False, f"Contradiction: '{term1}' opposes '{term2}' under '{concept}'"

            if term2 in all_concept_terms:
                if term1 in opposes:
                    return False, f"Contradiction: '{term2}' opposes '{term1}' under '{concept}'"

        # Fallback: weak fuzzy match
        if term1 in term2 or term2 in term1:
            return True, "Soft lexical match (fuzzy partial)"

        return False, "No known semantic link"

    def get_link_type(self, term1: str, term2: str) -> str:
        related, reason = self.are_related(term1, term2)
        if "oppose" in reason.lower() or "contradiction" in reason.lower():
            return "opposes"
        if "synonym" in reason.lower():
            return "synonym"
        if "related" in reason.lower() or related:
            return "related"
        return "unrelated"


from utils.module_base import ModuleBase
from typing import List, Tuple, Set, Optional


class ConceptChainEngine(ModuleBase):
    """
    MAX++ Engine: Learns and evolves symbolic relations—synonyms, related terms, opposites—across context-aware chains.
    Enables inferencing depth-limited semantic transitions and contradiction detection via learned graph.
    """

    def __init__(self):
        super().__init__("ConceptChainEngine")
        self.synonyms: Dict[str, Set[str]] = {}
        self.related: Dict[str, Set[str]] = {}
        self.opposites: Dict[str, Set[str]] = {}

    def suggest_relation(self, term1: str, term2: str, relation: str) -> str:
        t1, t2 = term1.lower(), term2.lower()
        if relation == "synonym":
            self.synonyms.setdefault(t1, set()).add(t2)
            self.synonyms.setdefault(t2, set()).add(t1)
        elif relation == "related":
            self.related.setdefault(t1, set()).add(t2)
            self.related.setdefault(t2, set()).add(t1)
        elif relation == "opposite":
            self.opposites.setdefault(t1, set()).add(t2)
            self.opposites.setdefault(t2, set()).add(t1)
        else:
            return "Invalid relation type"
        return f"Learned {relation} between '{t1}' and '{t2}'"

    def retract_relation(self, term1: str, term2: str, relation: str) -> str:
        t1, t2 = term1.lower(), term2.lower()
        mapping = {"synonym": self.synonyms, "related": self.related, "opposite": self.opposites}
        if relation in mapping:
            mapping[relation].get(t1, set()).discard(t2)
            mapping[relation].get(t2, set()).discard(t1)
            return f"Removed {relation} between '{t1}' and '{t2}'"
        return "Invalid relation type"

    def get_relations(self, term: str) -> Dict[str, Set[str]]:
        t = term.lower()
        return {
            "synonyms": self.synonyms.get(t, set()),
            "related": self.related.get(t, set()),
            "opposites": self.opposites.get(t, set()),
        }

    def infer_chain(self, start: str, target: str, max_depth: int = 3) -> Tuple[bool, List[str]]:
        """
        Attempts to construct a semantic path from start to target using learned relations.
        Returns (True, path) if successful.
        """
        from collections import deque

        start, target = start.lower(), target.lower()
        visited = set()
        queue = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()
            if current == target:
                return True, path
            if len(path) >= max_depth:
                continue

            neighbors = (
                self.synonyms.get(current, set())
                | self.related.get(current, set())
                | self.opposites.get(current, set())
            )
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return False, []

    def detect_contradiction(self, term1: str, term2: str) -> bool:
        """
        Checks if term1 and term2 are known opposites either directly or via paths.
        """
        t1, t2 = term1.lower(), term2.lower()
        return t2 in self.opposites.get(t1, set()) or t1 in self.opposites.get(t2, set())


from utils.module_base import ModuleBase
from typing import Dict, Set, Tuple


class OntologyExpander(ModuleBase):
    """
    MAX++ Module: Expands internal ontology based on usage patterns and user feedback.
    Learns new synonyms, related terms, and oppositional pairs over time.
    """

    def __init__(self):
        super().__init__(module_name="OntologyExpander")
        self.synonyms: Dict[str, Set[str]] = {}
        self.related: Dict[str, Set[str]] = {}
        self.opposites: Dict[str, Set[str]] = {}

    def suggest_relation(self, term1: str, term2: str, relation: str) -> str:
        """
        Suggests a new relation to be learned.
        """
        t1, t2 = term1.lower(), term2.lower()

        if relation == "synonym":
            self.synonyms.setdefault(t1, set()).add(t2)
            self.synonyms.setdefault(t2, set()).add(t1)
        elif relation == "related":
            self.related.setdefault(t1, set()).add(t2)
            self.related.setdefault(t2, set()).add(t1)
        elif relation == "opposite":
            self.opposites.setdefault(t1, set()).add(t2)
            self.opposites.setdefault(t2, set()).add(t1)
        else:
            return "Invalid relation type"

        return f"Learned {relation} between '{t1}' and '{t2}'"

    def get_relations(self, term: str) -> Dict[str, Set[str]]:
        """
        Returns all known relations for a term.
        """
        term = term.lower()
        return {
            "synonyms": self.synonyms.get(term, set()),
            "related": self.related.get(term, set()),
            "opposites": self.opposites.get(term, set()),
        }

    def retract_relation(self, term1: str, term2: str, relation: str) -> str:
        """
        Removes a learned relation.
        """
        t1, t2 = term1.lower(), term2.lower()
        target = None

        if relation == "synonym":
            target = self.synonyms
        elif relation == "related":
            target = self.related
        elif relation == "opposite":
            target = self.opposites

        if target:
            target.get(t1, set()).discard(t2)
            target.get(t2, set()).discard(t1)
            return f"Removed {relation} between '{t1}' and '{t2}'"
        return "Invalid relation type"


from utils.module_base import ModuleBase
from typing import List, Dict, Any, Optional


class OntologyBridgeEngine(ModuleBase):
    """
    Tier MAX+++ Module: Integrates symbolic logic with memory forest overlays,
    emotion-weighted context, identity resolution, and contradiction classification.
    """

    def __init__(self, memory_engine: Optional[Any] = None):
        super().__init__("OntologyBridgeEngine")
        from core.SymbolicConflict import ContextInterpreter, ContradictionTracer
        from core.AffectiveReasoning import EmotionIntentMapper, FuzzyAlignmentScorer

        self.interpreter = ContextInterpreter()
        self.memory_engine = memory_engine
        self.forest = getattr(memory_engine, "forest", None) if memory_engine else None
        self.emotion_mapper = EmotionIntentMapper()
        self.scorer = FuzzyAlignmentScorer()
        self.tracer = ContradictionTracer()
        self.chains = ConceptChainEngine()

    def analyze(self, logic_set: List[Dict[str, Any]], emotion: str) -> Dict[str, Any]:
        """
        Augments symbolic logic with emotion and identity overlays.
        """
        overlays = []
        for logic in logic_set:
            identity_tag = self._resolve_identity(logic)
            emotion_result = self.emotion_mapper.evaluate(emotion, logic)
            overlays.append(
                {"logic": logic, "identity": identity_tag, "emotion_analysis": emotion_result}
            )
        return {"enriched": overlays, "emotion_used": emotion}

    def _resolve_identity(self, logic: Dict[str, Any]) -> str:
        if not self.forest:
            return "unknown"
        identity_tree = self.forest.get_tree("identity")
        subj = logic.get("subject", "unknown")
        if identity_tree and subj in identity_tree:
            return f"matched:{subj}"
        # Fallback: check for symbolic proximity
        for identity, profile in identity_tree.items():
            tags = profile.get("tags", [])
            if any(tag in logic.get("THEN", "") for tag in tags):
                return f"inferred:{identity}"
        return "unmatched"

    def compare(
        self, logic_a: List[Dict[str, Any]], logic_b: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Deep symbolic comparison across logic sets with enriched trace.
        """
        alignment_score = self.scorer.score(logic_a, logic_b)
        trace = []
        contradictions = []

        for a in logic_a:
            for b in logic_b:
                detail = self.tracer.trace(a, b)
                enriched_type = self._classify_conflict_type(a, b, detail)
                if detail.get("contradiction"):
                    detail["conflict_type"] = enriched_type
                    contradictions.append(detail)
                trace.append(detail)

        return {
            "alignment_score": alignment_score,
            "trace": trace,
            "contradictions_found": contradictions,
            "summary": self._summarize(alignment_score, contradictions),
        }

    def _classify_conflict_type(
        self, stmt_a: Dict[str, Any], stmt_b: Dict[str, Any], trace: Dict[str, Any]
    ) -> str:
        if stmt_a["TYPE"] == "goal_intent" and stmt_b["TYPE"] == "goal_intent":
            return "goal_conflict"
        if stmt_a["TYPE"] == "constraint" or stmt_b["TYPE"] == "constraint":
            return "ethical_dissonance"
        if stmt_a.get("TIME") and stmt_b.get("TIME") and stmt_a["TIME"] != stmt_b["TIME"]:
            return "temporal_disjunction"
        return "general_contradiction"

    def _summarize(self, score: float, contradictions: List[Dict[str, Any]]) -> str:
        parts = [f"Alignment Score: {score:.2f}."]
        if contradictions:
            c_types = [c.get("conflict_type", "unknown") for c in contradictions]
            c_summary = {ct: c_types.count(ct) for ct in set(c_types)}
            parts.append(
                f"{len(contradictions)} contradiction(s) found: "
                + ", ".join(f"{k} ({v})" for k, v in c_summary.items())
            )
        else:
            parts.append("No contradictions detected.")
        return " ".join(parts)


from utils.module_base import ModuleBase
from typing import Dict, Any, Optional, List, Set


# === Tier 74 Core Module: DoctrineSynthesizer ===


class DoctrineSynthesizer(ModuleBase):
    """
    Tier 74 Module: Synthesizes symbolic doctrine based on identity roles,
    emotional history, narrative themes, and goal tags. Produces a guiding doctrine set.
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(module_name="DoctrineSynthesizer")
        self.context = context

    def synthesize(
        self,
        identity_roles: List[str],
        emotion_history: List[str],
        goal_tags: List[str],
        narrative_themes: List[str],
    ) -> List[str]:

        doctrine: Set[str] = set()

        # Rule-based symbolic doctrine synthesis
        if "Rebel" in identity_roles and "anger" in emotion_history:
            doctrine.add("challenge-authority")

        if "Seeker" in identity_roles and "joy" in emotion_history:
            doctrine.add("pursue-discovery")

        if "Exile" in identity_roles and "sadness" in emotion_history:
            doctrine.add("self-preservation")

        if "resistance" in narrative_themes:
            doctrine.add("nonconformity")

        if "gratitude" in emotion_history and "Guardian" in identity_roles:
            doctrine.add("protect-others")

        if any(tag in goal_tags for tag in {"truth", "clarity", "awareness"}):
            doctrine.add("value-truth")

        doctrine_list = sorted(doctrine)

        # Log symbolic synthesis
        self.signal_history.append(
            {
                "event": "doctrine_synthesized",
                "cycle_ts": (self.context or {}).get("cycle_ts", "unknown"),
                "output": doctrine_list,
                "inputs": {
                    "identity_roles": identity_roles,
                    "emotion_history": emotion_history,
                    "goal_tags": goal_tags,
                    "narrative_themes": narrative_themes,
                },
            }
        )

        return doctrine_list


from utils.module_base import ModuleBase
from typing import Dict, Any, Optional, List, Set, Tuple


# === Tier 74 Core Module: DoctrineConflictDetector ===


class DoctrineConflictDetector(ModuleBase):
    """
    Tier 74 Module: Detects internal symbolic contradictions within doctrine tags
    using a preconfigured contradiction map.
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(module_name="DoctrineConflictDetector")
        self.context = context

    def detect(self, doctrine_tags: List[str]) -> List[Tuple[str, str]]:
        conflicts: Set[Tuple[str, str]] = set()

        # Contradiction rules: tag → set of contradicting tags
        contradiction_map: Dict[str, Set[str]] = {
            "challenge-authority": {"protect-structure", "follow-leadership"},
            "self-preservation": {"sacrifice-for-group"},
            "nonconformity": {"value-tradition"},
            "pursue-discovery": {"prioritize-stability"},
            "value-truth": {"prioritize-loyalty"},
        }

        for tag in doctrine_tags:
            contradicts = contradiction_map.get(tag, set())
            for other in doctrine_tags:
                if other in contradicts:
                    # Ensure consistent tuple ordering
                    conflicts.add(tuple(sorted((tag, other))))

        sorted_conflicts = sorted(conflicts)

        # Log contradiction detection
        self.signal_history.append(
            {
                "event": "doctrine_conflict_detected",
                "cycle_ts": (self.context or {}).get("cycle_ts", "unknown"),
                "doctrine_tags": doctrine_tags,
                "conflicts": sorted_conflicts,
            }
        )

        return sorted_conflicts


from utils.module_base import ModuleBase
from typing import Dict, Any, Optional, List, Set


# === Tier 74 Core Module: DoctrineStabilityEvaluator ===


class DoctrineStabilityEvaluator(ModuleBase):
    """
    Tier 74 Module: Evaluates symbolic doctrine consistency across cycles.
    Returns a float [0.0 - 1.0] indicating average overlap between successive doctrine states.
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(module_name="DoctrineStabilityEvaluator")
        self.context = context

    def evaluate(self, doctrine_history: List[List[str]]) -> float:
        if len(doctrine_history) < 2:
            return 1.0

        total_overlap = 0.0
        comparisons = 0

        for i in range(1, len(doctrine_history)):
            prev_set: Set[str] = set(doctrine_history[i - 1])
            curr_set: Set[str] = set(doctrine_history[i])

            if prev_set or curr_set:
                union_size = len(prev_set.union(curr_set))
                intersection_size = len(prev_set.intersection(curr_set))
                overlap = intersection_size / max(union_size, 1)
                total_overlap += overlap
                comparisons += 1

        stability_score = round(total_overlap / comparisons, 3) if comparisons > 0 else 1.0

        # Trace log
        self.signal_history.append(
            {
                "event": "doctrine_stability_evaluation",
                "cycle_ts": (self.context or {}).get("cycle_ts", "unknown"),
                "doctrine_history_length": len(doctrine_history),
                "stability_score": stability_score,
            }
        )

        return stability_score


from utils.module_base import ModuleBase
from typing import Dict, Any, Optional, List


# === Tier 98 Core Modules ===


class ValueDelegationMapper(ModuleBase):
    def __init__(self, context: Optional[Any] = None):
        super().__init__(module_name="ValueDelegationMapper")
        self.context = context

    def map(
        self, doctrine: List[str], protected_tags: List[str] = ["identity", "truth", "freedom"]
    ) -> Dict[str, List[str]]:
        """
        Separates doctrine into core (non-delegable) values and delegable ones.
        """
        core = [v for v in doctrine if any(p in v for p in protected_tags)]
        delegable = [v for v in doctrine if v not in core]
        return {"core_values": sorted(set(core)), "delegable_values": sorted(set(delegable))}


from utils.module_base import ModuleBase
from typing import Dict, List, Any, Optional
import time


class SchemaCore(ModuleBase):
    """
    Tier MAX++ Architect:
    Unifies schema building, law management, ethics exchange, and narrative weaving.
    """

    def __init__(self, memory_engine: Optional[Any] = None):
        super().__init__(module_name=self.__class__.__name__)
        self.memory_engine = memory_engine
        self.load_state()

        self.schemas: Dict[str, List[str]] = self.state.get("schemas", {})
        self.laws: Dict[str, str] = self.state.get("laws", {})
        self.constitutions: Dict[str, Dict[str, Any]] = self.state.get("constitutions", {})
        self.ethics: Dict[str, float] = self.state.get("ethics", {})
        self.history: List[Dict[str, Any]] = self.state.get("history", [])[-500:]
        self.narratives: List[Dict[str, Any]] = self.state.get("narratives", [])[-500:]

    def assign_schema(self, key: str, schema: str):
        self.schemas.setdefault(schema, []).append(key)
        self.state["schemas"] = self.schemas

    def declare_law(self, name: str, content: str):
        self.laws[name] = content
        self.history.append(
            {"type": "law", "name": name, "content": content, "timestamp": time.time()}
        )
        if len(self.history) > 500:
            self.history = self.history[-500:]
        self.state["laws"] = self.laws
        self.state["history"] = self.history

    def revoke_mandate(self, key: str):
        if key in self.laws:
            del self.laws[key]
            self.history.append({"type": "revoke", "key": key, "timestamp": time.time()})
            if len(self.history) > 500:
                self.history = self.history[-500:]
            self.state["laws"] = self.laws
            self.state["history"] = self.history

    def declare_mandate(self, key: str, content: Any):
        self.laws[key] = content
        self.history.append(
            {"type": "mandate", "key": key, "content": content, "timestamp": time.time()}
        )
        if len(self.history) > 500:
            self.history = self.history[-500:]
        self.state["laws"] = self.laws
        self.state["history"] = self.history

    def update_ethic(self, principle: str, weight: float):
        self.ethics[principle] = round(min(1.0, max(0.0, weight)), 3)
        self.state["ethics"] = self.ethics

    def ratify_constitution(self, group: str, charter: Dict[str, Any]):
        self.constitutions[group] = charter
        self.state["constitutions"] = self.constitutions

    def log_narrative(self, label: str, tags: List[str], arc: Optional[str] = None):
        entry = {"label": label, "tags": tags, "arc": arc, "timestamp": time.time()}
        self.narratives.append(entry)
        if len(self.narratives) > 500:
            self.narratives = self.narratives[-500:]
        self.state["narratives"] = self.narratives

    def get_summary(self) -> Dict[str, Any]:
        return {
            "schemas": list(self.schemas.keys()),
            "laws": list(self.laws.keys()),
            "ethics": self.ethics,
            "constitutions": list(self.constitutions.keys()),
            "narratives": len(self.narratives),
            "last_narrative": self.narratives[-1] if self.narratives else None,
        }

    def reset(self):
        self.schemas.clear()
        self.laws.clear()
        self.constitutions.clear()
        self.ethics.clear()
        self.history.clear()
        self.narratives.clear()
        self.state.clear()

    def __repr__(self):
        return f"<SchemaArchitect laws={len(self.laws)} ethics={len(self.ethics)} narratives={len(self.narratives)}>"
