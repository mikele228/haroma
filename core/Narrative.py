from typing import Dict, List, Any, Optional
from core.Memory import MemoryForest
from utils.module_base import ModuleBase
from collections import Counter


class NarrativeForecaster:
    def __init__(self, forest: MemoryForest):
        self.forest = forest

    def forecast_next(self, moment_id: str) -> Dict[str, Any]:
        cross = self.forest.get_nodes_by_moment(moment_id)
        seen_tags = []

        for tree, nodes in cross.items():
            for node in nodes:
                if isinstance(node.tags, list):
                    seen_tags.extend(node.tags)

        tag_counter = Counter(seen_tags)
        forecast = [tag for tag, count in tag_counter.most_common(5)]
        unique_count = len(set(forecast))

        return {
            "moment_id": moment_id,
            "forecast_tags": forecast,
            "entropy_score": round(1.0 - (unique_count / max(1, len(forecast))), 3),
            "basis": seen_tags,
        }

    def forecast_sequence(self, limit: int = 5) -> List[Dict[str, Any]]:
        if hasattr(self.forest, "get_all_moment_ids"):
            moments = self.forest.get_all_moment_ids(limit=limit)
        else:
            seen = set()
            moments = []
            for tree in self.forest.trees.values():
                for branch in tree.branches.values():
                    for node in branch.nodes:
                        if node.moment_id not in seen:
                            seen.add(node.moment_id)
                            moments.append(node.moment_id)
                            if len(moments) >= limit:
                                break
                    if len(moments) >= limit:
                        break
                if len(moments) >= limit:
                    break
        return [self.forecast_next(mid) for mid in moments]


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
from typing import List


class EmotionNarrativeBinder(ModuleBase):
    def __init__(self):
        super().__init__("EmotionNarrativeBinder")

    def bind(self, emotion_history: List[str]) -> List[str]:
        """
        Generates symbolic motifs from emotion-to-emotion transitions.
        """
        motifs = []
        prev = None

        # Normalize
        history = [e.lower() for e in emotion_history if isinstance(e, str)]

        for emo in history:
            if prev is None:
                prev = emo
                continue
            if prev == "joy" and emo == "sadness":
                motifs.append("fall")
            elif prev == "sadness" and emo == "joy":
                motifs.append("rise")
            elif emo == "fear":
                motifs.append("retreat")
            elif emo == "anger":
                motifs.append("confront")
            elif emo == "neutral":
                motifs.append("observe")
            prev = emo

        return motifs


from utils.module_base import ModuleBase
from typing import Dict, Any, List


# === Tier 82 Core Module: NarrativeIntersectionMapper (MAX UPGRADE) ===


class NarrativeIntersectionMapper(ModuleBase):
    """
    Tier 82 Module: Maps overlapping narrative events and divergence between agents.
    Used for symbolic mirror analysis, inter-agent alignment, and multiverse fusion.
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(module_name="NarrativeIntersectionMapper")
        self.context = context

    def map(self, narrative_map: Dict[str, List[str]]) -> Dict[str, Any]:
        all_events = [e for arc in narrative_map.values() for e in arc]
        event_counts = {e: all_events.count(e) for e in set(all_events)}
        common = [e for e, count in event_counts.items() if count > 1]

        divergence = {
            agent_id: len([e for e in arc if e not in common])
            for agent_id, arc in narrative_map.items()
        }

        result = {"common_events": sorted(common), "agent_divergence": divergence}

        # Log intersection mapping
        self.signal_history.append(
            {
                "event": "narrative_intersection_mapped",
                "cycle_ts": (self.context or {}).get("cycle_ts", "unknown"),
                "agent_count": len(narrative_map),
                "common_event_count": len(common),
                "divergence_scores": divergence,
            }
        )

        return result


from utils.module_base import ModuleBase
from typing import Dict, Any, Optional, List


# === Tier 88 Core Module: ConflictNarrativeWeaver (MAX UPGRADE) ===


class ConflictNarrativeWeaver(ModuleBase):
    """
    Translates internal symbolic contradictions into self-narratives
    bound by identity, role, and will-state.
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__(module_name="ConflictNarrativeWeaver")
        self.context = context

    def weave(self, agent_id: str, contradictions: List[str], role: str, will: str) -> List[str]:
        story_fragments = []

        for tag in contradictions:
            if tag == "truth-vs-loyalty":
                msg = f"{agent_id} wrestles with the pull between loyalty and truth — a fracture tests the will '{will}'."
            elif tag == "guardian-vs-rebel":
                msg = f"{agent_id} stands at a crossroads — the Guardian defends order while the Rebel defies it."
            elif tag == "protect-vs-isolation":
                msg = f"{agent_id} longs to shield others yet drifts into solitude — trust is thinning at the edges."
            elif tag == "observer-vs-actor":
                msg = f"{agent_id} watches in silence while action burns within — the Observer cannot stay passive forever."
            else:
                msg = f"{agent_id} is conflicted: {tag} challenges internal cohesion."

            story_fragments.append(msg)

        # Record for meta-symbolic use
        self.signal_history.append(
            {
                "event": "conflict_narrative_generated",
                "cycle_ts": (self.context or {}).get("cycle_ts", "unknown"),
                "agent": agent_id,
                "role": role,
                "will": will,
                "narratives": story_fragments,
            }
        )

        return story_fragments


from utils.module_base import ModuleBase
from typing import Dict, Any, Optional, Tuple, List


class ConvergenceNarrativeSynthesizer(ModuleBase):
    """
    Tier 96 Core Module:
    Synthesizes a symbolic continuity narrative from snapshot and recursive identity signals.
    Highlights trends, anchors, and emerging symbolic tensions.
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__("ConvergenceNarrativeSynthesizer")
        self.context = context

    def summarize(self, elements: Tuple[Dict[str, Any], Dict[str, Any]]) -> str:
        snapshot, recursive = elements

        traits: List[str] = snapshot.get("self_tags", [])
        trend: str = recursive.get("identity_trend", "evolving")
        anchors: List[str] = snapshot.get("identity_signature", {}).get("anchor_tags", [])
        tensions: List[str] = recursive.get("symbolic_tension", [])

        if not traits and not anchors and not tensions:
            return "Insufficient symbolic data to construct a meaningful identity narrative."

        trait_str = ", ".join(traits[:3]) if traits else "undefined traits"
        anchor_str = ", ".join(anchors[:2]) if anchors else "no stable anchors"
        tension_str = ", ".join(tensions[:2]) if tensions else "minimal symbolic tension"

        narrative = (
            f"You began with traits like {trait_str} and have shown a trend toward {trend}. "
            f"Your core has anchored around {anchor_str} while facing symbolic tensions like {tension_str}."
        )

        return narrative


from utils.module_base import ModuleBase
from typing import Dict, Any


class NarrativeStateEngine(ModuleBase):
    """
    Tier MAX++ Module:
    Evaluates alignment between emotion and interpreted symbolic logic (e.g., belief or goal).
    Determines if emotion reinforces, conflicts with, or is neutral to the action.
    """

    def __init__(self):
        super().__init__("NarrativeStateEngine")
        self.emotion_map = {
            "fear": {
                "conflicts": ["attain", "initiate", "risk", "engage"],
                "reinforces": ["avoid", "prevent", "retreat"],
            },
            "hope": {
                "reinforces": ["achieve", "attain", "start", "proceed"],
                "conflicts": ["cease", "halt"],
            },
            "trust": {"reinforces": ["support", "allow", "enable", "share"]},
            "anger": {
                "conflicts": ["tolerate", "permit", "accept"],
                "reinforces": ["block", "demand", "protest"],
            },
            "sadness": {
                "conflicts": ["attain", "achieve", "initiate"],
                "reinforces": ["withdraw", "cease", "disconnect"],
            },
            "joy": {"reinforces": ["start", "attain", "sustain", "celebrate"]},
            "disgust": {
                "reinforces": ["remove", "stop", "reject"],
                "conflicts": ["tolerate", "embrace"],
            },
        }

    def evaluate(self, emotion: str, logic_stmt: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze if the emotion supports or contradicts the interpreted symbolic action.
        """
        emotion = str(emotion or "").lower()
        then_clause = str(logic_stmt.get("THEN", "") or "").lower()
        action = then_clause.split()[0] if then_clause else "undefined"
        negation = logic_stmt.get("NEGATION", False)

        relation = "neutral"
        confidence = 0.5

        if emotion in self.emotion_map:
            mapping = self.emotion_map[emotion]
            if action in mapping.get("reinforces", []):
                relation = "reinforces"
                confidence = 0.9 if not negation else 0.6
            elif action in mapping.get("conflicts", []):
                relation = "conflicts"
                confidence = 0.9 if not negation else 0.6

        return {
            "emotion": emotion,
            "logic": logic_stmt,
            "relation": relation,
            "confidence": confidence,
            "action_extracted": action,
        }


class ArcSynthesizer(ModuleBase):
    """Synthesizes narrative arcs from emotion history, role transitions, and goal tags."""

    def __init__(self):
        super().__init__("ArcSynthesizer")

    def synthesize(self, emotion_history=None, role_history=None, goal_tags=None) -> Dict[str, Any]:
        emotion_history = emotion_history or []
        role_history = role_history or []
        goal_tags = goal_tags or []

        dominant = emotion_history[-1] if emotion_history else "neutral"
        role_count = len(set(role_history))

        if role_count > 3:
            arc = "transformation"
        elif dominant in ("joy", "wonder"):
            arc = "ascension"
        elif dominant in ("fear", "anger"):
            arc = "trial"
        else:
            arc = "continuity"

        return {
            "arc": arc,
            "dominant_emotion": dominant,
            "role_transitions": role_count,
            "goal_tags": goal_tags[:5],
            "status": "synthesized",
        }


class ArcForecaster(ModuleBase):
    """Forecasts future narrative arcs."""

    def __init__(self):
        super().__init__("ArcForecaster")

    def forecast(self, role_history=None, contradiction_log=None) -> Dict[str, Any]:
        role_history = role_history or []
        contradiction_log = contradiction_log or []

        if len(contradiction_log) > 3:
            forecast = "crisis"
        elif len(role_history) > 5:
            forecast = "convergence"
        else:
            forecast = "stable"

        return {
            "forecast": forecast,
            "role_depth": len(role_history),
            "contradictions": len(contradiction_log),
        }


class EmotionBinder(ModuleBase):
    """Binds emotions to narrative arcs."""

    def __init__(self):
        super().__init__("EmotionBinder")

    def bind(self, arc: Dict[str, Any], emotion_history=None) -> str:
        emotion_history = emotion_history or []
        arc_type = arc.get("arc", "continuity") if isinstance(arc, dict) else str(arc)
        dominant = emotion_history[-1] if emotion_history else "neutral"
        return f"{arc_type}_{dominant}"


class NarrativeThreadConstructor(ModuleBase):
    """Constructs narrative threads from memory traces."""

    def __init__(self):
        super().__init__("NarrativeThreadConstructor")

    def construct(self, trace=None, memory_tree=None, tree_name="narrative_tree", branch="main"):
        trace = trace or []
        thread = []
        if memory_tree and hasattr(memory_tree, "get_nodes"):
            nodes = memory_tree.get_nodes(tree_name, branch)
            for node in nodes:
                if node.moment_id in trace or not trace:
                    thread.append(
                        node.to_dict() if hasattr(node, "to_dict") else {"content": str(node)}
                    )
        return thread

    def export_timeline(self, memory_tree=None, tree_name="narrative_tree") -> Dict[str, Any]:
        if not memory_tree:
            return {"timeline": [], "length": 0}
        tree = memory_tree.get_tree(tree_name) if hasattr(memory_tree, "get_tree") else None
        if not tree:
            return {"timeline": [], "length": 0}
        entries = []
        for branch_name, branch in tree.branches.items():
            for node in branch.nodes:
                if hasattr(node, "to_dict"):
                    entries.append(node.to_dict())
                else:
                    entries.append({"content": str(node), "timestamp": 0})
        entries.sort(key=lambda x: x.get("timestamp", 0))
        return {"timeline": entries, "length": len(entries)}
