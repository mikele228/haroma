"""
DiscourseProcessor — Compositional NLU for HaromaX6 (Phase 15).

Builds structured discourse frames from NLU parse output, resolves
references across turns, classifies modality/polarity, detects nesting
from clausal complements, and maintains a discourse state stack
(open questions, commitments, topic graph).
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import hashlib


_MODALITY_AUXILIARIES = {
    "can": "possible",
    "could": "possible",
    "may": "possible",
    "might": "possible",
    "must": "certain",
    "shall": "certain",
    "should": "requested",
    "would": "conditional",
    "will": "certain",
}


@dataclass
class DiscourseFrame:
    frame_type: str  # assertion / command / belief / condition (legacy: question)
    agent: str = ""
    action: str = ""
    patient: str = ""
    modality: str = "certain"  # certain / possible / requested / conditional
    polarity: bool = True  # True = affirmative, False = negated
    nested: List["DiscourseFrame"] = field(default_factory=list)
    source_turn: int = 0

    def content_hash(self) -> str:
        raw = f"{self.frame_type}|{self.agent}|{self.action}|{self.patient}|{self.modality}|{self.polarity}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_type": self.frame_type,
            "agent": self.agent,
            "action": self.action,
            "patient": self.patient,
            "modality": self.modality,
            "polarity": self.polarity,
            "nested": [n.to_dict() for n in self.nested],
            "source_turn": self.source_turn,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiscourseFrame":
        frame = cls(
            frame_type=d.get("frame_type", "assertion"),
            agent=d.get("agent", ""),
            action=d.get("action", ""),
            patient=d.get("patient", ""),
            modality=d.get("modality", "certain"),
            polarity=d.get("polarity", True),
            source_turn=d.get("source_turn", 0),
        )
        frame.nested = [cls.from_dict(n) for n in d.get("nested", [])]
        return frame


@dataclass
class DiscourseResult:
    frames: List[DiscourseFrame] = field(default_factory=list)
    resolved_references: List[Dict[str, str]] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    commitments: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frames": [f.to_dict() for f in self.frames],
            "resolved_references": self.resolved_references,
            "open_questions": self.open_questions,
            "commitments": self.commitments,
            "topics": self.topics,
        }


_INTENT_TO_FRAME = {
    "declarative": "assertion",
    "utterance": "assertion",
    "statement": "assertion",
    "interrogative": "assertion",
    "question": "assertion",
    "imperative": "command",
    "exclamatory": "assertion",
    "none": "assertion",
}

_BELIEF_VERBS = frozenset(
    {
        "think",
        "believe",
        "know",
        "suspect",
        "assume",
        "suppose",
        "feel",
        "hope",
        "wish",
        "doubt",
        "imagine",
        "realize",
        "understand",
        "expect",
    }
)

_CONDITIONAL_MARKERS = frozenset(
    {
        "if",
        "unless",
        "provided",
        "assuming",
        "whether",
    }
)


class DiscourseProcessor:
    """Builds structured discourse frames, resolves references,
    and maintains discourse state across turns."""

    def __init__(self):
        self._qud_stack: List[str] = []
        self._commitments: List[str] = []
        self._topic_graph: List[Dict[str, Any]] = []
        self._entity_registry: Dict[str, str] = {}

    def process(
        self,
        nlu_result: Dict[str, Any],
        conversation_history: Optional[List[Dict]] = None,
        cycle_id: int = 0,
    ) -> DiscourseResult:
        """Build discourse frames from NLU output."""
        if not nlu_result:
            return DiscourseResult()

        frames = self._build_frames(nlu_result, cycle_id)

        sub_clauses = nlu_result.get("subordinate_clauses", [])
        for sc in sub_clauses:
            nested_frame = DiscourseFrame(
                frame_type="belief" if sc.get("dep") in ("ccomp", "xcomp") else "condition",
                agent=sc.get("subject", ""),
                action=sc.get("verb", ""),
                patient=sc.get("object", ""),
                modality=self._classify_modality(sc.get("aux_verbs", [])),
                polarity=not sc.get("negated", False),
                source_turn=cycle_id,
            )
            if frames:
                frames[0].nested.append(nested_frame)

        entity_mentions = self._collect_entity_mentions(nlu_result)
        resolved = self._resolve_references(nlu_result, entity_mentions, conversation_history)

        nlu_with_cycle = dict(nlu_result)
        nlu_with_cycle["_cycle_id"] = cycle_id
        self._update_discourse_state(frames, nlu_with_cycle)

        return DiscourseResult(
            frames=frames,
            resolved_references=resolved,
            open_questions=list(self._qud_stack[-5:]),
            commitments=list(self._commitments[-10:]),
            topics=[t.get("topic", "") for t in self._topic_graph[-5:]],
        )

    def _build_frames(self, nlu_result: Dict[str, Any], cycle_id: int) -> List[DiscourseFrame]:
        intent = nlu_result.get("intent", "declarative")
        frame_type = _INTENT_TO_FRAME.get(intent, "assertion")
        relations = nlu_result.get("relations", [])
        aux_verbs = nlu_result.get("aux_verbs", [])
        modality = self._classify_modality(aux_verbs)

        frames: List[DiscourseFrame] = []
        if relations:
            for rel in relations:
                verb = rel.get("predicate", "")
                is_belief = verb.replace("not_", "") in _BELIEF_VERBS
                ft = "belief" if is_belief else frame_type

                frame = DiscourseFrame(
                    frame_type=ft,
                    agent=rel.get("subject", ""),
                    action=verb,
                    patient=rel.get("object", ""),
                    modality=modality,
                    polarity=not rel.get("negated", False),
                    source_turn=cycle_id,
                )
                frames.append(frame)
        else:
            frames.append(
                DiscourseFrame(
                    frame_type=frame_type,
                    modality=modality,
                    polarity=not nlu_result.get("negated", False),
                    source_turn=cycle_id,
                )
            )

        return frames

    def _classify_modality(self, aux_verbs: List[str]) -> str:
        for aux in aux_verbs:
            m = _MODALITY_AUXILIARIES.get(aux.lower())
            if m:
                return m
        return "certain"

    def _collect_entity_mentions(self, nlu_result: Dict[str, Any]) -> Dict[str, str]:
        mentions: Dict[str, str] = {}
        for ent in nlu_result.get("entities", []):
            text = ent.get("text", "").lower()
            etype = ent.get("type", "CONCEPT")
            if text:
                mentions[text] = etype
                self._entity_registry[text] = etype
        return mentions

    def _resolve_references(
        self,
        nlu_result: Dict[str, Any],
        current_entities: Dict[str, str],
        conversation_history: Optional[List[Dict]],
    ) -> List[Dict[str, str]]:
        resolved: List[Dict[str, str]] = []

        for coref in nlu_result.get("coref_entities", []):
            pronoun = coref.get("pronoun", "")
            antecedent = coref.get("antecedent", "")
            if antecedent:
                resolved.append(
                    {
                        "pronoun": pronoun,
                        "resolved_to": antecedent,
                        "source": "nlu_coref",
                    }
                )

        for rel in nlu_result.get("relations", []):
            for role in ("subject", "object"):
                text = rel.get(role, "").lower()
                if text in ("it", "that", "this", "they", "them", "the one"):
                    if self._entity_registry:
                        candidates = list(self._entity_registry.keys())
                        best = candidates[-1] if candidates else text
                        resolved.append(
                            {
                                "pronoun": text,
                                "resolved_to": best,
                                "source": "registry_recency",
                            }
                        )

        return resolved

    def _update_discourse_state(self, frames: List[DiscourseFrame], nlu_result: Dict[str, Any]):
        for frame in frames:
            if frame.frame_type == "command":
                c = f"{frame.action} {frame.patient}".strip()
                if c and c not in self._commitments:
                    self._commitments.append(c)
                    if len(self._commitments) > 30:
                        self._commitments = self._commitments[-30:]

        noun_phrases = nlu_result.get("noun_phrases", [])
        for np_text in noun_phrases[:3]:
            self._topic_graph.append(
                {
                    "topic": np_text,
                    "turn": nlu_result.get("_cycle_id", 0),
                }
            )
        if len(self._topic_graph) > 50:
            self._topic_graph = self._topic_graph[-50:]

    def get_open_questions(self) -> List[str]:
        return list(self._qud_stack[-5:])

    def get_recent_entity_mentions(self, n: int = 10) -> List[str]:
        return list(self._entity_registry.keys())[-n:]

    def answer_question(self, question_text: str):
        """Mark a question as answered (remove from QUD)."""
        self._qud_stack = [q for q in self._qud_stack if q != question_text]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qud_stack": self._qud_stack,
            "commitments": self._commitments,
            "topic_graph": self._topic_graph[-30:],
            "entity_registry": dict(list(self._entity_registry.items())[-100:]),
        }

    def from_dict(self, data: Dict[str, Any]):
        self._qud_stack = data.get("qud_stack", [])
        self._commitments = data.get("commitments", [])
        self._topic_graph = data.get("topic_graph", [])
        self._entity_registry = data.get("entity_registry", {})

    def stats(self) -> Dict[str, Any]:
        return {
            "open_questions": len(self._qud_stack),
            "commitments": len(self._commitments),
            "topics_tracked": len(self._topic_graph),
            "entities_registered": len(self._entity_registry),
        }
