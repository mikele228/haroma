"""
InterlocutorModel — Theory of Mind for HaromaX6.

Builds and maintains a persistent, evolving model of each
interlocutor from conversation turns and NLU output. Tracks
inferred beliefs, goals, emotional state, topics of interest,
and communication style per speaker.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class InterlocutorState:
    speaker: str
    inferred_beliefs: List[str] = field(default_factory=list)
    belief_frames: List[Dict[str, Any]] = field(default_factory=list)
    inferred_goals: List[str] = field(default_factory=list)
    emotional_state: Dict[str, float] = field(
        default_factory=lambda: {
            "polarity": 0.0,
            "subjectivity": 0.0,
        }
    )
    topics_of_interest: List[str] = field(default_factory=list)
    interaction_count: int = 0
    communication_style: str = "unknown"
    last_seen_cycle: int = 0
    _intent_counts: Dict[str, int] = field(default_factory=lambda: Counter())
    _entity_counts: Dict[str, int] = field(default_factory=lambda: Counter())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker": self.speaker,
            "inferred_beliefs": self.inferred_beliefs[-20:],
            "belief_frames": self.belief_frames[-30:],
            "inferred_goals": self.inferred_goals[-20:],
            "emotional_state": self.emotional_state,
            "topics_of_interest": self.topics_of_interest[:10],
            "interaction_count": self.interaction_count,
            "communication_style": self.communication_style,
            "last_seen_cycle": self.last_seen_cycle,
            "intent_counts": dict(self._intent_counts),
            "entity_counts": dict(self._entity_counts),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InterlocutorState":
        state = cls(speaker=d.get("speaker", "unknown"))
        state.inferred_beliefs = d.get("inferred_beliefs", [])
        state.belief_frames = d.get("belief_frames", [])
        state.inferred_goals = d.get("inferred_goals", [])
        state.emotional_state = d.get("emotional_state", {"polarity": 0.0, "subjectivity": 0.0})
        state.topics_of_interest = d.get("topics_of_interest", [])
        state.interaction_count = d.get("interaction_count", 0)
        state.communication_style = d.get("communication_style", "unknown")
        state.last_seen_cycle = d.get("last_seen_cycle", 0)
        state._intent_counts = Counter(d.get("intent_counts", {}))
        state._entity_counts = Counter(d.get("entity_counts", {}))
        return state

    def summary(self) -> Dict[str, Any]:
        return {
            "speaker": self.speaker,
            "beliefs": len(self.inferred_beliefs),
            "belief_frames": len(self.belief_frames),
            "goals": len(self.inferred_goals),
            "polarity": round(self.emotional_state.get("polarity", 0.0), 3),
            "top_topics": self.topics_of_interest[:5],
            "style": self.communication_style,
            "interactions": self.interaction_count,
        }


_SMOOTHING = 0.3


class InterlocutorModel:
    """Maintains per-speaker mental models from conversation + NLU."""

    def __init__(self, max_speakers: int = 50):
        self.speakers: Dict[str, InterlocutorState] = {}
        self.max_speakers = max_speakers

    def update(
        self,
        speaker: str,
        content: str,
        nlu_result: Dict[str, Any],
        cycle_id: int,
        discourse_frames: Optional[List[Dict[str, Any]]] = None,
    ):
        if speaker in ("idle", "self") or not content:
            return

        state = self._get_or_create(speaker)
        state.interaction_count += 1
        state.last_seen_cycle = cycle_id

        intent = nlu_result.get("intent", "declarative")
        state._intent_counts[intent] = state._intent_counts.get(intent, 0) + 1

        if discourse_frames:
            existing_hashes = set()
            for bf in state.belief_frames:
                h = bf.get("content_hash", "")
                if h:
                    existing_hashes.add(h)

            for frame_dict in discourse_frames:
                import hashlib as _hl

                raw = (
                    f"{frame_dict.get('frame_type', '')}|"
                    f"{frame_dict.get('agent', '')}|"
                    f"{frame_dict.get('action', '')}|"
                    f"{frame_dict.get('patient', '')}"
                )
                h = _hl.md5(raw.encode()).hexdigest()[:12]
                if h not in existing_hashes:
                    frame_dict["content_hash"] = h
                    state.belief_frames.append(frame_dict)
                    existing_hashes.add(h)
            if len(state.belief_frames) > 60:
                state.belief_frames = state.belief_frames[-60:]

        if intent not in ("imperative",):
            for rel in nlu_result.get("relations", []):
                subj = rel.get("subject", "")
                pred = rel.get("predicate", "")
                obj = rel.get("object", "")
                if subj and pred and obj and not rel.get("negated"):
                    belief = f"{subj} {pred} {obj}"
                    if belief not in state.inferred_beliefs:
                        state.inferred_beliefs.append(belief)
                        if len(state.inferred_beliefs) > 50:
                            state.inferred_beliefs = state.inferred_beliefs[-50:]

        if intent == "imperative":
            goal = f"request: {content[:50]}"
            if goal not in state.inferred_goals:
                state.inferred_goals.append(goal)
                if len(state.inferred_goals) > 50:
                    state.inferred_goals = state.inferred_goals[-50:]

        sentiment = nlu_result.get("sentiment", {})
        new_pol = sentiment.get("polarity", 0.0)
        new_subj = sentiment.get("subjectivity", 0.0)
        old_pol = state.emotional_state.get("polarity", 0.0)
        old_subj = state.emotional_state.get("subjectivity", 0.0)
        state.emotional_state["polarity"] = round(
            old_pol * (1 - _SMOOTHING) + new_pol * _SMOOTHING, 3
        )
        state.emotional_state["subjectivity"] = round(
            old_subj * (1 - _SMOOTHING) + new_subj * _SMOOTHING, 3
        )

        for ent in nlu_result.get("entities", []):
            name = ent.get("text", "").lower()
            if name:
                state._entity_counts[name] = state._entity_counts.get(name, 0) + 1

        top_entities = sorted(state._entity_counts.items(), key=lambda x: x[1], reverse=True)
        state.topics_of_interest = [e[0] for e in top_entities[:10]]

        state.communication_style = self._classify_style(state)

    def get_model(self, speaker: str) -> Optional[InterlocutorState]:
        return self.speakers.get(speaker)

    def get_model_summary(self, speaker: str) -> Dict[str, Any]:
        state = self.speakers.get(speaker)
        if not state:
            return {"speaker": speaker, "known": False}
        d = state.summary()
        d["known"] = True
        return d

    def predict_interest(self, speaker: str) -> List[str]:
        state = self.speakers.get(speaker)
        if not state:
            return []
        return list(state.topics_of_interest[:5])

    def predict_reaction(self, speaker: str, proposed_strategy: str) -> float:
        state = self.speakers.get(speaker)
        if not state:
            return 0.5

        style = state.communication_style
        polarity = state.emotional_state.get("polarity", 0.0)

        style_prefs = {
            "inquisitive": {
                "inquire": 0.5,
                "inform": 0.9,
                "empathize": 0.5,
                "advance_goal": 0.6,
                "reflect": 0.3,
            },
            "directive": {
                "inquire": 0.4,
                "inform": 0.6,
                "empathize": 0.4,
                "advance_goal": 0.9,
                "reflect": 0.3,
            },
            "expressive": {
                "inquire": 0.5,
                "inform": 0.5,
                "empathize": 0.9,
                "advance_goal": 0.4,
                "reflect": 0.6,
            },
            "informative": {
                "inquire": 0.6,
                "inform": 0.7,
                "empathize": 0.4,
                "advance_goal": 0.6,
                "reflect": 0.5,
            },
            "unknown": {
                "inquire": 0.5,
                "inform": 0.5,
                "empathize": 0.5,
                "advance_goal": 0.5,
                "reflect": 0.5,
            },
        }
        base = style_prefs.get(style, style_prefs["unknown"]).get(proposed_strategy, 0.5)

        if polarity < -0.3 and proposed_strategy == "empathize":
            base = min(1.0, base + 0.15)
        if polarity > 0.3 and proposed_strategy == "inform":
            base = min(1.0, base + 0.1)

        return round(base, 3)

    def _get_or_create(self, speaker: str) -> InterlocutorState:
        if speaker not in self.speakers:
            if len(self.speakers) >= self.max_speakers:
                oldest = min(self.speakers.values(), key=lambda s: s.last_seen_cycle)
                del self.speakers[oldest.speaker]
            self.speakers[speaker] = InterlocutorState(speaker=speaker)
        return self.speakers[speaker]

    def _classify_style(self, state: InterlocutorState) -> str:
        counts = state._intent_counts
        total = sum(counts.values())
        if total < 2:
            return "unknown"

        imper = counts.get("imperative", 0) / total
        exclam = counts.get("exclamatory", 0) / total

        if imper > 0.3:
            return "directive"
        if exclam > 0.3 or state.emotional_state.get("subjectivity", 0) > 0.5:
            return "expressive"
        return "informative"

    def to_dict(self) -> Dict[str, Any]:
        return {speaker: state.to_dict() for speaker, state in self.speakers.items()}

    def from_dict(self, data: Dict[str, Any]):
        self.speakers.clear()
        for speaker, state_data in data.items():
            self.speakers[speaker] = InterlocutorState.from_dict(state_data)

    def stats(self) -> Dict[str, Any]:
        return {
            "tracked_speakers": len(self.speakers),
            "total_interactions": sum(s.interaction_count for s in self.speakers.values()),
        }
