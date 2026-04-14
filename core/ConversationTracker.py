"""
ConversationTracker — dialogue awareness for HaromaX6.

Tracks turns (who said what, when), detects topic shifts, and
provides conversation context summaries for the ActionGenerator
so responses are coherent across multiple exchanges.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import Counter
import time
import threading


@dataclass
class Turn:
    speaker: str
    content: str
    timestamp: float
    cycle_id: int
    emotion: str = "neutral"
    tags: List[str] = field(default_factory=list)
    response: str = ""
    _discourse: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "speaker": self.speaker,
            "content": self.content,
            "timestamp": self.timestamp,
            "cycle_id": self.cycle_id,
            "emotion": self.emotion,
            "tags": self.tags,
            "response": self.response,
        }
        if self._discourse:
            d["discourse"] = self._discourse
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Turn":
        return cls(
            speaker=d.get("speaker", "unknown"),
            content=d.get("content", ""),
            timestamp=d.get("timestamp", 0.0),
            cycle_id=d.get("cycle_id", 0),
            emotion=d.get("emotion", "neutral"),
            tags=d.get("tags", []),
            response=d.get("response", ""),
            _discourse=d.get("discourse"),
        )


class ConversationTracker:
    def __init__(self, max_history: int = 200):
        self.history: List[Turn] = []
        self.max_history = max_history
        self.active_topic: str = ""
        self._last_external_cycle: int = 0
        self._lock = threading.RLock()

    def record_input(
        self,
        content: str,
        speaker: str,
        cycle_id: int,
        emotion: str = "neutral",
        tags: Optional[List[str]] = None,
    ):
        with self._lock:
            if not content or content == "idle state":
                return

            turn = Turn(
                speaker=speaker,
                content=content,
                timestamp=time.time(),
                cycle_id=cycle_id,
                emotion=emotion,
                tags=tags or [],
            )
            self.history.append(turn)
            self._last_external_cycle = cycle_id

            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history :]

            self._update_topic(tags or [])

    def record_response(self, response: str, cycle_id: int):
        with self._lock:
            for turn in reversed(self.history):
                if turn.cycle_id == cycle_id and not turn.response:
                    turn.response = response
                    break

    def get_recent(self, n: int = 5, *, speaker: Optional[str] = None) -> List[Turn]:
        """Recent turns, optionally restricted to one interlocutor (``speaker`` key)."""
        with self._lock:
            if speaker is None:
                return self.history[-n:]
            matching = [t for t in self.history if t.speaker == speaker]
            return matching[-n:] if matching else []

    def get_last_input(self) -> Optional[Turn]:
        with self._lock:
            for turn in reversed(self.history):
                if turn.speaker != "self":
                    return turn
            return None

    def get_context_summary(self, speaker: Optional[str] = None) -> str:
        """Summarize recent dialogue. When *speaker* is set, only that interlocutor's turns."""
        with self._lock:
            if not self.history:
                return ""

            if speaker is None:
                recent = self.history[-4:]
            else:
                matching = [t for t in self.history if t.speaker == speaker]
                # Do not fall back to global history — would mix other users' dialogue.
                recent = matching[-4:] if matching else []
            if not recent:
                return ""

            parts: List[str] = []
            for turn in recent:
                speaker_label = turn.speaker if turn.speaker != "self" else "I"
                snippet = turn.content[:50]
                parts.append(f"{speaker_label}: {snippet}")
                if turn.response:
                    parts.append(f"I replied: {turn.response[:50]}")

            # ``active_topic`` is derived from global tag co-occurrence; omit when scoping
            # to one speaker so we do not leak another interlocutor's topic.
            if speaker is None and self.active_topic:
                parts.append(f"Current topic: {self.active_topic}")

            return " | ".join(parts)

    def detect_topic_shift(self, current_tags: List[str]) -> bool:
        with self._lock:
            if not self.history or not current_tags:
                return False

            recent_tags: List[str] = []
            for turn in self.history[-5:]:
                recent_tags.extend(turn.tags)

            if not recent_tags:
                return False

            recent_set = set(recent_tags)
            current_set = set(current_tags)
            overlap = len(recent_set & current_set)
            union = len(recent_set | current_set)

            if union == 0:
                return False

            similarity = overlap / union
            return similarity < 0.15

    def get_topic(self) -> str:
        with self._lock:
            return self.active_topic

    def is_in_conversation(self, current_cycle: int) -> bool:
        with self._lock:
            if not self.history:
                return False
            return (current_cycle - self._last_external_cycle) <= 5

    def _update_topic(self, new_tags: List[str]):
        recent_tags: List[str] = []
        for turn in self.history[-8:]:
            recent_tags.extend(turn.tags)

        if not recent_tags:
            self.active_topic = ""
            return

        common = Counter(recent_tags).most_common(3)
        self.active_topic = ", ".join(t[0] for t in common)

    def turn_count(self) -> int:
        with self._lock:
            return len(self.history)

    def store_discourse_snapshot(self, cycle_id: int, discourse_dict: Dict[str, Any]):
        """Attach discourse state to the matching turn."""
        with self._lock:
            for turn in reversed(self.history):
                if turn.cycle_id == cycle_id:
                    if turn._discourse is None:
                        turn._discourse = discourse_dict
                    break

    def get_open_questions(self) -> List[str]:
        """Retrieve open questions from the most recent discourse snapshot."""
        with self._lock:
            for turn in reversed(self.history):
                disc = getattr(turn, "_discourse", None)
                if disc:
                    return disc.get("open_questions", [])
            return []

    def get_recent_entity_mentions(self, n: int = 10) -> List[str]:
        """Collect recent entity mentions from discourse snapshots."""
        with self._lock:
            entities: List[str] = []
            for turn in reversed(self.history[-20:]):
                disc = getattr(turn, "_discourse", None)
                if disc:
                    for topic in disc.get("topics", []):
                        if topic and topic not in entities:
                            entities.append(topic)
                        if len(entities) >= n:
                            return entities
            return entities

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "history": [t.to_dict() for t in self.history[-self.max_history :]],
                "active_topic": self.active_topic,
                "last_external_cycle": self._last_external_cycle,
            }

    def from_dict(self, data: Dict[str, Any]):
        with self._lock:
            self.history = [Turn.from_dict(d) for d in data.get("history", [])]
            self.active_topic = data.get("active_topic", "")
            self._last_external_cycle = data.get("last_external_cycle", 0)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "turn_count": len(self.history),
                "active_topic": self.active_topic,
                "last_external_cycle": self._last_external_cycle,
                "unique_speakers": len(set(t.speaker for t in self.history)),
            }
