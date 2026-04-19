"""
Global Workspace Theory implementation for HaromaX6.

A fixed-capacity workspace where cognitive subsystems compete for
"conscious access."  Only the highest-salience coalitions are
broadcast; the rest remain unconscious but still stored for later
inspection.  This forces genuine integration -- subsystems must
produce salient output to influence downstream processing.
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field
import time
import threading


@dataclass
class Coalition:
    source: str
    content: Dict[str, Any]
    salience: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "content": self.content,
            "salience": round(self.salience, 4),
            "timestamp": self.timestamp,
        }


class GlobalWorkspace:
    _MAX_HISTORY = 500

    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        self._candidates: List[Coalition] = []
        self._contents: List[Coalition] = []
        self._unconscious: List[Coalition] = []
        self._history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def broadcast(self, source: str, content: Dict[str, Any], salience: float):
        with self._lock:
            self._candidates.append(
                Coalition(
                    source=source,
                    content=content,
                    salience=max(0.0, min(1.0, salience)),
                )
            )

    def select(self) -> List[Coalition]:
        with self._lock:
            self._candidates.sort(key=lambda c: c.salience, reverse=True)
            self._contents = self._candidates[: self.capacity]
            self._unconscious = self._candidates[self.capacity :]
            self._history.append(
                {
                    "timestamp": time.time(),
                    "winners": [c.source for c in self._contents],
                    "losers": [c.source for c in self._unconscious],
                    "top_salience": self._contents[0].salience if self._contents else 0.0,
                }
            )
            if len(self._history) > self._MAX_HISTORY:
                self._history = self._history[-self._MAX_HISTORY :]
            self._candidates = []
            return list(self._contents)

    def get_contents(self) -> List[Coalition]:
        with self._lock:
            return list(self._contents)

    def get_unconscious(self) -> List[Coalition]:
        with self._lock:
            return list(self._unconscious)

    def integrate(self) -> Dict[str, Any]:
        """Merge all winning coalitions into a single integrated representation."""
        with self._lock:
            if not self._contents:
                return {"integrated": False, "sources": []}
            merged: Dict[str, Any] = {
                "integrated": True,
                "sources": [],
                "total_salience": 0.0,
            }
            for c in self._contents:
                merged["sources"].append(c.source)
                merged["total_salience"] += c.salience
                for k, v in c.content.items():
                    key = f"{c.source}_{k}"
                    merged[key] = v
            merged["total_salience"] = round(merged["total_salience"], 3)
            merged["integration_density"] = round(
                merged["total_salience"] / max(self.capacity, 1), 3
            )
            return merged

    def clear(self):
        with self._lock:
            self._candidates.clear()
            self._contents.clear()
            self._unconscious.clear()

    def stats(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "current_winners": len(self._contents),
            "current_unconscious": len(self._unconscious),
            "total_selections": len(self._history),
        }
