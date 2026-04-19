"""
WorkingMemory — cross-cycle short-term buffer for HaromaX6.

Bridges the gap between the per-cycle GlobalWorkspace (which resets
every cycle) and long-term MemoryForest. Holds recent perceptions,
active thoughts, conversation fragments, and goal context with
salience-based decay.

Capacity ~12 items (cognitive science: 7 +/- 5).
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import hashlib
import threading


@dataclass
class WorkingMemoryItem:
    content: str
    source: str
    salience: float
    cycle_added: int
    item_type: str  # percept, thought, dialogue, goal, insight
    tags: List[str] = field(default_factory=list)
    decay_rate: float = 0.08
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.md5(self.content.encode()).hexdigest()[:10]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "source": self.source,
            "salience": round(self.salience, 3),
            "cycle_added": self.cycle_added,
            "item_type": self.item_type,
            "tags": self.tags,
            "decay_rate": self.decay_rate,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkingMemoryItem":
        return cls(
            content=d.get("content", ""),
            source=d.get("source", ""),
            salience=d.get("salience", 0.5),
            cycle_added=d.get("cycle_added", 0),
            item_type=d.get("item_type", "thought"),
            tags=d.get("tags", []),
            decay_rate=d.get("decay_rate", 0.08),
            content_hash=d.get("content_hash", ""),
        )


class WorkingMemory:
    def __init__(self, capacity: int = 12):
        self.capacity = capacity
        self.items: List[WorkingMemoryItem] = []
        self._eviction_threshold = 0.05
        self._lock = threading.RLock()

    def add(
        self,
        content: str,
        source: str,
        salience: float,
        item_type: str = "thought",
        tags: Optional[List[str]] = None,
        cycle: int = 0,
        decay_rate: float = 0.08,
    ):
        with self._lock:
            content_hash = hashlib.md5(content.encode()).hexdigest()[:10]
            if self.has_item(content_hash):
                for item in self.items:
                    if item.content_hash == content_hash:
                        item.salience = max(item.salience, salience)
                        return
                return

            new_item = WorkingMemoryItem(
                content=content,
                source=source,
                salience=salience,
                cycle_added=cycle,
                item_type=item_type,
                tags=tags or [],
                decay_rate=decay_rate,
                content_hash=content_hash,
            )

            if len(self.items) >= self.capacity:
                min_idx = min(range(len(self.items)), key=lambda i: self.items[i].salience)
                if self.items[min_idx].salience < salience:
                    self.items[min_idx] = new_item
            else:
                self.items.append(new_item)

    def tick(self, current_cycle: int):
        with self._lock:
            for item in self.items:
                age = max(1, current_cycle - item.cycle_added)
                decay = item.decay_rate * min(age / 10.0, 2.0)
                item.salience = max(0.0, item.salience - decay)

            self.items = [i for i in self.items if i.salience >= self._eviction_threshold]

    def get_context(
        self, item_type: Optional[str] = None, limit: Optional[int] = None
    ) -> List[WorkingMemoryItem]:
        with self._lock:
            result = self.items
            if item_type:
                result = [i for i in result if i.item_type == item_type]
            result = sorted(result, key=lambda i: i.salience, reverse=True)
            if limit:
                result = result[:limit]
            return result

    def promote_from_workspace(self, workspace_contents, cycle: int = 0):
        with self._lock:
            for coalition in workspace_contents:
                if hasattr(coalition, "to_dict"):
                    c = coalition.to_dict()
                else:
                    c = coalition if isinstance(coalition, dict) else {}

                source = c.get("source", "workspace")
                salience = c.get("salience", 0.5)
                content_data = c.get("content", {})

                if isinstance(content_data, dict):
                    text = content_data.get("content", "")
                    if not text:
                        text = str(content_data)[:80]
                else:
                    text = str(content_data)[:80]

                if not text or len(text) < 3:
                    continue

                content_hash = hashlib.md5(text.encode()).hexdigest()[:10]
                if not self.has_item(content_hash):
                    self.add(
                        content=text,
                        source=source,
                        salience=salience * 0.8,
                        item_type="thought",
                        cycle=cycle,
                        decay_rate=0.06,
                    )

    def has_item(self, content_hash: str) -> bool:
        with self._lock:
            return any(i.content_hash == content_hash for i in self.items)

    def occupancy(self) -> float:
        """Fraction of capacity currently in use (0.0 .. 1.0)."""
        with self._lock:
            return len(self.items) / max(self.capacity, 1)

    def to_context_string(self, limit: int = 5) -> str:
        with self._lock:
            top = self.get_context(limit=limit)
            if not top:
                return ""
            parts = []
            for item in top:
                snippet = item.content[:60]
                parts.append(f"[{item.item_type}] {snippet}")
            return " | ".join(parts)

    def to_dict(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [i.to_dict() for i in self.items]

    def from_dict(self, data: List[Dict[str, Any]]):
        with self._lock:
            self.items = [WorkingMemoryItem.from_dict(d) for d in data]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            type_counts: Dict[str, int] = {}
            for item in self.items:
                type_counts[item.item_type] = type_counts.get(item.item_type, 0) + 1
            avg_sal = sum(i.salience for i in self.items) / len(self.items) if self.items else 0.0
            return {
                "item_count": len(self.items),
                "capacity": self.capacity,
                "type_counts": type_counts,
                "avg_salience": round(avg_sal, 3),
            }
