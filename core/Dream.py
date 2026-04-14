from utils.module_base import ModuleBase
from typing import Dict, Any, List, Optional
import random
import time


# === Merged DreamCore ===
class DreamCore(ModuleBase):
    """
    Tier MAX+ Composite: Synthesizes symbolic dreams.
    Supports simulation, fusion, feedback binding, and narrative classification.
    """

    def __init__(self, memory_engine=None):
        super().__init__(module_name=self.__class__.__name__)
        self.memory_engine = memory_engine
        self.load_state()
        self.dreams: List[Dict[str, Any]] = self.state.get("dreams", [])
        self.classified: List[str] = self.state.get("classified", [])
        self.symbols: List[str] = self.state.get("symbols", [])
        self.feedback_log: List[Dict[str, Any]] = self.state.get("feedback_log", [])

    def generate_dream(
        self, seed: Optional[str] = None, archetype: Optional[str] = None
    ) -> Dict[str, Any]:
        seed_text = seed or f"fragment-{random.randint(1000, 9999)}"
        archetype = archetype or random.choice(["liberation", "entrapment", "transformation"])
        dream = {
            "theme": archetype,
            "content": f"A symbolic sequence around {seed_text}",
            "timestamp": time.time(),
        }
        self.dreams.append(dream)
        self.state["dreams"] = self.dreams
        return dream

    def classify_dream(self, dream: Dict[str, Any]) -> str:
        theme = dream.get("theme", "unknown")
        if "liberation" in theme:
            label = "positive"
        elif "trap" in theme or "entrapment" in theme:
            label = "conflicted"
        elif "transformation" in theme:
            label = "transformative"
        else:
            label = "neutral"
        self.classified.append(label)
        self.state["classified"] = self.classified
        return label

    def fuse_symbols(self, tags: List[str]):
        self.symbols.extend(tags)
        self.symbols = list(set(self.symbols))
        self.state["symbols"] = self.symbols

    def bind_feedback(self, insight: str, overlay_tags: Optional[List[str]] = None):
        feedback = {"insight": insight, "tags": overlay_tags or [], "timestamp": time.time()}
        self.feedback_log.append(feedback)
        self.state["feedback_log"] = self.feedback_log

    def summarize(self) -> Dict[str, Any]:
        return {
            "dream_count": len(self.dreams),
            "classified_labels": self.classified[-3:],
            "symbols_tracked": len(self.symbols),
            "last_feedback": self.feedback_log[-1] if self.feedback_log else None,
        }

    def reset(self):
        self.dreams.clear()
        self.classified.clear()
        self.symbols.clear()
        self.feedback_log.clear()
        self.state.clear()

    def __repr__(self):
        return f"<DreamSynthesizer dreams={len(self.dreams)} symbols={len(self.symbols)}>"
