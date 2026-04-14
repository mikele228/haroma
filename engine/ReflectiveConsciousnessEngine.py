from typing import Dict, Any, List
from utils.module_base import ModuleBase
from engine.EmotionEngine import EmotionEngine
from engine.IdentityEngine import IdentityEngine
from core.Dream import DreamCore
from core.Goal import get_shared_goal_engine


class ReflectiveConsciousnessEngine(ModuleBase):
    """Aggregates goal, emotion, dream, and context state into a reflective consciousness layer."""

    def __init__(self):
        super().__init__("ReflectiveConsciousnessEngine")
        self.goal = get_shared_goal_engine()
        self.emotion = EmotionEngine()
        self.dream = DreamCore()
        self.identity = IdentityEngine()
        self.context_state: Dict[str, Any] = {"active_goals": []}

    def reflect(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "emotion": self.emotion.summarize(),
            "goals": self.goal.summarize(),
            "dream": self.dream.summarize(),
            "identity": self.identity.summarize(),
        }

    def reset(self):
        self.emotion.reset()
        self.goal.reset()
        self.dream.reset()
        self.identity.reset()
        self.context_state = {"active_goals": []}
