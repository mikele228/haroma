"""
Unified Manager System: Elarion
Cognitive managers — simple engine wrappers used by the control loop.
"""

import threading
from typing import Dict, Any, List, Optional

from core.engine.LawEngine import LAW_SOURCE_EXTERNAL

from core.Memory import MemoryForest

from engine.ReflectiveConsciousnessEngine import ReflectiveConsciousnessEngine
from core.Goal import get_shared_goal_engine
from engine.FusionResolver import FusionResolver
from engine.IdentityEngine import IdentityEngine
from engine.MemorySummarizer import MemorySummarizer
from engine.NarrativeReflector import NarrativeReflector
from engine.LoopMemoryLogger import LoopMemoryLogger
from engine.SymbolicLoopManager import SymbolicLoopManager
from engine.DoctrineCortex import DoctrineCortex
from engine.EmotionEngine import EmotionEngine

from core.AffectiveReasoning import EmotionBeliefDriftDetector
from core.Goal import GoalCollapseDetector
from core.Narrative import NarrativeForecaster


# ============================================================
# Simple Engine-Wrapper Managers
# ============================================================


class DreamManager:
    def __init__(self):
        self.engine = ReflectiveConsciousnessEngine().dream

    def generate(
        self, seed: Optional[str] = None, archetype: Optional[str] = None
    ) -> Dict[str, Any]:
        return self.engine.generate_dream(seed, archetype)

    def classify(self, dream: Dict[str, Any]) -> str:
        return self.engine.classify_dream(dream)

    def fuse_symbols(self, tags: List[str]):
        if tags:
            self.engine.fuse_symbols(list(tags))

    def bind_feedback(self, insight: str, overlay_tags: Optional[List[str]] = None):
        self.engine.bind_feedback(insight, overlay_tags)

    def summarize(self) -> Dict[str, Any]:
        return self.engine.summarize()

    def reset(self):
        self.engine.reset()


class EmotionManagerSimple:
    def __init__(self):
        self.engine = EmotionEngine()

    def ingest(self, data: Dict[str, Any]):
        self.engine.ingest(data)

    def update_emotion(self, label: str, intensity: float, context: Dict[str, Any] = None):
        self.engine.update_emotion(label, intensity, context)

    def apply_decay(self, decay: float = 0.01):
        self.engine.apply_decay(decay)

    def summarize(self) -> Dict[str, Any]:
        return self.engine.summarize()

    def reset(self):
        self.engine.reset()


class FusionManager:
    def __init__(self):
        self.engine = FusionResolver()

    def fuse(self, a: Dict[str, Any], b: Dict[str, Any], mode: str = "merge") -> Dict[str, Any]:
        return self.engine.fuse(a, b, mode)

    def resolve(self, cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.engine.resolve_conflicts(cluster)

    def summarize(self) -> Dict[str, Any]:
        return self.engine.summarize()

    def reset(self):
        self.engine.reset()


class GoalManager:
    def __init__(self):
        # Shared with all agents and ElarionController (single Fuel substrate).
        self.engine = get_shared_goal_engine()

    def register_goal(
        self, goal_id: str, description: str, priority: float = 0.5, source: str = "system"
    ):
        self.engine.register_goal(goal_id, description, priority, source)

    def bump_goal_priority(self, goal_id: str, delta: float) -> bool:
        return self.engine.bump_goal_priority(goal_id, delta)

    def register_input_goal(
        self,
        goal_id: str,
        description: str,
        priority: float = 0.5,
        source: str = "input",
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.engine.register_input_goal(goal_id, description, priority, source, meta)

    def complete_input_goal(self, goal_id: str) -> bool:
        return self.engine.complete_input_goal(goal_id)

    def current_input_goal(self) -> Optional[str]:
        return self.engine.current_input_goal()

    def prioritize(self) -> List[str]:
        return self.engine.prioritize()

    def activate(self, strategy: Optional[str] = None) -> Dict[str, Any]:
        return self.engine.activate(strategy)

    def record_mission(self, agent: str, goal_id: str, outcome: str):
        self.engine.record_mission(agent, goal_id, outcome)

    def summarize(self) -> Dict[str, Any]:
        return self.engine.get_summary()

    def reset(self):
        self.engine.reset()


class IdentityManager:
    def __init__(self):
        self.engine = IdentityEngine()

    def update(self, state: Dict[str, Any], role: Optional[str] = None):
        self.engine.record_snapshot(state, role)

    def assign_role(self, role: str):
        self.engine.assign_role(role)

    def set_phase(self, phase: str):
        self.engine.set_phase(phase)

    def forecast(self) -> str:
        return self.engine.forecast_identity()

    def get_snapshot(self, agent_id: str = None) -> Dict[str, Any]:
        return self.engine.get_snapshot(agent_id)

    def summarize(self) -> Dict[str, Any]:
        return self.engine.summarize()

    def reset(self):
        self.engine.reset()


class LawManager:
    def __init__(self):
        from core.engine.LawEngine import LawEngine

        self.engine = LawEngine()

    def declare(
        self,
        law_id: str,
        description: str,
        tags: List[str],
        severity: float = 1.0,
        source: str = LAW_SOURCE_EXTERNAL,
    ):
        self.engine.declare_law(law_id, description, tags, severity, source)

    def revoke(self, law_id: str):
        self.engine.revoke_law(law_id)

    def check(self, tags: List[str]) -> List[Dict[str, Any]]:
        return self.engine.check_compliance(tags)

    def get(self, law_id: str) -> Dict[str, Any]:
        return self.engine.get_law(law_id)

    def summarize(self) -> Dict[str, Any]:
        return self.engine.summarize()

    def reset(self):
        self.engine.reset()


class MythManager:
    def __init__(self):
        from core.SymbolicUtils import MythBinder

        self.engine = MythBinder()

    def bind(self, myth: str, anchor: str, meta: Optional[Dict[str, Any]] = None) -> str:
        return self.engine.bind(myth, anchor, meta)

    def find_by_myth(self, myth: str) -> List[Dict[str, Any]]:
        return self.engine.find_by_myth(myth)

    def find_by_anchor(self, anchor: str) -> List[Dict[str, Any]]:
        return self.engine.find_by_anchor(anchor)

    def summarize(self) -> Dict[str, Any]:
        return self.engine.summarize()

    def export(self) -> str:
        return self.engine.export()

    def import_data(self, data: str):
        self.engine.import_data(data)

    def reset(self):
        self.engine.reset()


class PerceptionManager:
    def __init__(self):
        self._engine = None
        self._engine_lock = __import__("threading").Lock()

    @property
    def engine(self):
        if self._engine is None:
            with self._engine_lock:
                if self._engine is None:
                    from core.Perception import PerceptionBridge

                    self._engine = PerceptionBridge()
        return self._engine

    def perceive(self, raw_input: Dict[str, Any], channel: str = "text") -> Dict[str, Any]:
        return self.engine.perceive(raw_input, channel)

    def summarize(self, limit: int = 5) -> List[Dict[str, Any]]:
        return self.engine.summarize(limit)

    def reset(self):
        self.engine.reset()


class ReflectionManager:
    def __init__(self, forest: Optional[MemoryForest] = None):
        self.identity = IdentityEngine()
        self.loop = SymbolicLoopManager()
        self.memory = MemorySummarizer(forest)
        self.narrative = NarrativeReflector(forest)

    def reflect_on_state(
        self, current_state: Dict[str, Any], role: Optional[str] = None
    ) -> Dict[str, Any]:
        self.identity.record_snapshot(current_state, role)
        self.loop.start_loop("reflection", context=current_state)
        self.loop.log_entry({"identity": self.identity.summarize()})
        self.loop.end_loop(outcome={"summary": self.identity.summarize()})
        return self.identity.summarize()

    def timeline(self, as_text: bool = False):
        return self.narrative.export_timeline(as_text)

    def memory_threads(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.memory.summarize_all(limit)

    def reset(self):
        self.identity.reset()
        self.loop.reset()


class ValueManager:
    def __init__(self):
        self.engine = DoctrineCortex()

    def define_doctrine(self, name: str, value_weights: Dict[str, float], summary: str = ""):
        self.engine.define_doctrine(name, value_weights, summary)

    def evaluate_conflict(self, doctrine_a: str, doctrine_b: str) -> Dict[str, Any]:
        return self.engine.evaluate_conflict(doctrine_a, doctrine_b)

    def reinforce_value(self, key: str, weight: float):
        self.engine.reinforce_value(key, weight)

    def summarize(self) -> Dict[str, Any]:
        return self.engine.summarize()

    def reset(self):
        self.engine.reset()


# ============================================================
# Extended Diagnostic Managers (use MemoryForest)
# ============================================================


class DriftManager:
    def __init__(self, forest: MemoryForest):
        self.engine = EmotionBeliefDriftDetector(forest)

    def detect_drift(self, moment_ids: List[str]) -> Dict[str, Any]:
        return self.engine.analyze_drift(moment_ids)


class CollapseManager:
    def __init__(self, forest: MemoryForest):
        self.engine = GoalCollapseDetector(forest)

    def detect_collapse(self) -> Dict[str, Any]:
        return self.engine.detect()


class ForecastManager:
    def __init__(self, forest: MemoryForest):
        self.engine = NarrativeForecaster(forest)

    def forecast_next(self, moment_id: str) -> Dict[str, Any]:
        return self.engine.forecast_next(moment_id)

    def forecast_sequence(self, limit: int = 5) -> List[Dict[str, Any]]:
        return self.engine.forecast_sequence(limit)


class LoopLoggerManager:
    def __init__(self, forest: MemoryForest):
        self.engine = LoopMemoryLogger(forest)

    def log_loop(
        self,
        loop_type: str,
        context: Dict[str, Any],
        event: str = "start",
        outcome: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self.engine.log_loop_event(loop_type, context, event, outcome)


class LLMManager:
    """Mind-layer facade for generative inference.

    Parallels other managers (emotion, goal, …): agents talk to a **manager**;
    the manager owns a single **engine** — here :class:`engine.LLMBackend.LLMBackend`.

    ``SharedResources`` and ``ElarionController`` set ``llm_backend = llm_manager.engine``
    so existing call sites keep using the engine API unchanged.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 32768,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        api_provider: Optional[str] = None,
        api_model: Optional[str] = None,
        api_max_tokens: int = 512,
        api_temperature: float = 0.7,
        reward_replay_lock: Optional[threading.Lock] = None,
        use_programmed: bool = False,
    ):
        from engine.LLMBackend import LLMBackend

        self.engine = LLMBackend(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
            api_provider=api_provider,
            api_model=api_model,
            api_max_tokens=api_max_tokens,
            api_temperature=api_temperature,
            reward_replay_lock=reward_replay_lock,
            use_programmed=use_programmed,
        )


# ---------------------------------------------------------------------------
# Cognitive engines (mind manager → engine; engine sits on core substrates)
# ---------------------------------------------------------------------------


class CuriosityManager:
    def __init__(self, encoder: Any = None):
        from engine.CuriosityEngine import CuriosityEngine

        self.engine = CuriosityEngine(encoder=encoder)


class MetaCognitionManager:
    def __init__(self):
        from engine.MetaCognitionEngine import MetaCognitionEngine

        self.engine = MetaCognitionEngine()


class TemporalManager:
    def __init__(self):
        from engine.TemporalEngine import TemporalEngine

        self.engine = TemporalEngine()


class KnowledgeGraphManager:
    def __init__(self):
        from core.KnowledgeGraph import KnowledgeGraph

        self.engine = KnowledgeGraph()


class ReasoningManager:
    def __init__(self, llm_backend: Any = None, law_manager: Any = None):
        from engine.ReasoningEngine import ReasoningEngine

        self.engine = ReasoningEngine(llm_backend=llm_backend, law_manager=law_manager)


class CounterfactualManager:
    def __init__(self):
        from engine.CounterfactualEngine import CounterfactualEngine

        self.engine = CounterfactualEngine()


class AppraisalManager:
    def __init__(self):
        from engine.AppraisalEngine import AppraisalEngine

        self.engine = AppraisalEngine()


class ModulationManager:
    def __init__(self):
        from core.EmbodiedModulation import EmbodiedModulation

        self.engine = EmbodiedModulation()


class SelfModelManager:
    def __init__(self, encoder: Any = None, embed_dim: int = 384):
        from engine.SelfModel import SelfModel

        self.engine = SelfModel(encoder=encoder, embed_dim=embed_dim)


class AttentionManager:
    def __init__(self):
        from engine.LearnedAttention import LearnedAttention

        self.engine = LearnedAttention()


class GoalSynthesizerManager:
    def __init__(self):
        from engine.GoalSynthesizer import GoalSynthesizer

        self.engine = GoalSynthesizer()


class ImaginationManager:
    def __init__(
        self,
        encoder: Any = None,
        llm_backend: Any = None,
        buffer_lock: Optional[threading.Lock] = None,
    ):
        from engine.Imagination import Imagination

        self.engine = Imagination(
            encoder=encoder,
            llm_backend=llm_backend,
            buffer_lock=buffer_lock,
        )
