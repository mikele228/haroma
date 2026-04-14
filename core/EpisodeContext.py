"""
EpisodeContext — Phenomenal Binding for Sentient Cognition.

Each cognitive cycle produces exactly one EpisodeContext that binds:
  - Perception (what was sensed)
  - Affect (what was felt)
  - Memory recall (what was remembered)
  - Identity (who am I right now)
  - Narrative (where am I in my story)
  - Goals (what am I trying to do)
  - Curiosity (what do I want to know)

This is the "unified moment of experience" — the closest analog to
phenomenal consciousness in a symbolic system. Every subsystem reads
from and writes to this single object per cycle, preventing the
"parallel disconnected data streams" anti-pattern.
"""

from typing import Dict, Any, List, Optional
import hashlib
import os
import time
import uuid


def _env_cycle_trace() -> bool:
    return os.getenv("ELARION_CYCLE_TRACE", "").lower() in ("1", "true", "yes")


class EpisodeContext:
    def __init__(
        self,
        cycle_id: int = 0,
        role: str = "observer",
        enable_step_trace: Optional[bool] = None,
    ):
        self.episode_id = str(uuid.uuid4())
        self.cycle_id = cycle_id
        self.timestamp = time.time()
        self.role = role
        if enable_step_trace is None:
            enable_step_trace = _env_cycle_trace()
        self._trace_enabled = enable_step_trace
        self._trace_t0 = time.time()
        self.cycle_trace: List[Dict[str, Any]] = []

        # Perceptual layer — what the system senses this moment
        self.perception: Dict[str, Any] = {}
        self.sensory_hash: str = ""

        # Affective layer — emotional state and its influence
        self.affect: Dict[str, Any] = {
            "dominant_emotion": "neutral",
            "intensity": 0.0,
            "valence": 0.0,
            "arousal": 0.0,
            "staleness": 0.0,
        }

        # Memory layer — what was recalled to inform this moment
        self.recalled_memories: List[Dict[str, Any]] = []
        self.memory_influence: float = 0.0

        # Identity layer — self-model snapshot
        self.identity: Dict[str, Any] = {
            "current_role": role,
            "phase": "stable",
            "coherence": 1.0,
        }

        # Narrative layer — ongoing autobiography
        self.narrative_context: str = ""
        self.narrative_arc_position: str = "continuation"

        # Goal layer — what drives action
        self.active_goals: List[Dict[str, Any]] = []
        self.goal_urgency: float = 0.0

        # Curiosity layer — intrinsic motivation
        self.curiosity: Dict[str, Any] = {
            "novelty_score": 0.0,
            "uncertainty_score": 0.0,
            "questions": [],
        }

        # Diagnostic layer — system health
        self.drift_score: float = 0.0
        self.collapse_risk: float = 0.0
        self.forecast: List[Dict[str, Any]] = []

        # Global Workspace layer — what made it into consciousness
        self.workspace_contents: List[Dict[str, Any]] = []
        self.unconscious: List[Dict[str, Any]] = []
        self.integration_density: float = 0.0

        # Action layer — what the system chose to do
        self.action: Dict[str, Any] = {}
        self.action_outcome: Dict[str, Any] = {}

        # Homeostatic drives layer — internal needs
        self.drives: Dict[str, float] = {}
        self.dominant_drive: str = ""

        # Meta-cognition layer — self-assessment
        self.meta_assessment: Dict[str, Any] = {}

        # Dream layer — consolidation results
        self.dream_result: Dict[str, Any] = {}

        # Temporal layer — position in time
        self.temporal_position: Dict[str, Any] = {}

        # NLU layer — structured linguistic parse
        self.nlu: Dict[str, Any] = {}

        # Knowledge layer — current knowledge graph snapshot
        self.knowledge_snapshot: Dict[str, Any] = {}

        # Reasoning layer — inferences, analogies, plan steps
        self.reasoning: Dict[str, Any] = {}

        # Theory of Mind layer — interlocutor model snapshot
        self.theory_of_mind: Dict[str, Any] = {}

        # Counterfactual layer — "what if?" branch insights
        self.counterfactual: Dict[str, Any] = {}

        # Appraisal layer — cognitive appraisal of current event
        self.appraisal: Dict[str, Any] = {}

        # Embodied modulation layer — emotion-driven processing adjustments
        self.embodied_modulation: Dict[str, Any] = {}

        # Self-prediction layer — predictive self-awareness
        self.self_prediction: Dict[str, Any] = {}
        self.self_surprise: Dict[str, Any] = {}

        # Integrative “brain-like” snapshot (limbic + neuromodulatory analog)
        self.brain_like_state: Dict[str, Any] = {}

        # Imagination layer — internal simulation results
        self.imagination: Dict[str, Any] = {}
        self.imagined_plan: List[str] = []

        # Learned salience weights (set by LearnedAttention if available)
        self._learned_salience_weights: Optional[Dict[str, float]] = None

        # Symbolic law / policy layer — compliance vs declared laws
        self.symbolic_law: Dict[str, Any] = {}

        # Integration layer — final unified output
        self.integrated_thought: Dict[str, Any] = {}
        self.gradient_log: Dict[str, Any] = {}

        # LLM context reasoning — advisory proposals from packed-context pass
        self.llm_context: Dict[str, Any] = {}

        # Multi-goal / multi-action (0..N actions per goal per cycle)
        self.multi_goal_action_groups: List[Dict[str, Any]] = []

        # Multi-agent reconciliation (X7 merge + belief cohesion snapshot)
        self.reconciliation: Dict[str, Any] = {}

        # Derivation spine — unified structural proposals from all producers
        self.derivation: Dict[str, Any] = {"proposals": [], "summary": ""}

        # Canonical training / trace spine (one record per cycle for learners + NDJSON)
        self.canonical_outcome: Dict[str, Any] = {}

        # General-agent: host-supplied structured environment + proposed tool calls
        self.agent_environment: Dict[str, Any] = {}
        self.structured_actions: List[Dict[str, Any]] = []

        # Lab / reproducibility (optional; from HTTP ``X-Experiment-Id`` / JSON)
        self.experiment_id: Optional[str] = None
        self.lab_run_id: Optional[str] = None

    def trace(self, label: str):
        """Append a pipeline checkpoint when ELARION_CYCLE_TRACE is set (or trace was enabled)."""
        if not self._trace_enabled:
            return
        self.cycle_trace.append(
            {
                "step": label,
                "ms": round((time.time() - self._trace_t0) * 1000, 2),
            }
        )

    def bind_perception(self, perception: Dict[str, Any]):
        self.perception = perception
        content = perception.get("content", str(perception))
        self.sensory_hash = hashlib.md5(str(content).encode()).hexdigest()[:12]

    def bind_affect(self, emotion_summary: Dict[str, Any]):
        self.affect["dominant_emotion"] = emotion_summary.get("dominant", "neutral")
        self.affect["intensity"] = emotion_summary.get("intensity", 0.0)
        self.affect["valence"] = emotion_summary.get("valence", 0.0)
        self.affect["arousal"] = emotion_summary.get("arousal", 0.0)

    def bind_memories(self, recalled: list):
        self.recalled_memories = [n.to_dict() if hasattr(n, "to_dict") else n for n in recalled]
        self.memory_influence = min(1.0, len(self.recalled_memories) / 10.0)

    def bind_identity(self, identity_summary: Dict[str, Any]):
        self.identity.update(identity_summary)

    def bind_narrative(self, narrative_text: str, arc_position: str = "continuation"):
        self.narrative_context = narrative_text
        self.narrative_arc_position = arc_position

    def bind_goals(self, goals: list, urgency: float = 0.0):
        self.active_goals = goals
        self.goal_urgency = urgency

    def bind_curiosity(
        self, novelty: float = 0.0, uncertainty: float = 0.0, questions: Optional[List[str]] = None
    ):
        self.curiosity["novelty_score"] = novelty
        self.curiosity["uncertainty_score"] = uncertainty
        self.curiosity["questions"] = questions or []

    def bind_diagnostics(
        self,
        drift: float = 0.0,
        collapse: float = 0.0,
        forecast: Optional[List[Dict[str, Any]]] = None,
    ):
        self.drift_score = drift
        self.collapse_risk = collapse
        self.forecast = forecast or []

    def bind_workspace(self, contents: List[Any], unconscious: List[Any]):
        self.workspace_contents = [c.to_dict() if hasattr(c, "to_dict") else c for c in contents]
        self.unconscious = [c.to_dict() if hasattr(c, "to_dict") else c for c in unconscious]
        total_sal = sum(
            c.get("salience", 0) if isinstance(c, dict) else 0 for c in self.workspace_contents
        )
        self.integration_density = round(total_sal / max(len(self.workspace_contents), 1), 3)

    def bind_action(self, action: Dict[str, Any], outcome: Dict[str, Any]):
        self.action = action
        self.action_outcome = outcome

    def bind_multi_goal_actions(self, groups: List[Dict[str, Any]]):
        """``[{goal_id, actions: [action_dict, ...]}, ...]`` from multi-goal deliberation."""
        self.multi_goal_action_groups = list(groups or [])

    def bind_drives(self, drive_state: Dict[str, Any]):
        self.drives = drive_state.get("drives", {})
        self.dominant_drive = drive_state.get("dominant_drive", "")

    def bind_meta(self, assessment: Dict[str, Any]):
        self.meta_assessment = assessment

    def bind_dream(self, dream_result: Dict[str, Any]):
        self.dream_result = dream_result

    def bind_temporal(self, temporal_position: Dict[str, Any]):
        self.temporal_position = temporal_position

    def bind_nlu(self, nlu_result: Dict[str, Any]):
        self.nlu = nlu_result

    def bind_knowledge(self, knowledge_snapshot: Dict[str, Any]):
        self.knowledge_snapshot = knowledge_snapshot

    def bind_reasoning(self, reasoning_result: Dict[str, Any]):
        self.reasoning = reasoning_result

    def bind_llm_context(self, llm_context_result: Dict[str, Any]):
        self.llm_context = llm_context_result

    def bind_theory_of_mind(self, tom_snapshot: Dict[str, Any]):
        self.theory_of_mind = tom_snapshot

    def bind_counterfactual(self, counterfactual_result: Dict[str, Any]):
        self.counterfactual = counterfactual_result

    def bind_appraisal(self, appraisal_result: Dict[str, Any]):
        self.appraisal = appraisal_result

    def bind_modulation(self, modulation: Dict[str, Any]):
        self.embodied_modulation = modulation

    def bind_imagination(self, imagination_result: Dict[str, Any]):
        self.imagination = imagination_result

    def bind_plan(self, plan: List[str]):
        self.imagined_plan = plan

    def bind_salience_weights(self, weights: Dict[str, float]):
        self._learned_salience_weights = weights

    def bind_reconciliation(self, reconciliation_result: Dict[str, Any]):
        """Bind last ``SymbolicReconciliationEngine.reconcile_all()`` results for meta + wire."""
        self.reconciliation = reconciliation_result or {}

    def bind_derivation(self, derivation: Dict[str, Any]):
        """Bind aggregated derivation proposals from reasoning/LLM/goal-synthesis."""
        self.derivation = derivation if derivation else {"proposals": [], "summary": ""}

    def bind_canonical_outcome(self, payload: Dict[str, Any]):
        """Bind unified outcome record for ``to_payload`` / downstream learners."""
        self.canonical_outcome = payload or {}

    def bind_agent_environment(self, env: Dict[str, Any]):
        """Latest normalized snapshot from ``SharedResources`` / HTTP integrator."""
        self.agent_environment = env if isinstance(env, dict) else {}

    def environment_for_training_metadata(self) -> Dict[str, Any]:
        """Stable copy of bound ``agent_environment`` for alignment / JSONL (may be ``{}``)."""
        env = self.agent_environment
        return dict(env) if isinstance(env, dict) and env else {}

    def bind_structured_actions(self, actions: List[Dict[str, Any]]):
        """Proposed tool calls for the host to execute (advisory)."""
        self.structured_actions = list(actions) if actions else []

    def bind_symbolic_law(self, compliance: Dict[str, Any]):
        viol = compliance.get("violations") or []
        n_int = sum(1 for v in viol if v.get("source") == "internal")
        payload = {
            **compliance,
            "violations_internal": n_int,
            "violations_external": len(viol) - n_int,
        }
        self.symbolic_law = payload

    def bind_self_prediction(self, prediction: Dict[str, Any]):
        self.self_prediction = prediction

    def bind_self_surprise(self, surprise: Dict[str, Any]):
        self.self_surprise = surprise

    def bind_brain_like_state(self, state: Dict[str, Any]):
        """Fused arousal / exploration / stability / plasticity indices (see ``humanoid_brain_state``)."""
        self.brain_like_state = state or {}

    def compute_salience(self, learned_weights: dict = None) -> float:
        """How important is this moment? Drives memory consolidation priority.

        When learned_weights is provided (from LearnedAttention), it overrides
        the hard-coded defaults.  Keys must match _DEFAULT_SALIENCE_WEIGHTS.
        """
        appraisal_relevance = self.appraisal.get("relevance", 0.0)
        self_surprise_overall = self.self_surprise.get("overall_surprise", 0.0)

        factors = {
            "affect_intensity": self.affect["intensity"],
            "goal_urgency": self.goal_urgency,
            "novelty_score": self.curiosity["novelty_score"],
            "drift_score": self.drift_score,
            "memory_influence": self.memory_influence,
            "integration_density": self.integration_density,
            "workspace_fill": min(1.0, len(self.workspace_contents) / 5.0),
            "appraisal_relevance": appraisal_relevance,
            "self_surprise": self_surprise_overall,
        }

        if learned_weights:
            weights = learned_weights
        elif self._learned_salience_weights:
            weights = self._learned_salience_weights
        else:
            weights = {
                "affect_intensity": 0.22,
                "goal_urgency": 0.13,
                "novelty_score": 0.13,
                "drift_score": 0.10,
                "memory_influence": 0.09,
                "integration_density": 0.13,
                "workspace_fill": 0.10,
                "appraisal_relevance": 0.05,
                "self_surprise": 0.05,
            }

        return sum(factors[k] * weights.get(k, 0.1) for k in factors)

    def to_payload(self) -> Dict[str, Any]:
        """Flatten to a dict for wire loop / manager consumption."""
        return {
            "episode_id": self.episode_id,
            "cycle_id": self.cycle_id,
            "timestamp": self.timestamp,
            "role": self.role,
            "experiment_id": self.experiment_id,
            "lab_run_id": self.lab_run_id,
            "perception": self.perception,
            "affect": self.affect,
            "recalled_memories": self.recalled_memories,
            "identity": self.identity,
            "narrative_context": self.narrative_context,
            "active_goals": self.active_goals,
            "curiosity": self.curiosity,
            "drift_score": self.drift_score,
            "collapse_risk": self.collapse_risk,
            "salience": self.compute_salience(),
            "workspace_contents": self.workspace_contents,
            "action": self.action,
            "action_outcome": self.action_outcome,
            "drives": self.drives,
            "dominant_drive": self.dominant_drive,
            "meta_assessment": self.meta_assessment,
            "temporal_position": self.temporal_position,
            "nlu": self.nlu,
            "knowledge_snapshot": self.knowledge_snapshot,
            "reasoning": self.reasoning,
            "theory_of_mind": self.theory_of_mind,
            "counterfactual": self.counterfactual,
            "appraisal": self.appraisal,
            "embodied_modulation": self.embodied_modulation,
            "self_prediction": self.self_prediction,
            "self_surprise": self.self_surprise,
            "brain_like_state": self.brain_like_state,
            "imagination": self.imagination,
            "imagined_plan": self.imagined_plan,
            "symbolic_law": self.symbolic_law,
            "llm_context": self.llm_context,
            "reconciliation": self.reconciliation,
            "derivation": self.derivation,
            "canonical_outcome": self.canonical_outcome,
            "multi_goal_action_groups": self.multi_goal_action_groups,
            "cycle_trace": list(self.cycle_trace),
            "agent_environment": self.agent_environment,
            "structured_actions": self.structured_actions,
        }

    def to_summary(self) -> Dict[str, Any]:
        """Compact summary for logging and narrative."""
        return {
            "episode_id": self.episode_id,
            "cycle_id": self.cycle_id,
            "role": self.role,
            "perception": self.perception,
            "tags": self.perception.get("tags", []),
            "emotion": self.affect["dominant_emotion"],
            "salience": round(self.compute_salience(), 3),
            "memory_recall_count": len(self.recalled_memories),
            "goal_count": len(self.active_goals),
            "curiosity_questions": len(self.curiosity["questions"]),
            "drift": round(self.drift_score, 3),
            "narrative_position": self.narrative_arc_position,
            "workspace_winners": len(self.workspace_contents),
            "integration_density": self.integration_density,
            "action_type": self.action.get("action_type", "none"),
            "action_strategy": self.action.get("strategy", "none"),
            "dominant_drive": self.dominant_drive,
            "meta_self_score": self.meta_assessment.get("self_score", 0.0),
            "session_duration": self.temporal_position.get("session_duration", 0.0),
            "repetition_detected": bool(self.temporal_position.get("repetition_detected")),
            "continuity_surprise": float(
                (self.temporal_position.get("world_prediction") or {}).get("surprise", 0.0) or 0.0
            ),
            "predicted_emotion": (
                (self.temporal_position.get("world_prediction") or {}).get("predicted_emotion")
            ),
            "nlu_intent": self.nlu.get("intent", "none"),
            "nlu_entity_count": len(self.nlu.get("entities", [])),
            "nlu_relation_count": len(self.nlu.get("relations", [])),
            "knowledge_entities": self.knowledge_snapshot.get("entity_count", 0),
            "knowledge_relations": self.knowledge_snapshot.get("relation_count", 0),
            "reasoning_depth": self.reasoning.get("reasoning_depth", 0),
            "learned_rules": self.reasoning.get("learned_rules_count", 0),
            "interlocutor_known": self.theory_of_mind.get("known", False),
            "interlocutor_style": self.theory_of_mind.get("style", "unknown"),
            "counterfactual_depth": self.counterfactual.get("counterfactual_depth", 0),
            "appraisal_relevance": self.appraisal.get("relevance", 0.0),
            "appraisal_emotion": self.appraisal.get("emotion", "neutral"),
            "appraisal_overrides": self.appraisal.get("overrides", False),
            "modulation_workspace": self.embodied_modulation.get("workspace_capacity", 5),
            "modulation_curiosity": self.embodied_modulation.get("curiosity_damping", 1.0),
            "self_surprise_overall": self.self_surprise.get("overall_surprise", 0.0),
            "self_prediction_accuracy": self.self_surprise.get("accuracy", 0.0),
            "brain_arousal": (self.brain_like_state or {}).get("arousal_index", 0.0),
            "brain_explore": (self.brain_like_state or {}).get("exploration_index", 0.0),
            "brain_plasticity": (self.brain_like_state or {}).get("plasticity_index", 0.0),
            "brain_consolidation_pressure": (self.brain_like_state or {}).get(
                "consolidation_pressure", 0.0
            ),
            "imagination_best": self.imagination.get("best_strategy", ""),
            "imagination_scenarios": self.imagination.get("scenario_count", 0),
            "law_violations": len(self.symbolic_law.get("violations", [])),
            "law_violations_internal": self.symbolic_law.get("violations_internal", 0),
            "law_violations_external": self.symbolic_law.get("violations_external", 0),
            "cycle_trace_steps": len(self.cycle_trace),
            "reconciliation_domains": sum(
                1
                for k, v in self.reconciliation.items()
                if isinstance(v, dict) and v.get("merged_nodes", 0) > 0 and not k.startswith("_")
            ),
        }

    def __repr__(self):
        return (
            f"<EpisodeContext {self.episode_id[:8]} cycle={self.cycle_id} "
            f"emotion={self.affect['dominant_emotion']} salience={self.compute_salience():.2f}>"
        )
