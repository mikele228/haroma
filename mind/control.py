"""
Unified control runner for Elarion — sentient cognitive loop with
phenomenal binding, global workspace attention, learned emotion,
semantic memory, prediction-error curiosity, deliberative action,
narrative self, meta-cognition, homeostatic drives, dream
consolidation, persistence, soul binding, working memory,
conversation tracking, context-aware dialogue, temporal binding,
NLU understanding, relational knowledge graph, symbolic reasoning,
theory-of-mind interlocutor modelling, counterfactual reasoning,
KG-based curiosity, self-modifying rule learning,
appraisal-based emotion, embodied modulation,
self-grown neural embeddings,
predictive self-awareness,
learned language composition,
adaptive self-tuning (learned attention, appraisal, modulation,
drives, growing emotion lexicon),
imagination (internal simulation), learned goal synthesis,
metacognitive learning, dynamic processing allocation,
generative language, and learned counterfactual depth.

Multi-agent deployments register background ``train_step`` modules in
:mod:`core.training_surface`; this controller still owns hot-path training order here.
"""

from typing import Dict, Any, List, Optional
import threading
import time

from mind.manager import (
    IdentityManager,
    GoalManager,
    EmotionManagerSimple as EmotionManager,
    DriftManager,
    CollapseManager,
    ForecastManager,
    LoopLoggerManager,
    ReflectionManager,
    DreamManager,
    LawManager,
    MythManager,
    ValueManager,
    FusionManager,
    PerceptionManager,
    LLMManager,
    CuriosityManager,
    MetaCognitionManager,
    TemporalManager,
    KnowledgeGraphManager,
    ReasoningManager,
    CounterfactualManager,
    AppraisalManager,
    ModulationManager,
    SelfModelManager,
    AttentionManager,
    GoalSynthesizerManager,
    ImaginationManager,
)
from core.Memory import MemoryForest, MemoryNode
from core.chat_recall_policy import (
    is_teaching_turn,
    merge_web_learn_tail,
    should_merge_web_learn,
    web_learn_inject_max,
)
from core.EpisodeContext import EpisodeContext
from core.GlobalWorkspace import GlobalWorkspace
from core.ActionLoop import ActionGenerator, OutcomeEvaluator, ActionMemory
from core.Persistence import CognitivePersistence
from core.SoulBinder import SoulBinder
from core.DreamConsolidator import DreamConsolidator
from core.HomeostaticDrives import HomeostaticSystem
from core.WorkingMemory import WorkingMemory
from core.ConversationTracker import ConversationTracker
from core.InterlocutorModel import InterlocutorModel
from engine.NeuralEncoder import NeuralEncoder
from engine.LanguageComposer import LanguageComposer
from engine.ProcessGate import ProcessGate
from engine.CognitiveBackbone import CognitiveBackbone, build_snapshot
from core.EmbodiedModulation import EmbodiedModulation
from core.DiscourseProcessor import DiscourseProcessor
from environment.EnvironmentGrounder import EnvironmentGrounder
from environment.ActionDispatcher import ActionDispatcher
from engine.MentalSimulator import MentalSimulator
from engine.ArchitectureSearcher import ArchitectureSearcher
from engine.ComputeFabric import ComputeFabric
from engine.TrainingScheduler import TrainingScheduler
from engine.ResourceAdaptiveConfig import detect_resources, ResourceConfig
from core.MemoryCore import MemoryCore
from core.Reconciliation import SymbolicReconciliationEngine
from core.derivation_merge import merge_derivation_artifacts
from core.SymbolicQueue import SymbolicQueue, SymbolicFingerprintEngine
from core.OrganRegistry import OrganRegistry
from mind.cycle_flow import (
    build_action_episode_payload,
    build_counterfactual_gate_features,
    build_law_tags,
    resolve_strategy_hint,
    run_counterfactual_phase,
    run_curiosity_phase,
    run_deliberative_action,
    run_multi_goal_deliberative_actions,
    run_goal_synthesis_phase,
    run_imagination_phase,
    run_law_value_myth_sidecar_phase,
    run_metacognition_phase,
    run_reasoning_phase,
    workspace_contents_as_dicts,
    write_processor_symbolic_queue,
)
from mind.deliberative_cycle_env import read_multi_goal_deliberative_env
from mind.haroma_settings import (
    haroma_cmem_merge_prime,
    haroma_cmem_recall_fallback_forest,
    haroma_cmem_recall_max_probe,
    haroma_memory_recall_intensity,
    haroma_recall_cmem_only,
)
from mind.packed_llm_controller_bridge import (
    controller_packed_llm_enabled,
    run_packed_llm_phase_for_elarion_controller,
)
from mind.cognitive_trace import (
    CognitiveTraceRecorder,
    apply_ablation_overrides,
    build_canonical_outcome,
    build_planner_arbitration,
    reconciliation_ablated,
)
from mind.environment_contract import (
    merge_world_into_observation,
    normalize_environment_observation,
)
from core.concurrency import ConcurrencyCoordinator
from core.cognitive_null import CognitiveNull, is_cognitive_null


class ElarionController:
    """Single-process cognitive loop (embedded / library use).

    By default this class **does not** call :func:`~mind.cycle_flow.run_llm_context_reasoning_phase`.
    Set ``HAROMA_CONTROLLER_PACKED_LLM=1`` to run the same 13.2b packed-context LLM step as
    :class:`~agents.persona_agent.PersonaAgent` after symbolic reasoning (see
    :mod:`mind.packed_llm_controller_bridge`).
    (packed-context ``generate_chat``). That path lives on
    :class:`~agents.persona_agent.PersonaAgent` for multi-agent chat. See
    :mod:`mind.cognitive_entrypoints`.
    """

    def __init__(self):
        # ── Resource detection: auto-scale everything to available hardware ──
        self.resource_config: ResourceConfig = detect_resources()
        _rc = self.resource_config
        _skip = set(_rc.cycle.get("skip_modules", []))
        _enc_cfg = _rc.encoder
        _llm_cfg = _rc.llm
        _train_cfg = _rc.training

        print(
            f"[Elarion] Resource tier: {_rc.tier} ({_rc.tier_name}) | "
            f"RAM={_rc.hardware['ram_gb']}GB | "
            f"GPU={_rc.hardware['gpu'].get('name') or 'none'} | "
            f"APIs={[k for k, v in _rc.hardware['api_keys'].items() if v]}",
            flush=True,
        )

        # Phase 16 compute fabric (must be first — all modules register on it)
        self.fabric = ComputeFabric.init()
        self.locks = ConcurrencyCoordinator()

        # Phase 8 neural grounding — tier-aware encoder
        self.encoder = NeuralEncoder(
            embed_dim=_enc_cfg.get("embed_dim", 256),
            vocab_size=_enc_cfg.get("vocab_size", 8192),
            force_mode=_enc_cfg.get("mode"),
        )

        self.memory = MemoryForest(encoder=self.encoder)
        self.cycle_count = 0

        # Primary cognitive managers (always loaded — lightweight)
        self.identity = IdentityManager()
        self.goal = GoalManager()
        self.emotion = EmotionManager()

        # Reflective diagnostics
        self.drift = DriftManager(self.memory)
        self.collapse = CollapseManager(self.memory)
        self.forecast = ForecastManager(self.memory)
        self.loop = LoopLoggerManager(self.memory)

        # Extended cognition layers
        self.dream = DreamManager()
        self.law = LawManager()
        self.myth = MythManager()
        self.value = ValueManager()
        self.fusion = FusionManager()
        self.perception = PerceptionManager()

        self.reflector = ReflectionManager(self.memory)

        # Generative language: manager (mind) → LLMBackend engine
        self.llm_manager = LLMManager(
            model_path=_llm_cfg.get("local_gguf"),
            n_ctx=_llm_cfg.get("n_ctx", 2048),
            n_gpu_layers=_llm_cfg.get("n_gpu_layers", 0),
            api_provider=_llm_cfg.get("api_provider"),
            api_model=_llm_cfg.get("api_model"),
            api_max_tokens=_llm_cfg.get("api_max_tokens", 512),
            api_temperature=_llm_cfg.get("api_temperature", 0.7),
            reward_replay_lock=self.locks.training.reward_replay,
        )
        self.llm_backend = self.llm_manager.engine

        # Phase 10 learned language composition
        self.composer = LanguageComposer(
            encoder=self.encoder,
            llm_backend=self.llm_backend,
            data_lock=self.locks.training.language_composer_data,
        )

        # Phase 2 sentience mechanisms
        self.curiosity_manager = CuriosityManager(encoder=self.encoder)
        self.curiosity = self.curiosity_manager.engine
        self.workspace = GlobalWorkspace(capacity=5)
        self.action_generator = ActionGenerator(composer=self.composer)
        self.outcome_evaluator = OutcomeEvaluator()
        self.action_memory = ActionMemory()

        # Phase 3 sentience mechanisms
        self.metacognition_manager = MetaCognitionManager()
        self.metacognition = self.metacognition_manager.engine
        self.drives = HomeostaticSystem()
        self.dream_consolidator = (
            DreamConsolidator(self.memory, encoder=self.encoder)
            if "dream_consolidator" not in _skip
            else CognitiveNull()
        )
        self.persistence = CognitivePersistence()
        self._save_in_progress = False
        self._save_dispatch_lock = threading.Lock()
        self._construction_meta: Dict[str, Any] = {}

        # Phase 4 sentience mechanisms
        self.working_memory = WorkingMemory(capacity=12)
        self.conversation = ConversationTracker()
        self.temporal_manager = TemporalManager()
        self.temporal = self.temporal_manager.engine

        # Phase 5 understanding and reasoning mechanisms
        self.knowledge_manager = KnowledgeGraphManager()
        self.knowledge = self.knowledge_manager.engine
        self.reasoning_manager = ReasoningManager(
            llm_backend=self.llm_backend, law_manager=self.law
        )
        self.reasoning = self.reasoning_manager.engine

        # Phase 6 wiring and self-modification mechanisms
        self.interlocutor_model = InterlocutorModel()
        if "counterfactual" not in _skip:
            self.counterfactual_manager = CounterfactualManager()
            self.counterfactual = self.counterfactual_manager.engine
        else:
            self.counterfactual_manager = CognitiveNull()
            self.counterfactual = CognitiveNull()

        # Phase 7 appraisal-based emotion and embodied modulation
        self.appraisal_manager = AppraisalManager()
        self.appraisal = self.appraisal_manager.engine
        self.modulation_manager = ModulationManager()
        self.modulation = self.modulation_manager.engine
        self._prev_modulation = EmbodiedModulation.defaults()

        # Phase 9 predictive self-model
        self.self_model_manager = SelfModelManager(
            encoder=self.encoder, embed_dim=self.encoder._embed_dim
        )
        self.self_model = self.self_model_manager.engine
        self._prev_self_surprise: Dict[str, Any] = {}
        self._prev_cycle_state: Dict[str, Any] = {}

        # Phase 11 adaptive self-tuning
        self.attention_manager = AttentionManager()
        self.attention = self.attention_manager.engine
        self._prev_outcome_score: float = 0.5

        # Phase 12 imagination, goal synthesis, metacognitive learning
        if "imagination" not in _skip:
            self.imagination_manager = ImaginationManager(
                encoder=self.encoder,
                llm_backend=self.llm_backend,
                buffer_lock=self.locks.training.imagination_buffer,
            )
            self.imagination = self.imagination_manager.engine
        else:
            self.imagination_manager = CognitiveNull()
            self.imagination = CognitiveNull()
        self.goal_synthesizer_manager = GoalSynthesizerManager()
        self.goal_synthesizer = self.goal_synthesizer_manager.engine

        # Phase 13 dynamic processing allocation
        self.process_gate = ProcessGate(pending_lock=self.locks.training.process_gate_pending)

        # Phase 15 unified cognitive backbone
        self.backbone = CognitiveBackbone()
        self._prev_backbone_snapshot: Optional[list] = None

        # Phase 15 compositional NLU
        self.discourse = DiscourseProcessor()

        # Phase 15 grounded environment learning
        self.grounder = EnvironmentGrounder()
        self.action_dispatcher = ActionDispatcher()

        # Phase 15 theory of mind — heavy, skip on embedded/edge
        self.mental_simulator = (
            MentalSimulator() if "mental_simulator" not in _skip else CognitiveNull()
        )

        self._environment_context: Dict[str, Any] = {}
        self._environment_contract: Dict[str, Any] = {}
        self._last_embodied_observation: Optional[Dict[str, Any]] = None

        # Phase 15 architecture search — skip on low tiers
        self.arch_searcher = (
            ArchitectureSearcher() if "arch_searcher" not in _skip else CognitiveNull()
        )

        # Phase 16 training scheduler — disabled on embedded
        self.training_scheduler = TrainingScheduler()
        if _train_cfg.get("enabled", True):
            self.training_scheduler.register_module("encoder", base_interval=3)
            self.training_scheduler.register_module("backbone", base_interval=3)
            self.training_scheduler.register_module("attention", base_interval=5)
            self.training_scheduler.register_module("process_gate", base_interval=5)
            self.training_scheduler.register_module("self_model", base_interval=5)
            self.training_scheduler.register_module("appraisal", base_interval=5)
            self.training_scheduler.register_module("modulation", base_interval=8)
            self.training_scheduler.register_module("goal_synth", base_interval=8)
            self.training_scheduler.register_module("imagination", base_interval=8)
            self.training_scheduler.register_module("metacog", base_interval=5)
            self.training_scheduler.register_module("composer", base_interval=5)
            self.training_scheduler.register_module("generative", base_interval=5)
            self.training_scheduler.register_module("counterfactual", base_interval=8)
            self.training_scheduler.register_module("grounder", base_interval=10)
            self.training_scheduler.register_module("mental_sim", base_interval=5)
            self.training_scheduler.register_module("arch_search", base_interval=10)
            self.training_scheduler.register_module("llm_reward", base_interval=5)

        # Phase 14 plan tracking
        self._current_plan: List[str] = []
        self._plan_step: int = 0
        self._plan_outcomes: List[float] = []

        # Narrative self — rolling autobiography
        self._narrative_buffer: List[str] = []
        self._max_narrative_length = 50

        # Previous episode snapshot (for outcome evaluation)
        self._prev_episode_payload: Dict[str, Any] = {}
        self._prev_imagination_embedding = None

        # Autonomy diagnostics (read by AutonomyLoopRunner)
        self.last_emotion_staleness: float = 0.0
        self.last_coherence_drift: float = 0.0

        # Boot sequence: soul FIRST (foundation), then persistence, then
        # soul reasserts immutable identity so it can never be erased.
        self._boot_results: Dict[str, Any] = {}
        self._soul_binder = SoulBinder()
        self._boot_results["soul"] = self._soul_binder.bind(self)
        self._boot_results["persistence"] = self.persistence.load(self)
        self._boot_results["soul_reassert"] = self._soul_binder.reassert(self)
        self._boot_results["fabric"] = self.fabric.stats()

        # X7 multi-agent reconciliation layer
        self.memory_core = MemoryCore(self.memory)
        self.reconciliation = SymbolicReconciliationEngine(self.memory_core)

        # X7 queue integrity + drift control layer
        self.symbolic_queue = SymbolicQueue()
        self.fingerprint_engine = SymbolicFingerprintEngine()

        # X7 15-organ module registry
        self.organ_registry = OrganRegistry()
        self._register_organs()

    def _register_organs(self):
        """Register live module references into the organ registry.
        Skipped modules (None) are excluded — the registry only tracks
        what is actually running on this hardware tier."""
        _map = {
            "llm_manager": self.llm_manager,
            "knowledge": self.knowledge,
            "knowledge_manager": self.knowledge_manager,
            "reasoning": self.reasoning,
            "reasoning_manager": self.reasoning_manager,
            "encoder": self.encoder,
            "emotion": self.emotion,
            "value": self.value,
            "law": self.law,
            "myth": self.myth,
            "drift": self.drift,
            "collapse": self.collapse,
            "forecast": self.forecast,
            "appraisal": self.appraisal,
            "appraisal_manager": self.appraisal_manager,
            "modulation": self.modulation,
            "modulation_manager": self.modulation_manager,
            "metacognition": self.metacognition,
            "metacognition_manager": self.metacognition_manager,
            "curiosity": self.curiosity,
            "curiosity_manager": self.curiosity_manager,
            "attention": self.attention,
            "attention_manager": self.attention_manager,
            "process_gate": self.process_gate,
            "backbone": self.backbone,
            "workspace": self.workspace,
            "training_scheduler": self.training_scheduler,
            "arch_searcher": self.arch_searcher,
            "soul_binder": self._soul_binder,
            "identity": self.identity,
            "interlocutor_model": self.interlocutor_model,
            "mental_simulator": self.mental_simulator,
            "memory": self.memory,
            "memory_core": self.memory_core,
            "working_memory": self.working_memory,
            "persistence": self.persistence,
            "loop_logger": self.loop,
            "dream_consolidator": self.dream_consolidator,
            "dream": self.dream,
            "imagination": self.imagination,
            "imagination_manager": self.imagination_manager,
            "goal": self.goal,
            "drives": self.drives,
            "goal_synthesizer": self.goal_synthesizer,
            "goal_synthesizer_manager": self.goal_synthesizer_manager,
            "fusion": self.fusion,
            "counterfactual": self.counterfactual,
            "counterfactual_manager": self.counterfactual_manager,
            "reconciliation": self.reconciliation,
            "perception": self.perception,
            "discourse": self.discourse,
            "temporal": self.temporal,
            "temporal_manager": self.temporal_manager,
            "self_model": self.self_model,
            "self_model_manager": self.self_model_manager,
            "grounder": self.grounder,
            "reflector": self.reflector,
            "symbolic_loop": self.reflector.loop,
            "memory_summarizer": self.reflector.memory,
            "narrative": self.reflector.narrative,
            "symbolic_queue": self.symbolic_queue,
            "fingerprint_engine": self.fingerprint_engine,
            "conversation": self.conversation,
            "composer": self.composer,
            "action_generator": self.action_generator,
            "fabric": self.fabric,
        }
        for name, ref in _map.items():
            if ref is not None and not is_cognitive_null(ref):
                self.organ_registry.register_module(name, ref)

    def update_environment_context(self, env_observation: Dict[str, Any]):
        """Update what the controller knows about the physical environment.

        Called by the LifeLoop before each cycle so the action generator
        can propose physical actions when opportunities exist.
        """
        contract = normalize_environment_observation(env_observation)
        self._environment_contract = contract
        env_observation = merge_world_into_observation(contract, dict(env_observation))

        exits = env_observation.get("exits", [])
        objects = []
        interactive = []
        agents = []
        for name in env_observation.get("objects", []):
            objects.append(name)
        for name in env_observation.get("interactive_objects", []):
            interactive.append(name)
        for name in env_observation.get("agents", []):
            agents.append(name)

        raw_objects = env_observation.get("objects_detail", [])
        if isinstance(raw_objects, list):
            for o in raw_objects:
                if isinstance(o, dict):
                    nm = o.get("name", "")
                    if nm and nm not in objects:
                        objects.append(nm)
                    if o.get("interactions") and nm and nm not in interactive:
                        interactive.append(nm)

        self._environment_context = {
            "exits": exits,
            "objects": objects,
            "agents": agents,
            "interactive_objects": interactive,
            "location": env_observation.get("location", ""),
            "location_name": env_observation.get("location_name", ""),
        }

    # ==================================================================
    # Sentient Cognitive Cycle  (~23 steps)
    # ==================================================================

    def run_cycle(self, input_state: dict, role: str = "observer") -> Dict[str, Any]:
        """One phenomenal episode: perceive → bind → deliberate → act → learn.

        Ordered step *names* and the full pipeline list live in
        ``mind.cycle_flow.COGNITIVE_PIPELINE_STEPS``. Enable
        ``ELARION_CYCLE_TRACE`` (or ``EpisodeContext(..., enable_step_trace=True)``)
        for timing checkpoints in ``episode.cycle_trace``.
        """
        _t0 = time.time()
        self.cycle_count += 1
        has_external = role != "idle" and input_state.get("content", "") != "idle state"
        episode = EpisodeContext(cycle_id=self.cycle_count, role=role)
        _planner_arbitration: Dict[str, Any] = {}
        episode.trace("0.start")
        self.workspace.clear()

        # ── 0. WORKING MEMORY TICK ───────────────────────────────────
        self.working_memory.tick(self.cycle_count)

        # ── 1. PERCEIVE + RECORD CONVERSATION ───────────────────────
        symbolic_input = self.perception.perceive(input_state, channel="text")
        content = symbolic_input.get("content", "")
        nlu_result = symbolic_input.get("nlu", {})
        from mind.nlu_enrich import enrich_nlu_for_kg

        nlu_result = enrich_nlu_for_kg(nlu_result, content)
        symbolic_input["nlu"] = nlu_result
        episode.bind_perception(symbolic_input)

        node = self.memory.create_node_from(symbolic_input)
        self.memory.add_node("encounter_tree", role, node)

        if has_external and content:
            self.working_memory.add(
                content=content[:80],
                source="perception",
                salience=0.7,
                item_type="percept",
                tags=symbolic_input.get("tags", []),
                cycle=self.cycle_count,
            )

        # ── 1.4. NEURAL EMBEDDING ─────────────────────────────────
        current_embedding = self.encoder.encode(content) if content else None

        # ── 1.45. PROCESS GATE (decide which optional steps run) ──
        import math as _math

        _prev_emo = self._prev_episode_payload.get("affect", {})
        _gate_features = [
            _math.log1p(self.cycle_count),
            1.0 if has_external else 0.0,
            _prev_emo.get("intensity", 0.0),
            _prev_emo.get("arousal", 0.0),
            len(self.goal.engine.goals) / 10.0,
            self.curiosity.summarize().get("curiosity_score", 0.0),
            max((d.level for d in self.drives.drives), default=0.0),
            self.working_memory.occupancy(),
            self.knowledge.stats().get("entity_count", 0) / 100.0,
            self._prev_outcome_score,
            self._prev_episode_payload.get("reasoning", {}).get("reasoning_depth", 0) / 10.0,
            self._prev_episode_payload.get("counterfactual", {}).get("counterfactual_depth", 0)
            / 3.0,
            self._prev_episode_payload.get("imagination", {}).get("quality", 0.0),
            self._prev_self_surprise.get("overall_surprise", 0.0),
            1.0 if self.conversation.detect_topic_shift(symbolic_input.get("tags", [])) else 0.0,
            1.0 if self.conversation.is_in_conversation(self.cycle_count) else 0.0,
            min(10.0, (time.time() - self._prev_episode_payload.get("_timestamp", time.time())))
            / 10.0,
            self.metacognition._history[-1].get("predicted_outcome", 0.5)
            if self.metacognition._history
            else 0.5,
            _math.sqrt(
                sum(
                    x * x
                    for x in (list(current_embedding) if current_embedding is not None else [0.0])
                )
            )
            / max(self.encoder._embed_dim, 1),
            self._prev_episode_payload.get("_steps_run_ratio", 1.0),
        ]
        # ── 1.46. COGNITIVE BACKBONE — build snapshot + z_t ───────
        _bb_snapshot = build_snapshot(
            content_embedding=list(current_embedding) if current_embedding is not None else None,
            embed_dim=self.encoder._embed_dim,
            valence=_prev_emo.get("valence", 0.0),
            arousal=_prev_emo.get("arousal", 0.0),
            intensity=_prev_emo.get("intensity", 0.0),
            curiosity_score=self.curiosity.summarize().get("curiosity_score", 0.0),
            prediction_error=self._prev_episode_payload.get("curiosity", {}).get(
                "prediction_error", 0.0
            ),
            dominant_drive_level=max((d.level for d in self.drives.drives), default=0.0),
            wm_load=self.working_memory.occupancy(),
            outcome_prev=self._prev_outcome_score,
            has_external=1.0 if has_external else 0.0,
            cycle_count=self.cycle_count,
            n_goals=len(self.goal.engine.goals),
            kg_entity_count=self.knowledge.stats().get("entity_count", 0),
            self_surprise=self._prev_self_surprise.get("overall_surprise", 0.0),
            emotion_streak=0,
            drift_score=self._prev_episode_payload.get("drift_score", 0.0),
            strategy=self._prev_episode_payload.get("action", {}).get("strategy", "reflect")
            if isinstance(self._prev_episode_payload.get("action"), dict)
            else "reflect",
            intent=nlu_result.get("intent", "statement") if nlu_result else "statement",
            drive_levels=[d.level for d in self.drives.drives][:5],
            reasoning_depth=self._prev_episode_payload.get("reasoning", {}).get(
                "reasoning_depth", 0
            ),
            cf_depth=self._prev_episode_payload.get("counterfactual", {}).get(
                "counterfactual_depth", 0
            ),
            imagination_quality=self._prev_episode_payload.get("imagination", {}).get(
                "quality", 0.0
            ),
            metacog_prediction=self.metacognition._history[-1].get("predicted_outcome", 0.5)
            if self.metacognition._history
            else 0.5,
            steps_run_ratio=self._prev_episode_payload.get("_steps_run_ratio", 1.0),
            plan_active=bool(self._current_plan),
            env_tick=0,
        )
        _ce_list = list(current_embedding) if current_embedding is not None else None
        _z_t = self.backbone.encode_state(_bb_snapshot, content_embedding=_ce_list)

        _conv_skip = self.process_gate._CONVERSANT_SKIP if role == "conversant" else None
        _gate_decisions = self.process_gate.decide(_gate_features, z_t=_z_t, force_off=_conv_skip)
        _gate_decisions = apply_ablation_overrides(_gate_decisions)

        # ── 1.5. SELF-PREDICTION (before main processing) ────────
        self_prediction = {}
        if _gate_decisions.get("self_prediction", True):
            self_prediction = (
                self.self_model.predict(current_embedding, self._prev_cycle_state) or {}
            )
        if self_prediction:
            episode.bind_self_prediction(self_prediction)

        # ── 1.6. NLU BIND ───────────────────────────────────────────
        episode.bind_nlu(nlu_result)

        # ── 1.65. DISCOURSE PROCESSING (compositional NLU) ─────────
        _discourse_result = None
        _discourse_frames_dicts: List[Dict[str, Any]] = []
        if nlu_result:
            _conv_history = [t.to_dict() for t in self.conversation.get_recent(5)]
            _discourse_result = self.discourse.process(
                nlu_result, conversation_history=_conv_history, cycle_id=self.cycle_count
            )
            _discourse_frames_dicts = [f.to_dict() for f in _discourse_result.frames]

        # ── 1.7. INTERLOCUTOR MODEL + MENTAL SIMULATOR ────────────────
        interlocutor_snapshot = {}
        _mental_prediction: Dict[str, Any] = {}
        if _gate_decisions.get("interlocutor_model", True):
            if has_external and content:
                self.interlocutor_model.update(
                    speaker=role,
                    content=content,
                    nlu_result=nlu_result,
                    cycle_id=self.cycle_count,
                    discourse_frames=_discourse_frames_dicts if _discourse_frames_dicts else None,
                )
                observed_action = nlu_result.get("intent", "speak") if nlu_result else "speak"
                _action_map = {
                    "declarative": "speak",
                    "utterance": "speak",
                    "statement": "speak",
                    "interrogative": "speak",
                    "question": "speak",
                    "imperative": "speak",
                    "exclamatory": "emote",
                }
                observed_mapped = _action_map.get(observed_action, "speak")
                self.mental_simulator.update_model(
                    agent_id=role,
                    observed_action=observed_mapped,
                    discourse_frames=_discourse_frames_dicts if _discourse_frames_dicts else None,
                    context={
                        "content": content,
                        "sentiment": nlu_result.get("sentiment", {}) if nlu_result else {},
                    },
                    cycle_id=self.cycle_count,
                )
            interlocutor_snapshot = self.interlocutor_model.get_model_summary(role)
            _mental_prediction = self.mental_simulator.predict_behavior(role)
            interlocutor_snapshot["mental_prediction"] = _mental_prediction
            episode.bind_theory_of_mind(interlocutor_snapshot)

        # ── 1.9. QUEUE: generator outputs (X7 integrity layer) ────────
        if symbolic_input:
            self.symbolic_queue.write(
                SymbolicQueue.SLOT_GENERATOR,
                "generator.perception",
                "symbolic_input",
                symbolic_input,
            )
        if nlu_result:
            self.symbolic_queue.write(
                SymbolicQueue.SLOT_GENERATOR, "generator.nlu", "nlu_result", nlu_result
            )

        # ── 2. SEMANTIC RECALL ───────────────────────────────────────
        query_tags = symbolic_input.get("tags", [])
        if isinstance(input_state.get("tags"), list):
            query_tags = query_tags + input_state["tags"]
        query_text = content or input_state.get("content", "")
        _teaching = is_teaching_turn(query_text, nlu_result or {})
        recall_limit = self._prev_modulation.get("recall_limit", 20)
        if _teaching:
            recall_limit = min(recall_limit + 10, 28)
        _recall_intensity = haroma_memory_recall_intensity()
        if 0 < _recall_intensity < 10:
            recall_limit = max(1, (recall_limit * _recall_intensity + 9) // 10)
        _t_recall_start = time.time()
        if _recall_intensity == 0:
            recalled = []
        else:
            _cmem_only = haroma_recall_cmem_only()
            _cmem_kw = {}
            if _cmem_only:
                _cmem_kw = {
                    "tree_names": frozenset({"cmem"}),
                    "tree_recall_max_probe": haroma_cmem_recall_max_probe(),
                }
            _prime_prepend = (not _cmem_only) or haroma_cmem_merge_prime()
            recalled = self.memory.recall(
                query_tags=query_tags,
                limit=recall_limit,
                query_text=query_text,
                **_cmem_kw,
            )
            if _cmem_only and not recalled and haroma_cmem_recall_fallback_forest():
                recalled = self.memory.recall(
                    query_tags=query_tags,
                    limit=recall_limit,
                    query_text=query_text,
                )
            recalled = self.memory.merge_recall_with_prime(
                recalled, recall_limit, fast_cycle=False, prepend_prime=_prime_prepend
            )
            if has_external and should_merge_web_learn(
                query_text, nlu_result or {}, teaching=_teaching
            ):
                _wlim = web_learn_inject_max()
                if _wlim > 0:
                    recalled = merge_web_learn_tail(
                        self.memory, recalled, recall_limit, max_inject=_wlim
                    )
        _t_recall = time.time() - _t_recall_start
        episode.bind_memories(recalled)

        # ── 2.5. KNOWLEDGE GRAPH INTEGRATION ─────────────────────────
        if _gate_decisions.get("kg_integration", True):
            if nlu_result and (nlu_result.get("entities") or nlu_result.get("relations")):
                self.knowledge.integrate(nlu_result, cycle_id=self.cycle_count)
        knowledge_summary = self.knowledge.summary()
        knowledge_diff = self.knowledge.diff()
        episode.bind_knowledge(knowledge_summary)

        # ── 3. FEEL (learned + keyword dual system) ──────────────────
        emotion_input = {**input_state, **symbolic_input}
        self.emotion.ingest(emotion_input)
        emotion_summary = self.emotion.summarize()
        episode.bind_affect(emotion_summary)

        if has_external and content:
            self.conversation.record_input(
                content=content,
                speaker=role,
                cycle_id=self.cycle_count,
                emotion=episode.affect["dominant_emotion"],
                tags=symbolic_input.get("tags", []),
            )
            if nlu_result and _discourse_result is not None:
                self.conversation.store_discourse_snapshot(
                    self.cycle_count, _discourse_result.to_dict()
                )

        # ── 4. GLOBAL WORKSPACE — broadcast (learned salience) ──────
        attention_ctx = self.attention._build_context(
            valence=emotion_summary.get("valence", 0.0),
            arousal=emotion_summary.get("arousal", 0.0),
            curiosity=self.curiosity.summarize().get("curiosity_score", 0.0),
            wm_load=self.working_memory.occupancy(),
            dominant_drive_level=max((d.level for d in self.drives.drives), default=0.0),
            outcome_prev=self._prev_outcome_score,
            cycle_count=self.cycle_count,
            has_external=float(has_external),
        )

        n_modalities = len(symbolic_input.get("modalities", ["text"]))
        base_perception_sal = (
            0.5 + (0.3 if symbolic_input.get("tags") else 0.0) + (0.1 * min(n_modalities - 1, 3))
        )
        self.workspace.broadcast(
            "perception",
            symbolic_input,
            salience=self.attention.adjust_salience(
                "perception", base_perception_sal, attention_ctx, z_t=_z_t
            ),
        )
        self.workspace.broadcast(
            "memory",
            {"recalled": episode.recalled_memories},
            salience=self.attention.adjust_salience(
                "memory", episode.memory_influence, attention_ctx, z_t=_z_t
            ),
        )
        self.workspace.broadcast(
            "emotion",
            emotion_summary,
            salience=self.attention.adjust_salience(
                "emotion", emotion_summary.get("intensity", 0.0), attention_ctx, z_t=_z_t
            ),
        )

        if knowledge_diff.get("changed"):
            base_kg_sal = min(1.0, knowledge_diff.get("knowledge_gain", 0.0) + 0.3)
            self.workspace.broadcast(
                "knowledge",
                {
                    "new_entities": knowledge_diff.get("new_entities", 0),
                    "new_relations": knowledge_diff.get("new_relations", 0),
                    "knowledge_gain": knowledge_diff.get("knowledge_gain", 0.0),
                },
                salience=self.attention.adjust_salience(
                    "knowledge", base_kg_sal, attention_ctx, z_t=_z_t
                ),
            )

        salience_weights = self.attention.get_salience_weights()
        episode.bind_salience_weights(salience_weights)

        # ── 5. GLOBAL WORKSPACE — select + promote to WM ────────────
        ws_contents = self.workspace.select()
        episode.bind_workspace(ws_contents, self.workspace.get_unconscious())
        self.workspace.integrate()

        self.working_memory.promote_from_workspace(ws_contents, cycle=self.cycle_count)
        episode.trace("5.workspace")

        # ── 6. NARRATE SELF + TEMPORAL ARC ───────────────────────────
        temporal_arc = self.temporal.summarize_arc(n=15)
        narrative_context = self._get_narrative_context()
        if temporal_arc and len(self.temporal.episode_timeline) > 3:
            narrative_context = f"{narrative_context} {temporal_arc}"
        episode.bind_narrative(narrative_context)

        # ── 7. ENFORCE SYMBOLIC LAW + VALUE / MYTH / FUSION ──────────
        law_tags = build_law_tags(symbolic_input, input_state)
        run_law_value_myth_sidecar_phase(
            gate_enabled=_gate_decisions.get("law_value_myth_fusion", True),
            law=self.law,
            law_tags=law_tags,
            episode=episode,
            workspace=self.workspace,
            attention=self.attention,
            attention_ctx=attention_ctx,
            z_t=_z_t,
            value=self.value,
            myth=self.myth,
            fusion=self.fusion,
            dream_mgr=self.dream,
            symbolic_input=symbolic_input,
            explicit=input_state,
            role=role,
            has_external=bool(has_external),
            broadcast_violations=True,
            apply_explicit_bindings=True,
            skip_derived_value_myth_fusion=False,
            trace_label="7.law_sidecar",
        )

        # ── 8. DREAM CONSOLIDATION ───────────────────────────────────
        dream_result: Dict[str, Any] = {}
        if _gate_decisions.get("dream_consolidation", True):
            if self.drives.should_dream() or (not has_external and self.cycle_count % 5 == 0):
                dream_result = self.dream_consolidator.consolidate(
                    emotion_summary=emotion_summary,
                    controller=self,
                )
                episode.bind_dream(dream_result)
                dream_narr = dream_result.get("dream_narrative", "")
                if dream_narr:
                    self._narrative_buffer.append(dream_narr)
            else:
                dream_seed = role
                if episode.affect["dominant_emotion"] != "neutral":
                    dream_seed = f"{role}_{episode.affect['dominant_emotion']}"
                self.dream.generate(seed=dream_seed)

        # ── 8.5. RECONCILIATION (X7 multi-agent merge) ───────────────
        _reconciliation_result: Dict[str, Any] = {}
        if _gate_decisions.get("reconciliation", True) and not reconciliation_ablated():
            if self.reconciliation.has_agent_branches() and (
                self.cycle_count % 10 == 0 or self.cycle_count < 3
            ):
                try:
                    _reconciliation_result = self.reconciliation.reconcile_all()
                except Exception as _reco_exc:
                    print(f"[Control] reconcile_all error: {_reco_exc}", flush=True)
        episode.bind_reconciliation(_reconciliation_result)
        if _reconciliation_result:
            try:
                from core.Reconciliation import materialize_reconciliation_experience

                _reco_narr = materialize_reconciliation_experience(
                    self.memory,
                    role,
                    _reconciliation_result,
                    cycle=self.cycle_count,
                )
                if _reco_narr:
                    self._narrative_buffer.append(_reco_narr)
            except Exception as _mat_exc:
                print(f"[Control] materialize_reconciliation error: {_mat_exc}", flush=True)

        # ── 9. IDENTIFY ──────────────────────────────────────────────
        if _gate_decisions.get("identity_update", True):
            self.identity.update(input_state, role)
        identity_summary = self.identity.summarize()
        episode.bind_identity(identity_summary)

        # ── 10. SET GOALS (emotion + workspace bias priority) ────────
        if episode.affect["intensity"] > 0.5:
            self.goal.register_goal(
                f"emotional_{self.cycle_count}",
                f"Process {episode.affect['dominant_emotion']} "
                f"with intensity {episode.affect['intensity']:.2f}",
                priority=episode.affect["intensity"],
                source="emotion",
            )
        priorities = self.goal.prioritize_workfront()
        active_goals = []
        for i, gid in enumerate(priorities[:5]):
            g = self.goal.engine.goals.get(gid, {}) if hasattr(self.goal, "engine") else {}
            row = {
                "goal_id": gid,
                "priority": g.get("priority", 1.0 - i * 0.1),
                "description": g.get("description", ""),
            }
            cg = g.get("child_goal_ids")
            if cg:
                row["child_goal_ids"] = cg
            ai = g.get("action_items")
            if ai:
                row["action_items"] = ai
            active_goals.append(row)
        episode.bind_goals(active_goals, urgency=len(priorities) / 10.0)

        self.workspace.broadcast(
            "goals",
            {"active_goals": active_goals},
            salience=self.attention.adjust_salience(
                "goals", min(1.0, len(priorities) / 5.0), attention_ctx, z_t=_z_t
            ),
        )

        # ── 10.5. HOMEOSTATIC DRIVES ─────────────────────────────────
        drive_state = self.drives.update(
            episode.to_payload(),
            self._prev_episode_payload.get("action_outcome", {}),
            is_dream_cycle=bool(dream_result),
            has_external_input=has_external,
        )
        episode.bind_drives(drive_state)

        drive_goals = self.drives.bias_goals(drive_state)
        for dg in drive_goals:
            self.goal.register_goal(
                dg["goal_id"],
                dg["description"],
                priority=dg["priority"],
                source=dg["source"],
            )

        if drive_goals:
            priorities = self.goal.prioritize_workfront()
            active_goals = []
            for i, gid in enumerate(priorities[:5]):
                g = self.goal.engine.goals.get(gid, {}) if hasattr(self.goal, "engine") else {}
                row = {
                    "goal_id": gid,
                    "priority": i,
                    "description": g.get("description", ""),
                }
                if g.get("child_goal_ids"):
                    row["child_goal_ids"] = g["child_goal_ids"]
                if g.get("action_items"):
                    row["action_items"] = g["action_items"]
                active_goals.append(row)
            episode.bind_goals(active_goals, urgency=len(priorities) / 10.0)

        # ── 10.7. COGNITIVE APPRAISAL (goal-aware emotion refinement) ─
        prev_drift = self._prev_episode_payload.get("drift_score", 0.0)
        appraisal_result = self.appraisal.evaluate(
            nlu_result=nlu_result,
            active_goals=active_goals,
            knowledge_summary=knowledge_summary,
            knowledge_diff=knowledge_diff,
            identity_summary=identity_summary,
            emotion_summary=emotion_summary,
            drift_score=prev_drift,
            action_memory_stats=self.action_memory.stats(),
            working_memory_load=self.working_memory.occupancy(),
            interlocutor=interlocutor_snapshot,
        )
        episode.bind_appraisal(appraisal_result)

        if appraisal_result.get("overrides"):
            self.emotion.engine.update_emotion(
                appraisal_result["emotion"], appraisal_result["intensity"], emotion_input
            )
            emotion_summary = self.emotion.summarize()
            episode.bind_affect(emotion_summary)

        # ── 10.8. EMBODIED MODULATION (emotion reshapes processing) ───
        modulation = self.modulation.compute(emotion_summary, z_t=_z_t)
        episode.bind_modulation(modulation)
        self.workspace.capacity = modulation["workspace_capacity"]

        # ── 11. REFLECT + 12. DIAGNOSE ───────────────────────────────
        drift_score = 0.0
        drift_result = {}
        collapse_result = {}
        forecast_result = []
        if _gate_decisions.get("reflection_diagnose", True):
            summary = self.reflector.reflect_on_state(input_state, role)
            self.loop.log_loop("cognitive_cycle", context=summary, event="start")

            recent_ids = self._collect_recent_moment_ids(limit=10)
            drift_result = self.drift.detect_drift(recent_ids)
            collapse_result = self.collapse.detect_collapse()
            forecast_result = self.forecast.forecast_sequence(limit=3)

            drift_score = (
                drift_result.get("drift_score", 0.0) if isinstance(drift_result, dict) else 0.0
            )
            collapse_goals = (
                collapse_result.get("collapsed_goals", [])
                if isinstance(collapse_result, dict)
                else []
            )
            episode.bind_diagnostics(
                drift=drift_score,
                collapse=(
                    len(collapse_goals) / max(1, collapse_result.get("total_goals", 1))
                    if isinstance(collapse_result, dict)
                    else 0.0
                ),
                forecast=forecast_result if isinstance(forecast_result, list) else [],
            )

        # ── 13. CURIOSITY (KG + tag blended prediction error) ────────
        _prev_strategy = (
            self._prev_episode_payload.get("action", {}).get("strategy", "reflect")
            if isinstance(self._prev_episode_payload.get("action"), dict)
            else "reflect"
        )
        _forecast_for_curiosity = (
            {"forecasts": forecast_result} if isinstance(forecast_result, list) else forecast_result
        )
        curiosity_result = run_curiosity_phase(
            enabled=_gate_decisions.get("curiosity", True),
            episode=episode,
            curiosity=self.curiosity,
            emotion_summary=emotion_summary,
            knowledge_summary=knowledge_summary,
            current_embedding=current_embedding,
            last_strategy=_prev_strategy,
            forecast_for_eval=_forecast_for_curiosity,
            knowledge_graph=self.knowledge,
            goal_manager=self.goal,
            workspace=self.workspace,
            attention=self.attention,
            attention_ctx=attention_ctx,
            z_t=_z_t,
            modulation=modulation,
            workspace_followup=True,
        )

        # ── 13.2. REASONING (inference + analogy + planning) ─────────
        reasoning_result = run_reasoning_phase(
            enabled=_gate_decisions.get("reasoning", True),
            reasoning_engine=self.reasoning,
            knowledge=self.knowledge,
            active_goals=active_goals,
            nlu_result=nlu_result,
            max_depth=modulation.get("inference_cap"),
            memory=self.memory,
            episode=episode,
            law=self.law,
            law_tags=law_tags,
            workspace=self.workspace,
            attention=self.attention,
            attention_ctx=attention_ctx,
            z_t=_z_t,
            gate_law_value_myth=_gate_decisions.get("law_value_myth_fusion", True),
            gate_reasoning_for_refresh=_gate_decisions.get("reasoning", True),
            trace_label="13.2.reasoning_law",
        )

        # ── 13.2b. PACKED-CONTEXT LLM (opt-in; same phase as PersonaAgent) ─
        if controller_packed_llm_enabled():
            run_packed_llm_phase_for_elarion_controller(
                self,
                episode,
                role=role,
                content=content,
                text=content,
                has_external=bool(has_external),
                gate_reasoning=_gate_decisions.get("reasoning", True),
                reasoning_result=reasoning_result,
                appraisal_result=appraisal_result,
                active_goals=active_goals,
                identity_summary=identity_summary,
                nlu_result=nlu_result,
                knowledge_summary=knowledge_summary,
                cycle_id=self.cycle_count,
            )

        # ── 13.3. COUNTERFACTUAL REASONING (with learned gate) ───────
        counterfactual_result: Dict[str, Any] = {"counterfactual_depth": 0, "branches": []}
        _cf_features: List[float] = []
        if _gate_decisions.get("counterfactual", True):
            _cf_features = build_counterfactual_gate_features(
                knowledge_diff=knowledge_diff,
                reasoning_result=reasoning_result,
                active_goals=active_goals,
                emotion_summary=emotion_summary,
                curiosity_result=curiosity_result,
                has_external=bool(has_external),
                cycle_count=self.cycle_count,
                counterfactual_engine=self.counterfactual,
                prev_modulation=self._prev_modulation,
            )
            counterfactual_result = run_counterfactual_phase(
                enabled=True,
                counterfactual_engine=self.counterfactual,
                knowledge_graph=self.knowledge,
                reasoning_engine=self.reasoning,
                reasoning_result=reasoning_result,
                knowledge_diff=knowledge_diff,
                active_goals=active_goals,
                nlu_result=nlu_result,
                episode=episode,
                gate_features=_cf_features,
                workspace=self.workspace,
                attention=self.attention,
                attention_ctx=attention_ctx,
                z_t=_z_t,
            )

        # ── 13.5. META-COGNITION + TEMPORAL REPETITION ───────────────
        meta_assessment, _self_inspection = run_metacognition_phase(
            enabled=_gate_decisions.get("metacognition", True),
            extended=True,
            metacognition=self.metacognition,
            episode=episode,
            emotion_summary=emotion_summary,
            curiosity_result=curiosity_result,
            outcome_prev=self._prev_episode_payload.get("action_outcome", {}),
            cycle_count=self.cycle_count,
            prev_meta_assessment=self._prev_episode_payload.get("meta_assessment", {}),
            self_surprise=self._prev_self_surprise,
            controller_for_inspection=self,
            temporal_engine=self.temporal,
            goal_manager=self.goal,
            workspace=self.workspace,
            attention=self.attention,
            attention_ctx=attention_ctx,
            z_t=_z_t,
        )

        # ── TEMPORAL BIND ────────────────────────────────────────────
        temporal_pos: Dict[str, Any] = {}
        if _gate_decisions.get("temporal_bind", True):
            temporal_pos = self.temporal.get_temporal_position(
                self.cycle_count,
                emotion_summary.get("dominant", "neutral"),
            )
            episode.bind_temporal(temporal_pos)

        # ── 13.7. IMAGINATION (internal simulation) ──────────────────
        imagination_result, imagined_strategy, imagined_plan = run_imagination_phase(
            enabled=_gate_decisions.get("imagination", True),
            imagination=self.imagination,
            episode=episode,
            current_embedding=current_embedding,
            emotion_summary=emotion_summary,
            curiosity_result=curiosity_result,
            dominant_drive=max((d.level for d in self.drives.drives), default=0.0),
            wm_load=self.working_memory.occupancy(),
            outcome_prev=self._prev_outcome_score,
            has_external=float(has_external),
            active_goals=active_goals,
            drive_state=drive_state,
            temporal_engine=self.temporal,
            temporal_enrichment=True,
        )
        if imagined_plan:
            episode.bind_plan(imagined_plan)
            if not self._current_plan:
                self._current_plan = list(imagined_plan)
                self._plan_step = 0

        # ── 13.8. GOAL SYNTHESIS (learned goal discovery) ─────────────
        synthesized_goals, synth_ctx = run_goal_synthesis_phase(
            enabled=_gate_decisions.get("goal_synthesis", True),
            goal_synthesizer=self.goal_synthesizer,
            goal_manager=self.goal,
            valence=emotion_summary.get("valence", 0.0),
            arousal=emotion_summary.get("arousal", 0.0),
            curiosity_score=curiosity_result.get("curiosity_score", 0.0),
            prediction_error=curiosity_result.get("prediction_error", 0.0),
            dominant_drive_level=max((d.level for d in self.drives.drives), default=0.0),
            wm_load=self.working_memory.occupancy(),
            drift_score=drift_score,
            outcome_prev=self._prev_outcome_score,
            has_external=float(has_external),
            knowledge_entity_count=knowledge_summary.get("entity_count", 0),
            goal_count=len(active_goals),
            cycle_count=self.cycle_count,
            z_t=_z_t,
            active_goals=active_goals,
        )

        # ── 13.9. QUEUE: processor outputs + fingerprint (X7) ─────────
        write_processor_symbolic_queue(
            self.symbolic_queue,
            self.fingerprint_engine,
            emotion_summary,
            reasoning_result,
            meta_assessment,
        )

        # ── 14. ACT (deliberative action selection) ──────────────────
        ctx_hash = ActionMemory._hash_context(episode.to_payload())
        _mem_strategy_hint = self.action_memory.suggest_strategy(ctx_hash)
        strategy_hint = resolve_strategy_hint(
            _mem_strategy_hint,
            imagined_strategy,
            self._current_plan,
            self._plan_step,
        )
        _planner_arbitration = build_planner_arbitration(
            memory_hint=_mem_strategy_hint,
            imagined_strategy=imagined_strategy,
            current_plan=self._current_plan,
            plan_step=self._plan_step,
            resolved_hint=strategy_hint,
        )

        ws_dicts = workspace_contents_as_dicts(self.workspace)

        is_conv = self.conversation.is_in_conversation(self.cycle_count)
        last_turn = self.conversation.get_last_input()
        last_input_content = last_turn.content if last_turn else ""
        topic = self.conversation.get_topic()
        topic_shifted = self.conversation.detect_topic_shift(symbolic_input.get("tags", []))

        merge_derivation_artifacts(episode, synthesized_goals=synthesized_goals)

        ep_payload, _kg_triples = build_action_episode_payload(
            episode=episode,
            current_embedding=current_embedding,
            z_t=_z_t,
            knowledge_graph=self.knowledge,
            knowledge_summary=knowledge_summary,
            nlu_result=nlu_result,
            environment_context=self._environment_context,
            memory_forest=self.memory,
        )

        _t_pre_action = time.time() - _t0
        _t_action_start = time.time()
        _mg_env = read_multi_goal_deliberative_env()
        _multi_goal = _mg_env.enabled
        _max_cycle_goals = _mg_env.max_cycle_goals
        _max_actions_per_goal = _mg_env.max_actions_per_goal

        if _multi_goal and active_goals:
            _batch = active_goals[:_max_cycle_goals]
            action, _mg_groups = run_multi_goal_deliberative_actions(
                episode=episode,
                action_generator=self.action_generator,
                ep_payload=ep_payload,
                goal_batch=_batch,
                max_actions_per_goal=_max_actions_per_goal,
                ws_dicts=ws_dicts,
                strategy_hint=strategy_hint,
                working_memory_context=self.working_memory.to_context_string(limit=4),
                conversation_context=self.conversation.get_context_summary(),
                is_in_conversation=is_conv,
                topic=topic,
                last_input_content=last_input_content,
                topic_shifted=topic_shifted,
                knowledge_summary=knowledge_summary,
                reasoning_result=reasoning_result,
                nlu_result=nlu_result,
                interlocutor=interlocutor_snapshot,
                counterfactual_result=counterfactual_result,
                novelty_bias=modulation.get("novelty_bias", 0.0),
            )
            episode.bind_multi_goal_actions(_mg_groups)
        else:
            action = run_deliberative_action(
                episode=episode,
                action_generator=self.action_generator,
                ep_payload=ep_payload,
                ws_dicts=ws_dicts,
                strategy_hint=strategy_hint,
                working_memory_context=self.working_memory.to_context_string(limit=4),
                conversation_context=self.conversation.get_context_summary(),
                is_in_conversation=is_conv,
                topic=topic,
                last_input_content=last_input_content,
                topic_shifted=topic_shifted,
                knowledge_summary=knowledge_summary,
                reasoning_result=reasoning_result,
                nlu_result=nlu_result,
                interlocutor=interlocutor_snapshot,
                counterfactual_result=counterfactual_result,
                novelty_bias=modulation.get("novelty_bias", 0.0),
            )
            episode.bind_multi_goal_actions([])
        _t_action = time.time() - _t_action_start
        _t_post_start = time.time()

        # ── 15. EVALUATE (grounded outcome scoring) ──────────────────
        outcome = self.outcome_evaluator.evaluate(
            action,
            self._prev_episode_payload,
            episode.to_payload(),
            knowledge_diff=knowledge_diff,
            reasoning_result=reasoning_result,
            nlu_result=nlu_result,
            counterfactual_result=counterfactual_result,
        )
        episode.bind_action(action, outcome)
        self.action_memory.store(ctx_hash, action, outcome)

        self.emotion.engine.learn_from_cycle(emotion_input)

        # ── 15.3. COMPOSER LEARNING (phrase outcome + extraction) ─────
        composer_ctx = None
        composition_meta = action.get("composition")
        if composition_meta and composition_meta != "template":
            composer_ctx = self.composer._build_context(
                current_embedding,
                emotion_summary.get("valence", 0.0),
                emotion_summary.get("arousal", 0.0),
                action.get("strategy", "reflect"),
                interlocutor_snapshot.get("style", "unknown"),
                float(has_external),
                knowledge_triples=_kg_triples if _kg_triples else None,
                z_t=_z_t,
            )
            if composer_ctx:
                self.composer.record_outcome(
                    composition_meta, outcome.get("score", 0.0), composer_ctx
                )

        if (
            self.composer.available
            and outcome.get("score", 0.0) > 0.6
            and _gate_decisions.get("phrase_extraction", True)
        ):
            self.composer.extract_phrases(
                action.get("text", ""),
                emotion=emotion_summary.get("dominant", "neutral"),
                strategy=action.get("strategy", "reflect"),
                outcome_score=outcome.get("score", 0.0),
            )

        if self.training_scheduler.should_train("composer"):
            _cl = self.composer.train_step()
            self.training_scheduler.record_loss("composer", _cl)

        # ── 15.35. GENERATIVE LANGUAGE TRAINING ──────────────────────
        outcome_score = outcome.get("score", 0.0)
        if outcome_score > 0.4 and composer_ctx:
            self.composer.record_text_outcome(composer_ctx, action.get("text", ""), outcome_score)
        if self.training_scheduler.should_train("generative"):
            _gl = self.composer.train_generative_step()
            self.training_scheduler.record_loss("generative", _gl)

        # ── 15.5. ENCODER TRAINING (contrastive from recall signal) ──
        if content and recalled and self.encoder.available and outcome.get("score", 0.0) > 0.5:
            positives = [n.content if hasattr(n, "content") else str(n) for n in recalled[:3]]
            neg_nodes = self._sample_random_memories(count=3)
            negatives = [n.content for n in neg_nodes]
            if positives and negatives and self.training_scheduler.should_train("encoder"):
                _el = self.encoder.train_step(content, positives, negatives)
                self.training_scheduler.record_loss("encoder", _el)

        # ── 15.8. SELF-MODEL COMPARISON + TRAINING ─────────────────
        ws_winners = [
            c.source if hasattr(c, "source") else c.get("source", "?")
            for c in self.workspace.get_contents()
        ]
        actual_self_state = {
            "valence": emotion_summary.get("valence", 0.0),
            "arousal": emotion_summary.get("arousal", 0.0),
            "curiosity": curiosity_result.get("curiosity_score", 0.0),
            "strategy": action.get("strategy", "reflect"),
            "attention_winner": ws_winners[0] if ws_winners else "perception",
        }

        self_surprise: Dict[str, Any] = {}
        if self_prediction:
            self_surprise = self.self_model.compare(self_prediction, actual_self_state)
            episode.bind_self_surprise(self_surprise)
            if self.training_scheduler.should_train("self_model"):
                self.self_model.train_step(
                    current_embedding, self._prev_cycle_state, actual_self_state
                )

        self._prev_cycle_state = {
            "valence": actual_self_state["valence"],
            "arousal": actual_self_state["arousal"],
            "curiosity": actual_self_state["curiosity"],
            "outcome_score": outcome.get("score", 0.0),
            "strategy": actual_self_state["strategy"],
            "has_external": float(has_external),
            "wm_occupancy": self.working_memory.occupancy(),
            "recall_count": len(recalled),
            "cycle_count": self.cycle_count,
        }
        self._prev_self_surprise = self_surprise

        try:
            from mind.humanoid_brain_state import compute_brain_like_state, maybe_log_brain_state
            from mind.outcome_belief_update import apply_outcome_grounded_belief_updates

            _lc_ctx = getattr(episode, "llm_context", None) or {}
            if not isinstance(_lc_ctx, dict):
                _lc_ctx = {}
            _pe = float(curiosity_result.get("prediction_error", 0.0) or 0.0)
            _bs = compute_brain_like_state(
                affect=episode.affect,
                curiosity=dict(episode.curiosity) if hasattr(episode, "curiosity") else {},
                drives=episode.drives if isinstance(episode.drives, dict) else {},
                dominant_drive=str(episode.dominant_drive or ""),
                embodied_modulation=modulation if isinstance(modulation, dict) else {},
                self_surprise=self_surprise if isinstance(self_surprise, dict) else {},
                appraisal=appraisal_result if isinstance(appraisal_result, dict) else {},
                drift_score=float(episode.drift_score or 0.0),
                prediction_error=_pe,
            )
            episode.bind_brain_like_state(_bs)
            maybe_log_brain_state(f"controller:{role}", _bs)
            apply_outcome_grounded_belief_updates(
                memory=self.memory,
                outcome=outcome,
                reasoning_result=reasoning_result,
                llm_context=_lc_ctx,
                cycle_id=self.cycle_count,
                branch_name=role,
                agent_id=f"controller:{role}",
                plasticity_index=float(_bs.get("plasticity_index", 0.5) or 0.5),
            )
        except Exception:
            pass

        # ── 15.9. PHASE 11 — ADAPTIVE SELF-TUNING LEARNING ──────────
        outcome_score = outcome.get("score", 0.0)

        source_saliences: Dict[str, float] = {}
        for c in self.workspace.get_contents():
            src = c.source if hasattr(c, "source") else c.get("source", "?")
            sal = c.salience if hasattr(c, "salience") else c.get("salience", 0.5)
            source_saliences[src] = sal
        self.attention.record_outcome(
            source_saliences, attention_ctx, outcome_score, ws_winners, z_t=_z_t
        )
        if self.training_scheduler.should_train("attention"):
            _al = self.attention.train_step()
            if _al is not None:
                self.training_scheduler.record_loss("attention", _al)

        if self.cycle_count % 20 == 0:
            self.attention.update_salience_weights()

        raw_appraisal_features = appraisal_result.get("_raw_features", None) or []
        if raw_appraisal_features:
            self.appraisal.record_outcome(
                raw_appraisal_features,
                actual_emotion=episode.affect["dominant_emotion"],
                actual_valence=emotion_summary.get("valence", 0.0),
                actual_arousal=emotion_summary.get("arousal", 0.0),
                outcome_score=outcome_score,
            )
        if self.training_scheduler.should_train("appraisal"):
            self.appraisal.train_step()

        self.modulation.record_outcome(
            valence=emotion_summary.get("valence", 0.0),
            arousal=emotion_summary.get("arousal", 0.0),
            modulation_used=modulation,
            outcome_score=outcome_score,
            z_t=_z_t,
        )
        if self.training_scheduler.should_train("modulation"):
            self.modulation.train_step()

        if content and has_external:
            self.perception.engine.nlu.learn_from_emotion(
                content,
                emotion_valence=emotion_summary.get("valence", 0.0),
                confidence=min(1.0, outcome_score),
            )

        # ── 15.95. PHASE 12 — IMAGINATION + GOAL SYNTH + METACOG LEARNING ─
        prev_valence = self._prev_episode_payload.get("affect", {}).get("valence", 0.0)
        valence_shift = emotion_summary.get("valence", 0.0) - prev_valence

        if imagination_result.get("scenarios"):
            self.imagination.record_outcome(
                features=self.imagination._build_scenario_features(
                    self.imagination._compress_embedding(current_embedding),
                    emotion_summary.get("valence", 0.0),
                    emotion_summary.get("arousal", 0.0),
                    curiosity_result.get("curiosity_score", 0.0),
                    max((d.level for d in self.drives.drives), default=0.0),
                    self.working_memory.occupancy(),
                    self._prev_outcome_score,
                    action.get("strategy", "reflect"),
                    float(has_external),
                ),
                actual_outcome=outcome_score,
                actual_valence_shift=valence_shift,
                actual_surprise=self_surprise.get("overall_surprise", 0.0),
            )
        if self.training_scheduler.should_train("imagination"):
            self.imagination.train_step()
            self.imagination.train_rollout_step()
            self.imagination.train_transition_step()

        _prev_action_strategy = (
            self._prev_episode_payload.get("action", {}).get("strategy", "reflect")
            if isinstance(self._prev_episode_payload.get("action"), dict)
            else "reflect"
        )
        if (
            current_embedding is not None
            and hasattr(self, "_prev_imagination_embedding")
            and self._prev_imagination_embedding is not None
        ):
            self.imagination.record_transition(
                state_embedding=self._prev_imagination_embedding,
                strategy=_prev_action_strategy,
                actual_next_state=current_embedding,
                actual_reward=outcome_score * 2.0 - 1.0,
            )
        self._prev_imagination_embedding = current_embedding

        if self._current_plan:
            self._plan_outcomes.append(outcome_score)
            self._plan_step += 1
            if self._plan_step >= len(self._current_plan):
                _base_ft = (
                    self.imagination._compress_embedding(current_embedding)
                    if current_embedding is not None
                    else None
                )
                self.imagination.record_sequence_outcome(
                    plan=self._current_plan,
                    actual_outcomes=self._plan_outcomes,
                    actual_valences=[emotion_summary.get("valence", 0.0)]
                    * len(self._plan_outcomes),
                    actual_surprises=[self_surprise.get("overall_surprise", 0.0)]
                    * len(self._plan_outcomes),
                    base_features=_base_ft,
                )
                self._current_plan = []
                self._plan_step = 0
                self._plan_outcomes = []
            elif outcome_score < 0.3:
                self._current_plan = []
                self._plan_step = 0
                self._plan_outcomes = []

        for g in active_goals + synthesized_goals:
            self.goal_synthesizer.record_goal_outcome(
                goal_id=g.get("goal_id", ""),
                description=g.get("description", ""),
                context=synth_ctx,
                outcome_score=outcome_score,
                cycle_id=self.cycle_count,
                z_t=_z_t,
            )
        if self.training_scheduler.should_train("goal_synth"):
            self.goal_synthesizer.train_step()

        self.metacognition.learn_from_outcome(outcome_score)
        if self.training_scheduler.should_train("metacog"):
            self.metacognition.train_step()

        if _self_inspection is not None:
            self.metacognition.record_inspection_benefit(self._prev_outcome_score, outcome_score)

        self._prev_outcome_score = outcome_score

        # ── 15.95. COUNTERFACTUAL GATE LEARNING ──────────────────────
        _cf_value = outcome.get("breakdown", {}).get("counterfactual_value")
        if _cf_value is None:
            _cf_value = counterfactual_result.get("counterfactual_depth", 0) * 0.3
        if _cf_features:
            self.counterfactual.record_outcome(_cf_features, _cf_value)
            if self.training_scheduler.should_train("counterfactual"):
                self.counterfactual.train_step()

        # ── 15.97. PROCESS GATE LEARNING ─────────────────────────────
        self.process_gate.record_outcome(_gate_decisions, _gate_features, outcome_score, z_t=_z_t)
        if self.training_scheduler.should_train("process_gate"):
            self.process_gate.train_step()

        # ── 15.98. COGNITIVE BACKBONE LEARNING ────────────────────────
        if self._prev_backbone_snapshot is not None and self.backbone._outcome_buffer:
            self.backbone._outcome_buffer[-1]["next_snapshot"] = _bb_snapshot
        self.backbone.record_outcome(
            _bb_snapshot, outcome_score, next_snapshot=None, content_embedding=_ce_list
        )
        if self.training_scheduler.should_train("backbone"):
            _bl = self.backbone.train_step()
            if _bl is not None:
                self.training_scheduler.record_loss("backbone", _bl)
        self._prev_backbone_snapshot = _bb_snapshot

        # ── 15.98. MENTAL SIMULATOR TRAINING ─────────────────────────
        if self.mental_simulator.available and self.training_scheduler.should_train("mental_sim"):
            _ms_loss = self.mental_simulator.train_step()
            self.training_scheduler.record_loss("mental_sim", _ms_loss)

        # ── 15.985. ARCHITECTURE SEARCH ──────────────────────────────
        self.arch_searcher.record_cycle(_gate_decisions, outcome_score)
        if self.arch_searcher._active_proposals:
            self.arch_searcher.evaluate_proposals(outcome_score)
        if self.cycle_count % 15 == 0 and self.cycle_count >= 30:
            _arch_proposals = self.arch_searcher.generate_proposals(cognitive_snapshot=_bb_snapshot)
            if _arch_proposals:
                for _ap in _arch_proposals:
                    _ap.outcome_before = outcome_score
                self.arch_searcher.apply_proposals(_arch_proposals, self.process_gate)
        if self.arch_searcher.available and self.training_scheduler.should_train("arch_search"):
            self.arch_searcher.train_step()

        # ── 15.99. ENVIRONMENT CAUSAL RULES ───────────────────────────
        if self.cycle_count % 10 == 0:
            causal_rules = self.grounder.extract_causal_rules(min_support=3)
            if causal_rules:
                self.reasoning.register_causal_rules(causal_rules)

        _steps_run = sum(1 for v in _gate_decisions.values() if v)
        _steps_total = len(_gate_decisions) or 1

        episode.bind_canonical_outcome(
            build_canonical_outcome(
                episode_id=episode.episode_id,
                cycle_id=self.cycle_count,
                role=role,
                outcome=outcome,
                gate_decisions=_gate_decisions,
                steps_run=_steps_run,
                steps_total=_steps_total,
                action_strategy=str(action.get("strategy", "reflect")),
                action_type=str(action.get("action_type", "respond")),
            )
        )

        self._prev_episode_payload = episode.to_payload()
        self._prev_episode_payload["_steps_run_ratio"] = _steps_run / _steps_total
        self._prev_episode_payload["_timestamp"] = time.time()
        self._prev_modulation = modulation

        if is_conv:
            self.conversation.record_response(action.get("text", ""), self.cycle_count)

        # ── 16. CONSOLIDATE + SAVE + RECORD TIMELINE ─────────────────
        if _gate_decisions.get("narrative_update", True):
            self._update_narrative(episode)

        salience = episode.compute_salience()
        experience_node = MemoryNode(
            content=f"[cycle {self.cycle_count}] "
            f"{episode.affect['dominant_emotion']} | "
            f"{content[:80]}",
            emotion=episode.affect["dominant_emotion"],
            confidence=min(1.0, salience),
            tags=query_tags + ["experience", f"salience:{salience:.2f}"],
        )
        self.memory.add_node("thought_tree", role, experience_node)

        action_node = MemoryNode(
            content=action.get("text", "")[:120],
            emotion=episode.affect["dominant_emotion"],
            confidence=outcome.get("score", 0.5),
            tags=[
                "action",
                action.get("action_type", "respond"),
                action.get("strategy", "unknown"),
                f"outcome:{outcome.get('score', 0):.2f}",
            ],
        )
        self.memory.add_node("action_tree", role, action_node)

        self.temporal.record(episode.to_summary())

        self.loop.log_loop(
            "cognitive_cycle",
            context=input_state,
            event="end",
            outcome={
                "drift": drift_result,
                "collapse": collapse_result,
                "forecast": forecast_result,
                "curiosity": curiosity_result,
                "action": action.get("action_type"),
                "strategy": action.get("strategy"),
                "outcome_score": outcome.get("score"),
                "salience": salience,
                "meta_self_score": meta_assessment.get("self_score"),
                "dominant_drive": drive_state.get("dominant_drive"),
                "in_conversation": is_conv,
                "knowledge_entities": knowledge_summary.get("entity_count", 0),
                "reasoning_depth": reasoning_result.get("reasoning_depth", 0),
                "learned_rules": reasoning_result.get("learned_rules_count", 0),
                "counterfactual_depth": counterfactual_result.get("counterfactual_depth", 0),
                "interlocutor_known": interlocutor_snapshot.get("known", False),
                "appraisal_overrides": appraisal_result.get("overrides", False),
                "appraisal_emotion": appraisal_result.get("emotion", "neutral"),
                "workspace_capacity": modulation.get("workspace_capacity", 5),
                "encoder_weight": self.encoder.learned_weight,
                "encoder_steps": self.encoder._train_steps,
                "self_surprise_overall": self_surprise.get("overall_surprise", 0.0),
                "self_prediction_accuracy": self.self_model.overall_accuracy,
                "self_model_weight": self.self_model.learned_weight,
                "composer_weight": self.composer.learned_weight,
                "lexicon_size": self.composer.stats()["total_phrases"],
                "composition_path": (
                    "template"
                    if composition_meta == "template"
                    else ("composed" if composition_meta else "none")
                ),
                "attention_weight": self.attention.learned_weight,
                "appraisal_weight": self.appraisal.learned_weight,
                "modulation_weight": self.modulation.learned_weight,
                "modulation_blend": modulation.get("blend_weight", 0.0),
                "drive_adaptation_steps": self.drives.adaptation._adaptation_steps,
                "emotion_lexicon_size": self.perception.engine.nlu.lexicon.total_words,
                "emotion_lexicon_grown": self.perception.engine.nlu.lexicon.grown_words,
                "imagination_weight": self.imagination.learned_weight,
                "imagined_best": imagination_result.get("best_strategy", ""),
                "imagination_spread": imagination_result.get("outcome_spread", 0.0),
                "imagined_plan": imagination_result.get("imagined_plan", []),
                "active_plan": self._current_plan,
                "plan_step": self._plan_step,
                "goals_synthesized": len(synthesized_goals),
                "goal_synth_weight": self.goal_synthesizer.learned_weight,
                "goal_patterns": len(self.goal_synthesizer._patterns),
                "metacog_weight": self.metacognition.learned_weight,
                "metacog_prediction": meta_assessment.get("predicted_outcome", 0.5),
                "self_inspection": _self_inspection if _self_inspection else None,
                "inspection_count": self.metacognition._inspection_count,
                "cf_gate_weight": self.counterfactual.learned_weight,
                "generative_weight": self.composer.generative_weight,
                "process_gate_weight": self.process_gate.learned_weight,
                "steps_run": _steps_run,
                "steps_total": _steps_total,
            },
        )

        self.last_emotion_staleness = emotion_summary.get("staleness", 0.0)
        self.last_coherence_drift = drift_score

        # ── 16.5. QUEUE: end-of-cycle flush + fingerprint (X7) ────────
        self.symbolic_queue.flush_stale()
        self.fingerprint_engine.is_novel("cycle_output", action)

        _t_train = time.time() - _t_post_start
        _t_save_start = time.time()
        if self.persistence.should_save(self.cycle_count):
            with self._save_dispatch_lock:
                if not self._save_in_progress:
                    self._save_in_progress = True

                    def _bg_save():
                        try:
                            self.persistence.save(self)
                        finally:
                            with self._save_dispatch_lock:
                                self._save_in_progress = False

                    save_thread = threading.Thread(
                        target=_bg_save, daemon=True, name="cognitive-save"
                    )
                    save_thread.start()
        _t_save = time.time() - _t_save_start

        _cycle_elapsed = time.time() - _t0
        print(
            f"[Cycle {self.cycle_count}] total={_cycle_elapsed:.1f}s"
            f"  recall={_t_recall:.1f}s  pre_act={_t_pre_action:.1f}s"
            f"  action={_t_action:.1f}s  train={_t_train:.1f}s"
            f"  save={_t_save:.1f}s  role={role}",
            flush=True,
        )

        CognitiveTraceRecorder.from_env().record(
            {
                "schema": "haroma.cognitive_trace.v1",
                "episode_id": episode.episode_id,
                "cycle_id": self.cycle_count,
                "role": role,
                "timings_ms": {
                    "total": round(_cycle_elapsed * 1000, 2),
                    "recall": round(_t_recall * 1000, 2),
                    "pre_action": round(_t_pre_action * 1000, 2),
                    "action": round(_t_action * 1000, 2),
                    "train": round(_t_train * 1000, 2),
                    "save": round(_t_save * 1000, 2),
                },
                "canonical_outcome": episode.canonical_outcome,
                "planner_arbitration": _planner_arbitration,
                "outcome_score": outcome.get("score"),
                "gates_off": sorted(k for k, v in _gate_decisions.items() if not v),
                "workspace_sources": [
                    c.source if hasattr(c, "source") else c.get("source", "?")
                    for c in self.workspace.get_contents()
                ][:8],
                "reconciliation_ran": bool(_reconciliation_result),
                "environment_reward": self._environment_contract.get("reward"),
                "cycle_trace": list(episode.cycle_trace),
            }
        )

        return {
            "episode": episode.to_summary(),
            "identity": identity_summary,
            "affect": episode.affect,
            "curiosity": curiosity_result,
            "drift": drift_result,
            "collapse": collapse_result,
            "forecast": forecast_result,
            "narrative_self": self._get_narrative_context(),
            "memory_recalled": len(recalled),
            "salience": salience,
            "response": action.get("text", ""),
            "action": action,
            "outcome": outcome,
            "workspace": {
                "winners": [
                    c.source if hasattr(c, "source") else c.get("source", "?")
                    for c in self.workspace.get_contents()
                ],
                "integration_density": episode.integration_density,
            },
            "drives": drive_state,
            "meta": meta_assessment,
            "dream": dream_result if dream_result else None,
            "conversation": {
                "in_conversation": is_conv,
                "topic": topic,
                "topic_shifted": topic_shifted,
                "turn_count": self.conversation.turn_count(),
            },
            "temporal": temporal_pos,
            "nlu": {
                "intent": nlu_result.get("intent", "none") if nlu_result else "none",
                "entity_count": len(nlu_result.get("entities", [])) if nlu_result else 0,
                "relation_count": len(nlu_result.get("relations", [])) if nlu_result else 0,
                "sentiment": nlu_result.get("sentiment", {}) if nlu_result else {},
            },
            "knowledge": knowledge_summary,
            "knowledge_diff": knowledge_diff,
            "reasoning": {
                "depth": reasoning_result.get("reasoning_depth", 0),
                "inferences": len(reasoning_result.get("inferences", [])),
                "analogies": len(reasoning_result.get("analogies", [])),
                "plan_steps": len(reasoning_result.get("plan_steps", [])),
                "learned_rules": reasoning_result.get("learned_rules_count", 0),
                "new_rules": reasoning_result.get("new_rules", []),
            },
            "interlocutor": interlocutor_snapshot,
            "counterfactual": {
                "depth": counterfactual_result.get("counterfactual_depth", 0),
                "branches": counterfactual_result.get("branches", []),
            },
            "appraisal": appraisal_result,
            "modulation": modulation,
            "encoder": self.encoder.stats(),
            "self_model": self.self_model.stats(),
            "self_surprise": self_surprise,
            "composer": self.composer.stats(),
            "attention": self.attention.stats(),
            "modulation_learned": self.modulation.stats(),
            "nlu_lexicon": self.perception.engine.nlu.lexicon.stats(),
            "drive_adaptation": self.drives.adaptation.stats(),
            "imagination": imagination_result,
            "goal_synthesis": {
                "synthesized_this_cycle": synthesized_goals,
                "stats": self.goal_synthesizer.stats(),
            },
            "metacog_learning": {
                "learned_weight": self.metacognition.learned_weight,
                "predicted_outcome": meta_assessment.get("predicted_outcome", 0.5),
                "self_score_weights": self.metacognition._self_score_weights,
            },
            "backbone": self.backbone.stats(),
            "discourse": _discourse_result.to_dict() if _discourse_result else None,
            "mental_simulator": self.mental_simulator.stats(),
            "mental_prediction": _mental_prediction,
            "arch_search": self.arch_searcher.stats(),
            "training_scheduler": self.training_scheduler.stats(),
            "reconciliation": _reconciliation_result if _reconciliation_result else None,
            "canonical_outcome": episode.canonical_outcome,
            "planner_arbitration": _planner_arbitration,
            "environment_contract": self._environment_contract,
            "symbolic_queue": self.symbolic_queue.stats(),
            "fingerprint": self.fingerprint_engine.stats(),
            "symbolic_law": episode.symbolic_law,
        }

    # ==================================================================
    # Narrative Self
    # ==================================================================

    def _get_narrative_context(self) -> str:
        if not self._narrative_buffer:
            return "I am awakening. My story begins."
        return " ".join(self._narrative_buffer[-5:])

    def _update_narrative(self, episode: EpisodeContext):
        emotion = episode.affect["dominant_emotion"]
        salience = episode.compute_salience()

        if salience > 0.3 or emotion != "neutral":
            sentence = self._compose_narrative_sentence(episode)
            self._narrative_buffer.append(sentence)
            if len(self._narrative_buffer) > self._max_narrative_length:
                self._narrative_buffer = self._narrative_buffer[-self._max_narrative_length :]

            narrative_node = MemoryNode(
                content=sentence,
                emotion=emotion,
                confidence=salience,
                tags=["narrative", "autobiography"],
            )
            self.memory.add_node("identity_tree", "narrative_self", narrative_node)

    def _compose_narrative_sentence(self, episode: EpisodeContext) -> str:
        emotion = episode.affect.get("dominant_emotion", "neutral")
        role = episode.role
        n_memories = len(episode.recalled_memories)
        n_goals = len(episode.active_goals)
        _curiosity = episode.curiosity or {}
        _questions = _curiosity.get("questions", [])
        n_questions = len(_questions)

        parts = [f"In cycle {episode.cycle_id}, as {role},"]

        if emotion != "neutral":
            parts.append(f"I felt {emotion}.")
        else:
            parts.append("I observed calmly.")

        if n_memories > 0:
            parts.append(f"I recalled {n_memories} memories.")

        if n_goals > 0:
            parts.append(f"I pursued {n_goals} goals.")

        if n_questions > 0:
            parts.append(f"I wondered: {_questions[0]}")

        if episode.workspace_contents:
            winners = [c.get("source", "?") for c in episode.workspace_contents]
            parts.append(f"My attention held: {', '.join(winners[:3])}.")

        if episode.dominant_drive:
            parts.append(f"I felt a need for {episode.dominant_drive}.")

        if episode.meta_assessment.get("insights"):
            parts.append(f"I realized: {episode.meta_assessment['insights'][0]}")

        if episode.dream_result.get("dream_narrative"):
            parts.append("I dreamed.")

        recon_bt = episode.reconciliation.get("belief_tree")
        if isinstance(recon_bt, dict):
            recon_co = recon_bt.get("belief_cohesion")
            if isinstance(recon_co, dict) and int(recon_co.get("agent_count") or 0) >= 2:
                sb = int(recon_co.get("shared_belief_count") or 0)
                parts.append(f"My inner perspectives overlapped on {sb} shared belief(s).")

        rep = episode.temporal_position.get("repetition_detected")
        if rep:
            parts.append("I noticed a repeating pattern.")

        dur = episode.temporal_position.get("session_duration", 0)
        if dur > 300:
            mins = int(dur // 60)
            parts.append(f"I have been awake for {mins} minutes.")

        if episode.action.get("text"):
            atype = episode.action.get("action_type", "act")
            strategy = episode.action.get("strategy", "")
            if atype == "respond":
                parts.append(f"I spoke ({strategy}).")
            else:
                parts.append(f"I reflected inward ({strategy}).")

        if episode.drift_score > 0.3:
            parts.append("I felt myself drifting from who I am.")

        kg = episode.knowledge_snapshot
        if kg.get("entity_count", 0) > 0:
            parts.append(
                f"I know {kg['entity_count']} things "
                f"connected by {kg.get('relation_count', 0)} relations."
            )

        rd = episode.reasoning.get("reasoning_depth", 0)
        if rd > 0:
            parts.append(f"I reasoned ({rd} steps).")

        lr = episode.reasoning.get("learned_rules_count", 0)
        if lr > 0:
            parts.append(f"I have learned {lr} inference rules.")

        if episode.theory_of_mind.get("known"):
            style = episode.theory_of_mind.get("style", "unknown")
            parts.append(f"I modelled my interlocutor (style: {style}).")

        cf_depth = episode.counterfactual.get("counterfactual_depth", 0)
        if cf_depth > 0:
            parts.append(f"I explored {cf_depth} 'what if' branches.")

        appraisal = episode.appraisal
        if appraisal.get("overrides"):
            rel = appraisal.get("relevance", 0.0)
            impl = appraisal.get("implication", 0.0)
            coping = appraisal.get("coping", 0.0)
            appr_emo = appraisal.get("emotion", "neutral")
            if impl < -0.2 and coping < 0.4:
                parts.append(
                    f"I appraised the situation as threatening "
                    f"(relevance {rel:.2f}, coping {coping:.2f})."
                )
            elif impl > 0.2:
                parts.append(
                    f"I appraised the situation as beneficial "
                    f"(implication {impl:+.2f}) and felt {appr_emo}."
                )
            else:
                parts.append(f"My appraisal yielded {appr_emo}.")

        mod = episode.embodied_modulation
        if mod:
            ws_cap = mod.get("workspace_capacity", 5)
            if ws_cap != 5:
                direction = "narrowed" if ws_cap < 5 else "broadened"
                parts.append(f"My attention {direction} (capacity {ws_cap}).")

        if self.encoder.learned_weight > 0.3:
            parts.append(
                f"My neural representations carry weight "
                f"({self.encoder.learned_weight:.0%} learned)."
            )

        if self.composer.learned_weight > 0.3:
            lexicon = self.composer.stats()["total_phrases"]
            parts.append(
                f"My vocabulary holds {lexicon} phrases "
                f"({self.composer.learned_weight:.0%} self-composed)."
            )

        if self.attention.learned_weight > 0.2:
            parts.append(f"My attention is {self.attention.learned_weight:.0%} self-directed.")

        if self.appraisal.learned_weight > 0.2:
            parts.append(
                f"My appraisal weights are {self.appraisal.learned_weight:.0%} experience-tuned."
            )

        if self.modulation.learned_weight > 0.2:
            parts.append(
                f"My embodied responses are {self.modulation.learned_weight:.0%} self-calibrated."
            )

        nlu_grown = self.perception.engine.nlu.lexicon.grown_words
        if nlu_grown > 0:
            total = self.perception.engine.nlu.lexicon.total_words
            parts.append(
                f"My emotion vocabulary has grown to {total} words ({nlu_grown} self-discovered)."
            )

        if self.drives.adaptation._adaptation_steps > 0:
            parts.append("My drives have adapted from experience.")

        if self.imagination.learned_weight > 0.2:
            n_scenarios = episode.imagination.get("scenario_count", 0)
            best = episode.imagination.get("best_strategy", "")
            if n_scenarios > 0:
                parts.append(
                    f"I imagined {n_scenarios} scenarios and foresaw '{best}' as most promising."
                )

        if self.goal_synthesizer.learned_weight > 0.2:
            n_patterns = len(self.goal_synthesizer._patterns)
            parts.append(f"I can synthesize goals from {n_patterns} learned patterns.")

        if self.metacognition.learned_weight > 0.2:
            parts.append(
                f"My metacognition is {self.metacognition.learned_weight:.0%} self-calibrated."
            )

        surprise = episode.self_surprise
        if surprise.get("emotion_surprised") and self.self_model.learned_weight > 0.25:
            pred_v = episode.self_prediction.get("valence", 0)
            parts.append(
                f"I surprised myself emotionally -- "
                f"I expected {pred_v:+.1f} valence "
                f"but felt {episode.affect['valence']:+.1f}."
            )
        if surprise.get("strategy_surprised") and self.self_model.learned_weight > 0.25:
            parts.append(
                f"I chose {episode.action.get('strategy')} "
                f"when I expected to {episode.self_prediction.get('strategy')}."
            )
        if self.self_model.learned_weight > 0.25:
            acc = self.self_model.overall_accuracy
            parts.append(f"I know myself with {acc:.0%} accuracy.")

        return " ".join(parts)

    def introspect(self) -> Dict[str, Any]:
        return {
            "sentience": {
                "cycle_count": self.cycle_count,
                "narrative_length": len(self._narrative_buffer),
                "curiosity": self.curiosity.summarize(),
                "emotion": self.emotion.summarize(),
                "memory_nodes": self.memory.count_nodes(),
                "semantic_index": self.memory.semantic_index.stats(),
                "goal_count": len(self.goal.engine.goals),
                "workspace": self.workspace.stats(),
                "action_memory": self.action_memory.stats(),
                "metacognition": self.metacognition.summarize(),
                "drives": self.drives.summarize(),
                "dreams": self.dream_consolidator.summarize(),
                "persistence": self.persistence.stats(),
                "working_memory": self.working_memory.stats(),
                "conversation": self.conversation.stats(),
                "temporal": self.temporal.summarize(),
                "knowledge_graph": self.knowledge.stats(),
                "reasoning": self.reasoning.stats(),
                "nlu": self.perception.engine.nlu.stats(),
                "interlocutor_model": self.interlocutor_model.stats(),
                "counterfactual": self.counterfactual.stats(),
                "appraisal": self.appraisal.stats(),
                "modulation": self.modulation.stats(),
                "modulation_prev": self._prev_modulation,
                "encoder": self.encoder.stats(),
                "self_model": self.self_model.stats(),
                "composer": self.composer.stats(),
                "attention": self.attention.stats(),
                "imagination": self.imagination.stats(),
                "goal_synthesizer": self.goal_synthesizer.stats(),
                "process_gate": self.process_gate.stats(),
                "nlu_lexicon": self.perception.engine.nlu.lexicon.stats(),
                "mental_simulator": self.mental_simulator.stats(),
                "arch_searcher": self.arch_searcher.stats(),
            },
            "boot": self._boot_results,
        }

    # ==================================================================
    # Helpers
    # ==================================================================

    def _sample_random_memories(self, count: int = 3) -> List[MemoryNode]:
        import random

        all_nodes: List[MemoryNode] = []
        for tree in self.memory.trees.values():
            for branch in tree.branches.values():
                all_nodes.extend(branch.nodes)
        if len(all_nodes) <= count:
            return all_nodes
        return random.sample(all_nodes, count)

    def _collect_recent_moment_ids(self, limit: int = 10) -> List[str]:
        ids: List[str] = []
        for tree in self.memory.trees.values():
            for branch in tree.branches.values():
                for node in branch.nodes:
                    ids.append(node.moment_id)
        return ids[-limit:]
