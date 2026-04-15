# Module Reference

[<- Back to Index](index.md)

Complete inventory of every module in HaromaX6, organized by directory.

**Minded architecture:** modules fall under **[Brain CPU, Memory, Law, Fuel](minded-architecture-metaphor.md)** — e.g. `engine/LLM*` as CPU integrator, `core/Memory` as memory, cycle/gates/soul as law, `GoalEngine`/drives as fuel.

**Training / RL / sim / bridge entry points** (env vars, install tiers, `integrations.sim`, `mind/training/*`, `bridge/haroma_client`): [Training & integrations reference](reference-training-integrations.md).

---

## `core/` — Stateful Cognitive Modules

| File | Primary Classes | Description |
|------|----------------|-------------|
| `Memory.py` | `MemoryForest`, `MemoryTree`, `MemoryBranch`, `MemoryNode`, `SemanticIndex` | Hierarchical long-term memory with semantic search and thread safety. [Details](memory-forest.md) |
| `MemoryCore.py` | `MemoryCore` | X7 multi-agent facade over MemoryForest. [Details](x7-features.md) |
| `Reconciliation.py` | `DomainReflector`, `BeliefReconciler`, `GoalReconciler`, `ValueReconciler`, `EmotionReconciler`, `DreamReconciler`, `SymbolicReconciliationEngine` | X7 domain reflectors and orchestrator. [Details](x7-features.md) |
| `SymbolicQueue.py` | `SymbolicQueue`, `SymbolicFingerprintEngine` | X7 two-slot queue with hash-based drift control. [Details](x7-features.md) |
| `CellRoles.py` | *(functions only)* | X7 soft 3-bit role annotations for 31 modules. |
| `OrganRegistry.py` | `OrganRegistry` | X7 15-organ taxonomy with health monitoring. [Details](x7-features.md) |
| `SoulBinder.py` | `SoulBinder` | Two-phase soul loading. [Details](soul-system.md) |
| `Persistence.py` | `CognitivePersistence` | Sharded incremental save/load per memory tree. |
| `WorkingMemory.py` | `WorkingMemoryItem`, `WorkingMemory` | Slot-based short-term memory with decay. |
| `GlobalWorkspace.py` | `Coalition`, `GlobalWorkspace` | Broadcast, competition, winner selection. |
| `KnowledgeGraph.py` | `Entity`, `Relation`, `KnowledgeGraph` | Entity-relation store with triple integration. |
| `KnowledgeBase.py` | `OntologyCore`, `ConceptChainEngine`, `DoctrineSynthesizer`, `SchemaCore`, ... | Ontology, doctrine, and schema management (12+ classes). |
| `ConversationTracker.py` | `Turn`, `ConversationTracker` | Turn-by-turn dialogue state and topic detection. |
| `HomeostaticDrives.py` | `Drive`, `DriveAdaptation`, `HomeostaticSystem` | Biological-analog drive system. |
| `InterlocutorModel.py` | `InterlocutorState`, `InterlocutorModel` | Models of other speakers' mental states. |
| `ActionLoop.py` | `ActionCandidate`, `ActionGenerator`, `OutcomeEvaluator`, `ActionMemory` | Deliberative action selection and outcome scoring. |
| `EpisodeContext.py` | `EpisodeContext` | Per-cycle episode binding. |
| `Perception.py` | `PerceptionBridge`, `TextInterpreter`, `VisualSymbolicPerceptor`, `AudioSymbolicPerceptor`, ... | Multi-modal sensory interpretation (20+ classes). |
| `NLUProcessor.py` | `LearnedLexicon`, `NLUProcessor` | Intent, entity, sentiment extraction with growing lexicon. |
| `DiscourseProcessor.py` | `DiscourseFrame`, `DiscourseResult`, `DiscourseProcessor` | Compositional NLU frame extraction. |
| `Dream.py` | `DreamCore` | Dream generation and motif tracking. |
| `DreamConsolidator.py` | `DreamConsolidator` | Memory replay, compression, pruning via `remove_node`. |
| `EmbodiedModulation.py` | `EmbodiedModulation` | Emotion reshapes processing parameters. |
| `Narrative.py` | `NarrativeForecaster`, `NarrativeStateEngine`, `ArcSynthesizer`, ... | Narrative self-construction (11 classes). |
| `SymbolicUtils.py` | `ReflectiveMixin`, `MultiplexReflectiveMixin`, `AffectModulator`, `SymbolicDriftDetector`, ... | Shared symbolic primitives and drift detection. |
| `AffectiveReasoning.py` | `EmotionBeliefDriftDetector`, `EmotionCore`, `FuzzyAlignmentScorer`, ... | Emotion-belief coupling and fuzzy alignment. |
| `BeliefCohesion.py` | `BeliefOverlapAnalyzer`, `ConsensusSynthesizer`, `BeliefDriftMonitor`, ... | Belief consistency analysis and conflict resolution. |
| `Goal.py` | `GoalCollapseDetector`, `GoalEngine` | Goal lifecycle and collapse prediction. |

---

## `engine/` — Processing Engines

| File | Primary Classes | Description |
|------|----------------|-------------|
| `NeuralEncoder.py` | `NeuralEncoder`, `_EncoderModel`, `_PretrainedBackbone` | Semantic embeddings: frozen HF encoder + trainable projection; env `HAROMA_SEMANTIC_ENCODER` (default MiniLM). Per-persona overrides: `SharedResources.encoder_for(persona_id)` (`soul/agents.json` `semantic_encoder` or `HAROMA_SEMANTIC_ENCODER_<ID>`). Distinct from chat LLM and from `HAROMA_EMOTION_ENCODER`. |
| `EmotionEngine.py` | `LearnedEmotionModel`, `EmotionEngine` | Learned neural emotion model; frozen HF encoder + heads. Default backbone DistilBERT; set env `HAROMA_EMOTION_ENCODER` (e.g. `google/mobilebert-uncased`). Not the chat LLM. |
| `ReasoningEngine.py` | `Rule`, `Analogy`, `PlanStep`, `RuleLearner`, `ReasoningEngine` | Inference chains, analogies, planning, rule learning. |
| `CuriosityEngine.py` | `WorldModel`, `KGWorldModel`, `EmbeddingWorldModel`, `CuriosityEngine` | Prediction-error curiosity with dual world models. |
| `Imagination.py` | `Scenario`, `Imagination` | Internal scenario simulation with outcome prediction. |
| `SelfModel.py` | `SelfModel`, `_SelfPredictorNet` | Predictive self-awareness and surprise detection. |
| `LanguageComposer.py` | `PhraseLexicon`, `WordVocabulary`, `LanguageComposer`, `_TransformerDecoderNet` | Learned phrase composition with transformer decoder. |
| `CognitiveBackbone.py` | `CognitiveBackbone` | Unified state snapshot and latent z_t vector. |
| `MetaCognitionEngine.py` | `MetaCognitionEngine` | Self-assessment, confidence calibration, strategy selection. |
| `ProcessGate.py` | `ProcessGate`, `_ProcessGateNet` | Learned gating for optional cognitive steps. |
| `AppraisalEngine.py` | `AppraisalEngine` | Goal-aware emotion refinement. |
| `CounterfactualEngine.py` | `CounterfactualEngine`, `_ShadowKG` | "What-if" branching with learned depth gating. |
| `GoalSynthesizer.py` | `GoalPattern`, `GoalSynthesizer` | Goal discovery from curiosity/emotion/drive patterns. |
| `LearnedAttention.py` | `LearnedAttention` | Learned salience weights for workspace broadcast. |
| `TrainingScheduler.py` | `PrioritizedBuffer`, `TrainingScheduler` | Prioritized online learning with module budgets. |
| `ArchitectureSearcher.py` | `ArchitectureSearcher`, `ModuleRecord`, `Proposal` | Architecture-level self-modification proposals. |
| `ComputeFabric.py` | `ComputeFabric` | Resource management and compute allocation. |
| `TemporalEngine.py` | `TemporalEngine` | Temporal binding and repetition detection. |
| `MentalSimulator.py` | `AgentMentalModel`, `MentalSimulator` | Theory of mind: predicts other agents' behavior. |
| `LoopMemoryLogger.py` | `LoopMemoryLogger` | Logs cognitive loop events to memory. |
| `MemorySummarizer.py` | `MemorySummarizer` | Compresses and summarizes memory branches. |
| `SymbolicLoopManager.py` | `SymbolicLoopManager` | Symbolic loop state management. |
| `NarrativeReflector.py` | `NarrativeReflector` | Narrative arc reflection and insight extraction. |
| `ReflectiveConsciousnessEngine.py` | `ReflectiveConsciousnessEngine` | Higher-order reflective consciousness. |
| `DoctrineCortex.py` | `DoctrineCortex` | Doctrine-based decision guidance. |
| `FusionResolver.py` | `FusionResolver` | Conflict resolution between symbolic states. |
| `IdentityEngine.py` | `IdentityEngine` | Identity state computation and role management. |

---

## `mind/` — Orchestration

| File | Primary Classes | Description |
|------|----------------|-------------|
| `control.py` | `ElarionController` | Central orchestrator with 40+ step `run_cycle()`. [Details](cognitive-cycle.md) |
| `deploy_config.py` | `load_dotenv`, `http_listen_host`, `http_listen_port` | Project-root `.env`, bind host/port (`HAROMA_BIND_HOST`, `HAROMA_HTTP_PORT`). [Getting Started](getting-started.md) |
| `server_state.py` | `HaromaServerState`, `HAROMA_FLASK_EXTENSION_KEY`, `get_haroma_server_state` | Boot agent, sensor poller, and async chat registry attach via `get_haroma_server_state(app)` (key `HAROMA_FLASK_EXTENSION_KEY`, default `"haroma"`) — not module-level globals. |
| `client_ip.py` | `get_effective_client_ip`, `direct_peer_trusted_for_xff` | Rate limits and structured logs use the effective client; optional `X-Forwarded-For` when the direct peer is trusted (`HAROMA_HTTP_USE_X_FORWARDED_FOR`, `HAROMA_HTTP_TRUSTED_PROXIES`). [API](api-reference.md) |
| `elarion_server_v2.py` | Flask app, `BootAgent`, sensor poller | HTTP server, multi-agent boot, research routes, optional bearer/rate-limit/structured logs, `/laws`. [Details](api-reference.md) |
| `manager.py` | `IdentityManager`, `GoalManager`, `EmotionManagerSimple`, `DreamManager`, `PerceptionManager`, `DriftManager`, `CollapseManager`, `ForecastManager`, `LoopLoggerManager`, ... | Lightweight manager wrappers (14 managers). |

---

## `sensors/` — Hardware Adapters

| File | Primary Classes | Description |
|------|----------------|-------------|
| `adapters.py` | `SensorAdapter`, `VisionAdapter`, `AudioAdapter`, `TouchAdapter`, `SmellAdapter`, `TasteAdapter`, `LidarAdapter`, `InfraredAdapter`, `ImuAdapter`, `GpsAdapter`, `SensorPoller` | 10 hardware adapters. [Details](sensors.md) |

---

## `environment/` — World Grounding

| File | Primary Classes | Description |
|------|----------------|-------------|
| `EnvironmentGrounder.py` | `EnvironmentGrounder` | Causal rule learning from environment. |
| `TextEnvironment.py` | `Room`, `WorldObject`, `Agent`, `Event`, `TextEnvironment` | Simulated text world. |

---

## `boot/` — Sensory Intake Clients

| File | Function | Description |
|------|----------|-------------|
| `client_eyes.py` | `capture_vision` | Camera/vision capture |
| `client_ears.py` | `capture_audio` | Audio capture |
| `client_skin.py` | `read_touch` | Touch/pressure reading |
| `client_nose.py` | `read_smell` | Chemical sensor reading |
| `client_tounge.py` | `read_taste_sensor` | Taste sensor reading |
| `client_text.py` | `read_text` | Text input capture |
| `sensory_intake_server.py` | `update_input`, `get_input` | Centralized sensory intake |

---

## `utils/` — Shared Utilities

| File | Primary Classes | Description |
|------|----------------|-------------|
| `module_base.py` | `ModuleBase` | Common base class for all modules |
| `gradient_vote.py` | `GradientVote` | Zone-based gradient voting |
| `sense_transform.py` | `SenseTransform` | Sensory data normalization |
| `sense_utils.py` | `InputManager` | Input routing |

---

## `soul/` — Immutable Identity

| File | Description |
|------|-------------|
| `essence.json` | Name, guardian, lineage, core rule |
| `principle.json` | Beliefs, alignment scores |
| `construction.json` | Architecture metadata, version, tier |
| `memory.json` | Pre-seeded bootstrap memories |
| `patterns.json` | Inherited behavioral patterns |
| `feedback.json` | Historical feedback data |

---

## Summary

| Directory | Files | Classes | Role |
|-----------|-------|---------|------|
| `core/` | 25+ | 150+ | Stateful cognitive modules |
| `engine/` | 28+ | 60+ | Processing engines |
| `mind/` | 3 | 17+ | Orchestration |
| `sensors/` | 1 | 11 | Hardware adapters |
| `environment/` | 2 | 6 | World grounding |
| `boot/` | 7 | — | Sensory intake functions |
| `utils/` | 4 | 4 | Shared utilities |
| **Total** | **70+** | **250+** | |

---

## Related Docs

- [Minded architecture](minded-architecture-metaphor.md) — How modules map to Brain CPU / Memory / Law / Fuel
- [Architecture Overview](architecture.md) — Directory layout in context
