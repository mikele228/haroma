"""
SharedResources -- thread-safe container of all cognitive state shared
across agents.

Holds the shared cognitive stack for the **multi-agent** deployment
(``BootAgent``, TrueSelf, Personas).  For library use, ``ElarionController``
in ``mind.control`` should follow the same **phase order**; see
``mind.cycle_flow`` / ``COGNITIVE_PIPELINE_STEPS``.

Each agent receives a reference and accesses only what it needs.  Mutable
collections inside (MemoryForest, KnowledgeGraph, etc.) use internal locks;
this class holds them together and provides boot-time initialization
(soul bind -> persistence load -> soul reassert).

``signals`` (:class:`agents.runtime_signals.RuntimeSignals`) tracks HTTP chat
depth stack and last background training completion for :class:`agents.background_cadence.BackgroundCadence`.
Optional :class:`core.self_model_train_batch.SelfModelTrainBatch` on ``_self_model_last_train_ctx``
feeds background self-model training.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Iterator, List, Optional

from core.MeaningLexicon import MeaningLexicon
from core.self_model_train_batch import SelfModelTrainBatch
from core.concurrency import ConcurrencyCoordinator
from core.cognitive_null import CognitiveNull, is_cognitive_null
from agents.runtime_signals import RuntimeSignals
from mind.cognitive_observability import CognitiveMetrics

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _llm_boot_preflight_ok() -> bool:
    """Windows: run LLMBackend import+construct in a subprocess.

    Native crashes (no Python traceback) during ``import engine.LLMBackend`` or
    ``LLMBackend()`` would otherwise kill the whole server. If the child dies or
    errors, skip the LLM stack and use :class:`CognitiveNull` so boot can finish.

    Set ``HAROMA_LLM_BOOT_PREFLIGHT=0`` to skip (faster when stable). Preflight
    defaults **on** only for ``os.name == "nt"``.
    """
    flag = str(os.environ.get("HAROMA_LLM_BOOT_PREFLIGHT", "") or "").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return True
    if os.name != "nt":
        return True
    root = _PROJECT_ROOT
    script = (
        "import os, sys\n"
        f"os.chdir({root!r})\n"
        f"sys.path.insert(0, {root!r})\n"
        "from engine.LLMBackend import LLMBackend\n"
        "LLMBackend(model_path=None, n_ctx=512, n_gpu_layers=0)\n"
        "print('HAROMA_PREFLIGHT_OK', flush=True)\n"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=root,
        )
    except subprocess.TimeoutExpired:
        print(
            "[SharedResources] LLM boot preflight: timeout — using CognitiveNull for LLM",
            flush=True,
        )
        return False
    ok = r.returncode == 0 and "HAROMA_PREFLIGHT_OK" in (r.stdout or "")
    if not ok:
        print(
            "[SharedResources] LLM boot preflight failed "
            f"(rc={r.returncode}) — using CognitiveNull for LLM. "
            f"stdout={r.stdout!r} stderr={r.stderr!r}",
            flush=True,
        )
    return ok


# =====================================================================
# Agent configuration loader
# =====================================================================

_DEFAULT_AGENT_CONFIG: Dict[str, Any] = {
    "routing_strategy": "broadcast_claim",
    "claim_timeout_ms": 500,
    "max_personas": 5,
    "initial_personas": [
        {
            "id": "primary",
            "name": "Core",
            "affinity": {
                "topics": [],
                "emotion_range": "all",
                "is_default": True,
            },
        }
    ],
    "background": {
        "tick_interval": 5.0,
        "reconcile_every_n_ticks": 2,
        "dream_every_n_ticks": 5,
        "training_enabled": True,
        "spawn_on_divergence_threshold": 0.7,
        "autonomy_enabled": True,
        "autonomy_initiative_every_n_ticks": 7,
        "autonomy_stimulus_queue_max": 12,
    },
    "input": {
        # Lower = less queue latency for ``depth=normal`` (HTTP waits on InputAgent tick).
        "tick_interval": 0.12,
        # Reserved (MessageBus stores it; no timed dead-letter sweep yet).
        "dead_letter_timeout_ms": 2000,
    },
    "persona": {
        "tick_interval": 0.5,
    },
    "llm": {
        "mode": "auto",
        "prefer_local": False,
        "local_gguf": None,
    },
}


def load_agent_config(
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load agent configuration from soul/agents.json, merging with defaults."""
    if path is None:
        path = os.path.join(_PROJECT_ROOT, "soul", "agents.json")
    import copy

    config = copy.deepcopy(_DEFAULT_AGENT_CONFIG)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            _deep_merge(config, user_config)
            print(f"[SharedResources] Loaded agent config from {path}", flush=True)
        except Exception as exc:
            print(
                f"[SharedResources] Failed to load {path}: {exc}, using defaults",
                flush=True,
            )
    else:
        print(
            f"[SharedResources] No agent config at {path}, using defaults",
            flush=True,
        )
    return config


def _deep_merge(base: dict, override: dict):
    """Recursively merge *override* into *base* in place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _apply_programmed_llm(llm_cfg: Dict[str, Any]) -> None:
    """Rule-based LLM stand-in: no GGUF load, no API calls."""
    llm_cfg["use_programmed"] = True
    llm_cfg["local_gguf"] = None
    llm_cfg["api_provider"] = None
    llm_cfg["api_model"] = None
    print(
        "[SharedResources] LLM engine=programmed — rule-based responder (no weights)",
        flush=True,
    )


def _disable_programmed_stub() -> bool:
    """When true, never use ``ProgrammedLLM`` — require local GGUF or API."""
    return str(os.environ.get("HAROMA_DISABLE_PROGRAMMED_LLM", "") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def apply_soul_llm_overrides(
    soul_llm: Any,
    llm_cfg: Dict[str, Any],
) -> None:
    """Apply ``soul/agents.json`` ``llm`` onto tier-detected ``llm_cfg`` (mutates *llm_cfg*).

    ``mode``:
      - ``auto`` (default): keep auto-selected ``local_gguf`` from ``models/`` / resource tier.
      - ``api_only``: disable local GGUF (API only on tier that provides it).

    ``engine``:
      - ``programmed``: use ``engine.ProgrammedLLM`` (instant, no GGUF/API).

    ``prefer_local``: if true, clear API provider so ``LLMBackend`` loads the local GGUF
    even when API keys exist (tier "cloud").

    Env ``HAROMA_LLM_MODE=local`` or ``local_only`` forces ``prefer_local`` behavior.
    Env ``HAROMA_LLM_ENGINE=programmed`` forces the programmed responder.
    Env ``HAROMA_DISABLE_PROGRAMMED_LLM=1`` skips programmed stub (need real GGUF/API).
    """
    env_engine = str(os.environ.get("HAROMA_LLM_ENGINE", "") or "").lower().strip()
    if env_engine == "programmed" and not _disable_programmed_stub():
        _apply_programmed_llm(llm_cfg)
        return
    if env_engine == "programmed" and _disable_programmed_stub():
        print(
            "[SharedResources] HAROMA_DISABLE_PROGRAMMED_LLM=1 — ignoring "
            "HAROMA_LLM_ENGINE=programmed",
            flush=True,
        )

    env_mode = (os.environ.get("HAROMA_LLM_MODE") or "").lower().strip()
    if env_mode in ("local", "local_only"):
        llm_cfg["api_provider"] = None
        llm_cfg["api_model"] = None
        print(
            "[SharedResources] HAROMA_LLM_MODE=local — API disabled for LLM",
            flush=True,
        )

    if not isinstance(soul_llm, dict) or not soul_llm:
        return

    soul_engine = str(soul_llm.get("engine", "") or "").lower().strip()
    if soul_engine == "programmed" and not _disable_programmed_stub():
        _apply_programmed_llm(llm_cfg)
        return
    if soul_engine == "programmed" and _disable_programmed_stub():
        print(
            "[SharedResources] HAROMA_DISABLE_PROGRAMMED_LLM=1 — ignoring soul llm.engine=programmed",
            flush=True,
        )

    mode = str(soul_llm.get("mode", "auto")).lower().strip()
    if soul_llm.get("prefer_local") is True:
        llm_cfg["api_provider"] = None
        llm_cfg["api_model"] = None
        print(
            "[SharedResources] llm.prefer_local=true — using local GGUF only",
            flush=True,
        )

    raw_path = soul_llm.get("local_gguf")
    if raw_path is not None and str(raw_path).strip():
        rp = str(raw_path).strip()
        if not os.path.isabs(rp):
            p = os.path.normpath(os.path.join(_PROJECT_ROOT, rp))
        else:
            p = os.path.abspath(os.path.expanduser(rp))
        if os.path.isfile(p):
            llm_cfg["local_gguf"] = p
            print(f"[SharedResources] llm.local_gguf (soul): {p}", flush=True)
        else:
            print(
                f"[SharedResources] llm.local_gguf not found, skipping: {p}",
                flush=True,
            )
            if mode == "auto" and llm_cfg.get("local_gguf"):
                print(
                    f"[SharedResources] llm.mode=auto — keeping ResourceConfig "
                    f"local_gguf: {llm_cfg['local_gguf']}",
                    flush=True,
                )
            else:
                print(
                    "[SharedResources] Tip: set llm.local_gguf to an existing "
                    "models/*.gguf path, or rely on auto-selection from models/.",
                    flush=True,
                )

    if mode == "api_only":
        llm_cfg["local_gguf"] = None
        print(
            "[SharedResources] llm.mode=api_only — cleared local_gguf",
            flush=True,
        )

    if _disable_programmed_stub():
        llm_cfg["use_programmed"] = False


# =====================================================================
# SharedResources
# =====================================================================


class SharedResources:
    """Holds every cognitive module that agents share.

    Initialized by BootAgent.  Individual agents store a reference to this
    and access only the fields they need.

    Cognitive engines (curiosity, reasoning, knowledge, …) are booted via
    ``mind.manager`` *Manager* classes; each pair ``*_manager`` / short name
    shares the same underlying ``.engine`` instance as ``ElarionController``.
    """

    def __init__(self):
        self.agent_config: Dict[str, Any] = {}

        # Resource tier (set during boot)
        self.resource_config = None

        # Core memory
        self.memory = None  # MemoryForest
        self.memory_core = None  # MemoryCore (multi-agent branch API)

        # Neural / LLM
        self.encoder = None  # NeuralEncoder
        self._persona_encoder_cache: Dict[str, Any] = {}
        self._base_semantic_model_id: str = ""
        self.llm_manager = None  # LLMManager → .engine is LLMBackend
        self.llm_backend = None  # alias: llm_manager.engine (compat)
        self.fabric = None  # ComputeFabric

        # Knowledge (managers own engines; short names = .engine for agent compat)
        self.knowledge_manager = None  # KnowledgeGraphManager
        self.knowledge = None  # KnowledgeGraph
        self.reasoning_manager = None  # ReasoningManager
        self.reasoning = None  # ReasoningEngine

        # Managers (lightweight, mostly stateless wrappers)
        self.identity = None  # IdentityManager
        self.goal = None  # GoalManager
        self.dream_mgr = None  # DreamManager
        self.law = None  # LawManager
        self.myth = None  # MythManager
        self.value = None  # ValueManager
        self.fusion = None  # FusionManager
        self.perception = None  # PerceptionManager

        # Cognitive engines (mind.manager → engine)
        self.curiosity_manager = None  # CuriosityManager
        self.curiosity = None  # CuriosityEngine
        self.metacognition_manager = None  # MetaCognitionManager
        self.metacognition = None  # MetaCognitionEngine
        self.temporal_manager = None  # TemporalManager
        self.temporal = None  # TemporalEngine
        self.counterfactual_manager = None  # CounterfactualManager
        self.counterfactual = None  # CounterfactualEngine
        self.appraisal_manager = None  # AppraisalManager
        self.appraisal = None  # AppraisalEngine
        self.modulation_manager = None  # ModulationManager
        self.self_model_manager = None  # SelfModelManager
        self.self_model = None  # SelfModel
        self._self_model_last_train_ctx: Optional[SelfModelTrainBatch] = None
        self.attention_manager = None  # AttentionManager
        self.imagination_manager = None  # ImaginationManager
        self.imagination = None  # Imagination
        self.goal_synthesizer_manager = None  # GoalSynthesizerManager
        self.goal_synthesizer = None  # GoalSynthesizer

        # Composition / language
        self.composer = None  # LanguageComposer
        self.virtual_llm_tree = None  # VirtualLLMTree (llm_tree + LLMBackend facade)
        self.discourse = None  # DiscourseProcessor
        self.meaning_lexicon = MeaningLexicon()

        # Action
        self.action_generator = None  # ActionGenerator
        self.outcome_evaluator = None  # OutcomeEvaluator
        self.action_memory = None  # ActionMemory

        # Attention / modulation / workspace template
        self.attention = None  # LearnedAttention
        self.modulation = None  # EmbodiedModulation
        self.process_gate = None  # ProcessGate
        self.backbone = None  # CognitiveBackbone

        # Consolidation
        self.dream_consolidator = None  # DreamConsolidator
        self.drives = None  # HomeostaticSystem

        # Persistence
        self.persistence = None  # CognitivePersistence
        self.soul_binder = None  # SoulBinder

        # Infrastructure
        self.training_scheduler = None  # TrainingScheduler
        self.arch_searcher = None  # ArchitectureSearcher
        self.mental_simulator = None  # MentalSimulator
        self.grounder = None  # EnvironmentGrounder
        self.action_dispatcher = None  # ActionDispatcher

        # X7 layers
        self.reconciliation = None  # SymbolicReconciliationEngine
        self.symbolic_queue = None  # SymbolicQueue
        self.fingerprint_engine = None  # SymbolicFingerprintEngine
        self.organ_registry = None  # OrganRegistry

        # Interlocutor
        self.interlocutor_model = None  # InterlocutorModel

        # Reflective diagnostics
        self.drift = None  # DriftManager
        self.collapse = None  # CollapseManager
        self.forecast = None  # ForecastManager
        self.loop = None  # LoopLoggerManager
        self.reflector = None  # ReflectionManager

        # Personality (seeded by SoulBinder._bind_personality)
        self.personality_seed: Dict[str, float] = {}
        self._personality_profiles: List[Any] = []

        # Boot metadata
        self.boot_results: Dict[str, Any] = {}
        self.boot_time: float = 0.0
        self.cycle_count: int = 0

        # Board / CEO governance (filled in ``initialize`` after config load)
        self.goal_board: Optional[Any] = None

        self.locks = ConcurrencyCoordinator()
        # Backward-compatible aliases (same lock objects as ``locks``).
        self._cycle_lock = self.locks.cycle
        self._neural_lock = self.locks.neural
        self._http_chat_lock = self.locks.http_chat
        self._autonomy_metrics_lock = self.locks.autonomy_metrics
        self._autonomy_stim_lock = self.locks.autonomy_stimulus

        # Multi-agent: BootAgent wires TrueSelf emotion / narrative for persistence
        self.persist_emotion = None
        self.persist_narrative_source = None
        self._pending_emotion_restore: Optional[Dict[str, Any]] = None
        self._pending_narrative_restore: Optional[List[str]] = None

        # HTTP /chat requests awaiting a reply (pause idle inner-dialogue)
        self._http_chat_inflight = 0
        self.signals = RuntimeSignals(self)
        self.cognitive_metrics = CognitiveMetrics()

        # Autonomy phases 1–5: metrics, stimulus queue, sandbox (filled in init)
        self._autonomy_metrics: Dict[str, int] = {}
        self._autonomy_stimulus_queue: List[Dict[str, Any]] = []
        self._autonomy_stimulus_queue_max: int = 12
        self.autonomy_sandbox: Optional[Any] = None

        # General-agent: latest structured environment from host (HTTP integrator)
        self._agent_environment_lock = threading.Lock()
        self._robot_bridge_metrics_lock = threading.Lock()
        self._robot_bridge_metrics: Dict[str, int] = {}
        # Lab research: run manifest + experiment id ring (set in ``initialize``)
        self.lab_run_id: str = ""
        self.run_manifest: Dict[str, Any] = {}
        self.agent_environment: Dict[str, Any] = {}
        self.agent_environment_error: str = ""
        self.agent_environment_received_at: float = 0.0
        # Robot physique/posture: merged across ``agent_environment`` updates (sensor fallbacks).
        self._robot_body_effective: Dict[str, Any] = {}
        self._robot_body_ever_observed: bool = False
        self._robot_body_field_stamps: Dict[str, float] = {}

        # Optional display names keyed by sanitized ``user_id`` (HTTP /chat); in-process only
        self._user_profile_lock = threading.Lock()
        self._user_display_names: Dict[str, str] = {}

    def set_user_display_name(self, user_id: str, display_name: str) -> None:
        """Store an optional label for a client ``user_id`` (from JSON ``display_name``)."""
        from mind.user_identity import sanitize_user_id

        uid = sanitize_user_id(user_id)
        if not uid:
            return
        label = str(display_name).strip()
        if not label:
            return
        with self._user_profile_lock:
            self._user_display_names[uid] = label[:160]

    def get_user_display_name(self, user_id: Optional[str]) -> str:
        from mind.user_identity import sanitize_user_id

        uid = sanitize_user_id(user_id)
        if not uid:
            return ""
        with self._user_profile_lock:
            return self._user_display_names.get(uid, "")

    @staticmethod
    def _autonomy_stimulus_fingerprint(text: str) -> str:
        t = " ".join(str(text).strip().split())
        return t[:120]

    def autonomy_bump(self, key: str, delta: int = 1) -> None:
        with self._autonomy_metrics_lock:
            self._autonomy_metrics[key] = self._autonomy_metrics.get(key, 0) + max(0, int(delta))

    def autonomy_metrics_snapshot(self) -> Dict[str, int]:
        with self._autonomy_metrics_lock:
            return dict(self._autonomy_metrics)

    def _robot_bridge_metrics_bump(self, key: str, delta: int = 1) -> None:
        with self._robot_bridge_metrics_lock:
            self._robot_bridge_metrics[key] = self._robot_bridge_metrics.get(key, 0) + max(
                0, int(delta)
            )

    def robot_bridge_metrics_snapshot(self) -> Dict[str, int]:
        """Counters for ``POST /robot/bridge/feedback`` (also surfaced under ``GET /status``)."""
        with self._robot_bridge_metrics_lock:
            return dict(self._robot_bridge_metrics)

    def set_agent_environment(self, raw: Any) -> Dict[str, Any]:
        """Validate and store latest ``agent_environment`` snapshot. Thread-safe."""
        from mind.environment_context import validate_agent_environment

        norm, err = validate_agent_environment(raw)
        with self._agent_environment_lock:
            if err:
                self.agent_environment = {}
                self.agent_environment_error = err
                self.agent_environment_received_at = time.time()
                return {"ok": False, "error": err}
            self.agent_environment = norm
            self.agent_environment_error = ""
            self.agent_environment_received_at = time.time()
            try:
                self._merge_robot_body_from_normalized(norm)
            except Exception as _rb:
                print(f"[SharedResources] robot_body merge: {_rb}", flush=True)
            return {"ok": True, "fingerprint": norm.get("fingerprint"), "domain": norm.get("domain")}

    def merge_robot_bridge_feedback(self, feedback_raw: Any) -> Dict[str, Any]:
        """Merge on-robot executor feedback into ``extensions.robot_bridge`` and re-store env."""
        from integrations.robot_http_bridge import merge_feedback_into_agent_environment

        with self._agent_environment_lock:
            current = dict(self.agent_environment) if self.agent_environment else {}
        merged, err = merge_feedback_into_agent_environment(current, feedback_raw)
        if err:
            self._robot_bridge_metrics_bump("feedback_posts_error", 1)
            return {"ok": False, "error": err}
        res = self.set_agent_environment(merged)
        if res.get("ok"):
            self._robot_bridge_metrics_bump("feedback_posts_ok", 1)
            ext = merged.get("extensions") if isinstance(merged.get("extensions"), dict) else {}
            rb = ext.get("robot_bridge") if isinstance(ext.get("robot_bridge"), dict) else {}
            rs = rb.get("results")
            n = len(rs) if isinstance(rs, list) else 0
            if n:
                self._robot_bridge_metrics_bump("feedback_result_rows_accepted", n)
        else:
            self._robot_bridge_metrics_bump("feedback_apply_failed", 1)
        return res

    def _merge_robot_body_from_normalized(self, norm: Dict[str, Any]) -> None:
        """Update long-lived robot body state from ``extensions.robot_body``."""
        from mind.robot_body_state import merge_robot_body_readings

        ext = norm.get("extensions")
        if not isinstance(ext, dict):
            ext = {}
        inc = ext.get("robot_body")
        now = time.time()
        eff, ev, stamps = merge_robot_body_readings(
            self._robot_body_effective,
            self._robot_body_ever_observed,
            inc if isinstance(inc, dict) else None,
            now=now,
            last_stamps=self._robot_body_field_stamps,
        )
        self._robot_body_effective = eff
        self._robot_body_ever_observed = ev
        self._robot_body_field_stamps = stamps

    def get_agent_environment_snapshot(self) -> Dict[str, Any]:
        """Copy of the latest environment with merged ``extensions.robot_body`` for cognition."""
        from mind.robot_body_state import build_robot_body_extension

        with self._agent_environment_lock:
            base = dict(self.agent_environment) if self.agent_environment else {}
            ext = dict(base.get("extensions") or {}) if isinstance(base.get("extensions"), dict) else {}
            snap_now = time.time()
            ext["robot_body"] = build_robot_body_extension(
                self._robot_body_effective,
                self._robot_body_ever_observed,
                self._robot_body_field_stamps,
                now=snap_now,
            )
            base["extensions"] = ext
            return base

    def agent_environment_status(self) -> Dict[str, Any]:
        from mind.lab_research import lab_events_snapshot
        from mind.robot_execution_contract import summarize_robot_bridge

        with self._agent_environment_lock:
            env = self.agent_environment
            rb = None
            if env and isinstance(env.get("extensions"), dict):
                rb = env["extensions"].get("robot_bridge")
            return {
                "has_environment": bool(env),
                "domain": env.get("domain") if env else "",
                "fingerprint": env.get("fingerprint", "") if env else "",
                "received_at": self.agent_environment_received_at,
                "last_error": self.agent_environment_error or None,
                "robot_body_defined": bool(
                    self._robot_body_ever_observed and self._robot_body_effective
                ),
                "robot_body_keys": list(self._robot_body_effective.keys())[:32],
                "robot_bridge": summarize_robot_bridge(rb if isinstance(rb, dict) else None),
                "robot_bridge_metrics": self.robot_bridge_metrics_snapshot(),
                "lab_run_id": getattr(self, "lab_run_id", "") or None,
                "lab_experiment_events": lab_events_snapshot(12),
            }

    def autonomy_stimulus_queue_len(self) -> int:
        with self._autonomy_stim_lock:
            return len(self._autonomy_stimulus_queue)

    def enqueue_autonomous_stimulus(
        self,
        text: str,
        source: str = "background",
        target_persona_id: Optional[str] = None,
    ) -> None:
        """Queue a self-initiated cue for personas.

        When the queue is at capacity, duplicates (same fingerprint as an
        existing entry) refresh that entry instead of evicting another.
        Optional *target_persona_id* limits consumption to that persona (plus
        anyone can still not steal: only matching ``pop_autonomous_stimulus``).
        """
        body = str(text).strip()[:600]
        fp = self._autonomy_stimulus_fingerprint(body)
        if not fp:
            return
        with self._autonomy_stim_lock:
            items = self._autonomy_stimulus_queue
            for i, ex in enumerate(items):
                if ex.get("fingerprint") == fp:
                    ex["t"] = time.time()
                    ex["source"] = source
                    ex["text"] = body
                    if target_persona_id:
                        ex["for_persona"] = target_persona_id
                    elif "for_persona" in ex and not target_persona_id:
                        ex.pop("for_persona", None)
                    items.append(items.pop(i))
                    self.autonomy_bump("stimulus_queue_deduped", 1)
                    return
            entry: Dict[str, Any] = {
                "text": body,
                "source": source,
                "t": time.time(),
                "fingerprint": fp,
            }
            if target_persona_id:
                entry["for_persona"] = target_persona_id
            max_q = max(1, self._autonomy_stimulus_queue_max)
            while len(items) >= max_q:
                items.pop(0)
                self.autonomy_bump("stimulus_queue_evicted", 1)
            items.append(entry)

    def peek_autonomous_stimulus(
        self,
        persona_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the next matching stimulus without removing it."""
        with self._autonomy_stim_lock:
            for ex in self._autonomy_stimulus_queue:
                if self._stimulus_matches_persona(ex, persona_id):
                    return dict(ex)
        return None

    @staticmethod
    def _stimulus_matches_persona(
        entry: Dict[str, Any],
        persona_id: Optional[str],
    ) -> bool:
        tgt = entry.get("for_persona")
        if persona_id is None:
            return not tgt
        if not tgt:
            return True
        return tgt == persona_id

    def pop_autonomous_stimulus(
        self,
        persona_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Remove and return the next stimulus visible to this persona.

        Entries without ``for_persona`` are visible to any persona. Targeted
        entries are visible only to the matching *persona_id*.
        """
        with self._autonomy_stim_lock:
            items = self._autonomy_stimulus_queue
            for i, ex in enumerate(items):
                if self._stimulus_matches_persona(ex, persona_id):
                    return dict(items.pop(i))
        return None

    def autonomy_tool_try(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.autonomy_sandbox is None:
            return {"ok": False, "error": "sandbox_uninitialized"}
        return self.autonomy_sandbox.try_execute(name, params, context=self)

    def autonomy_persist_blob(self) -> Dict[str, Any]:
        stim: List[Dict[str, Any]] = []
        with self._autonomy_stim_lock:
            stim = list(self._autonomy_stimulus_queue)
        sb = self.autonomy_sandbox.to_dict() if self.autonomy_sandbox is not None else {}
        return {
            "metrics": self.autonomy_metrics_snapshot(),
            "stimulus_queue": stim,
            "sandbox": sb,
        }

    def autonomy_restore_blob(self, data: Dict[str, Any]) -> None:
        if not data:
            return
        with self._autonomy_metrics_lock:
            m = data.get("metrics")
            if isinstance(m, dict):
                merged: Dict[str, int] = {}
                for k, v in m.items():
                    try:
                        merged[str(k)] = int(v)
                    except (TypeError, ValueError):
                        pass
                self._autonomy_metrics = merged
        with self._autonomy_stim_lock:
            self._autonomy_stimulus_queue.clear()
            max_q = max(1, self._autonomy_stimulus_queue_max)
            for item in data.get("stimulus_queue") or []:
                if isinstance(item, dict) and item.get("text"):
                    ent = dict(item)
                    if "fingerprint" not in ent:
                        ent["fingerprint"] = self._autonomy_stimulus_fingerprint(str(ent["text"]))
                    self._autonomy_stimulus_queue.append(ent)
            while len(self._autonomy_stimulus_queue) > max_q:
                self._autonomy_stimulus_queue.pop(0)
        sb = data.get("sandbox")
        if self.autonomy_sandbox is not None and isinstance(sb, dict):
            self.autonomy_sandbox.from_dict(sb)

    def http_chat_begin(self, depth: Optional[str] = None) -> None:
        with self._http_chat_lock:
            self._http_chat_inflight += 1
            self.signals.append_depth_under_http_lock(depth)

    def http_chat_end(self) -> None:
        with self._http_chat_lock:
            if self._http_chat_inflight > 0:
                self._http_chat_inflight -= 1
                self.signals.pop_depth_under_http_lock()

    @property
    def http_chat_inflight(self) -> int:
        with self._http_chat_lock:
            return self._http_chat_inflight

    def next_cycle(self) -> int:
        """Atomically increment and return the new cycle count."""
        with self._cycle_lock:
            self.cycle_count += 1
            return self.cycle_count

    @contextlib.contextmanager
    def neural_sync(self) -> Iterator[None]:
        """Exclusive section for shared neural modules (train vs forward).

        BackgroundAgent runs ``train_step`` via :mod:`core.training_surface` on encoder,
        composer, appraisal, imagination, etc.; PersonaAgent runs the full cognitive cycle that
        forwards through the same parameters. Hold this lock around training
        batches and around inference so optimizer updates never overlap reads.
        """
        lock = getattr(self, "_neural_lock", None)
        if lock is None:
            # Older / pickled shells without ``__init__`` run — attach lazily.
            lock = threading.RLock()
            self._neural_lock = lock
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    _BOOT_DEADLINE = 30.0
    _MODULE_TIMEOUT = 5.0

    def initialize(self):
        """Heavy initialization -- called by BootAgent.

        Every module construction is timed and logged.  If any single
        module exceeds _MODULE_TIMEOUT it is skipped (replaced by CognitiveNull).
        If the whole init exceeds _BOOT_DEADLINE it aborts remaining
        optional modules and returns what is ready.
        """
        t0 = time.time()
        self.agent_config = load_agent_config()
        _bg0 = self.agent_config.get("background", {})
        try:
            self._autonomy_stimulus_queue_max = max(
                4, min(64, int(_bg0.get("autonomy_stimulus_queue_max", 12)))
            )
        except (TypeError, ValueError):
            self._autonomy_stimulus_queue_max = 12

        def _elapsed():
            return time.time() - t0

        def _load(label, fn, *, optional=False):
            """Timed module load with 5s ceiling and debug logging."""
            if _elapsed() > self._BOOT_DEADLINE:
                if optional:
                    print(f"  [BOOT] SKIP  {label:30s}  (deadline exceeded)", flush=True)
                    return CognitiveNull()
                print(f"  [BOOT] FORCE {label:30s}  (past deadline, required)", flush=True)
            mt = time.time()
            try:
                result = fn()
            except Exception as exc:
                print(f"  [BOOT] FAIL  {label:30s}  {type(exc).__name__}: {exc}", flush=True)
                return CognitiveNull()
            dt = time.time() - mt
            tag = "SLOW!" if dt > self._MODULE_TIMEOUT else "ok"
            print(f"  [BOOT] {tag:5s} {label:30s}  {dt:.2f}s", flush=True)
            return result

        def _bind_manager(label, fn, *, optional=False):
            """Load a mind *Manager; return (manager, engine) or (CognitiveNull, CognitiveNull)."""
            m = _load(label, fn, optional=optional)
            if is_cognitive_null(m):
                return m, m
            return m, m.engine

        # -- resource detection (lazy import) ---------------------------
        def _import_detect():
            from engine.ResourceAdaptiveConfig import detect_resources

            return detect_resources()

        self.resource_config = _load("resource_detection", _import_detect)
        _rc = self.resource_config
        if is_cognitive_null(_rc):
            raise RuntimeError("Cannot detect resources -- aborting boot")
        _skip = set(_rc.cycle.get("skip_modules", []))
        _enc_cfg = _rc.encoder
        _llm_cfg = dict(_rc.llm)
        apply_soul_llm_overrides(
            self.agent_config.get("llm"),
            _llm_cfg,
        )

        _mem_cfg = self.agent_config.get("memory", {})
        if isinstance(_mem_cfg, dict) and _mem_cfg:
            if "seed_context_enabled" in _mem_cfg:
                os.environ.setdefault(
                    "HAROMA_MEMORY_SEED_ENABLED",
                    "1" if _mem_cfg["seed_context_enabled"] else "0",
                )
            if "seed_max_chars" in _mem_cfg:
                try:
                    os.environ.setdefault(
                        "HAROMA_SEED_CONTEXT_MAX_CHARS",
                        str(int(_mem_cfg["seed_max_chars"])),
                    )
                except (TypeError, ValueError):
                    pass
            if "touch_policy" in _mem_cfg:
                os.environ.setdefault(
                    "HAROMA_MEMORY_TOUCH_POLICY",
                    str(_mem_cfg["touch_policy"]),
                )
            if "llm_centric" in _mem_cfg:
                os.environ.setdefault(
                    "HAROMA_LLM_CENTRIC",
                    "1" if _mem_cfg["llm_centric"] else "0",
                )

        _goals_cfg = self.agent_config.get("goals", {})
        if _goals_cfg.get("fifo_input"):
            os.environ.setdefault("HAROMA_GOAL_FIFO_INPUT", "1")
        if _goals_cfg.get("unify_sensor_with_text"):
            os.environ.setdefault("HAROMA_GOAL_UNIFY_SENSOR", "1")
        if _goals_cfg.get("multi_goal_per_cycle"):
            os.environ.setdefault("HAROMA_MULTI_GOAL_PER_CYCLE", "1")
        if "max_cycle_goals" in _goals_cfg:
            try:
                os.environ.setdefault(
                    "HAROMA_MAX_CYCLE_GOALS",
                    str(int(_goals_cfg["max_cycle_goals"] or 3)),
                )
            except (TypeError, ValueError):
                pass
        if "max_actions_per_goal" in _goals_cfg:
            try:
                os.environ.setdefault(
                    "HAROMA_MAX_ACTIONS_PER_GOAL",
                    str(int(_goals_cfg["max_actions_per_goal"] or 2)),
                )
            except (TypeError, ValueError):
                pass

        from core.OrganizationalGoalBoard import OrganizationalGoalBoard

        _og = self.agent_config.get("organizational_goals", {})
        try:
            _cv = int(_og.get("consensus_votes", 2) or 2)
        except (TypeError, ValueError):
            _cv = 2
        try:
            _cttc = int(_og.get("ceo_ticks_to_complete", 3) or 3)
        except (TypeError, ValueError):
            _cttc = 3
        self.goal_board = OrganizationalGoalBoard(
            consensus_votes=_cv,
            ceo_ticks_to_complete=_cttc,
        )

        _train_cfg = _rc.training

        print(
            f"[SharedResources] Resource tier: {_rc.tier} ({_rc.tier_name}) | "
            f"RAM={_rc.hardware['ram_gb']}GB | "
            f"GPU={_rc.hardware['gpu'].get('name') or 'none'} | "
            f"APIs={[k for k, v in _rc.hardware['api_keys'].items() if v]}",
            flush=True,
        )

        # -- compute fabric (first -- modules register on it) ---------
        def _make_fabric():
            from engine.ComputeFabric import ComputeFabric

            return ComputeFabric.init()

        self.fabric = _load("ComputeFabric", _make_fabric)

        # -- neural encoder (triggers cached MiniLM load) --------------
        def _make_encoder():
            from engine.NeuralEncoder import NeuralEncoder

            return NeuralEncoder(
                embed_dim=_enc_cfg.get("embed_dim", 256),
                vocab_size=_enc_cfg.get("vocab_size", 8192),
                force_mode=_enc_cfg.get("mode"),
            )

        self.encoder = _load("NeuralEncoder", _make_encoder)
        try:
            from engine.ModelCache import DEFAULT_SEMANTIC_ENCODER_ID

            self._base_semantic_model_id = (
                os.environ.get("HAROMA_SEMANTIC_ENCODER", "").strip() or DEFAULT_SEMANTIC_ENCODER_ID
            )
        except Exception:
            self._base_semantic_model_id = ""

        # -- core memory -----------------------------------------------
        def _make_memory():
            from core.Memory import MemoryForest

            return MemoryForest(encoder=self.encoder)

        self.memory = _load("MemoryForest", _make_memory)

        # -- managers (lightweight, always loaded) ----------------------
        def _mgr(module, cls_name, *args):
            def _fn():
                mod = __import__(module, fromlist=[cls_name])
                cls = getattr(mod, cls_name)
                return cls(*args)

            return _fn

        self.identity = _load("IdentityManager", _mgr("mind.manager", "IdentityManager"))
        self.goal = _load("GoalManager", _mgr("mind.manager", "GoalManager"))

        def _make_drift():
            from mind.manager import DriftManager as _Cls

            return _Cls(self.memory)

        def _make_collapse():
            from mind.manager import CollapseManager as _Cls

            return _Cls(self.memory)

        def _make_forecast():
            from mind.manager import ForecastManager as _Cls

            return _Cls(self.memory)

        def _make_loop_logger():
            from mind.manager import LoopLoggerManager as _Cls

            return _Cls(self.memory)

        def _make_reflector():
            from mind.manager import ReflectionManager as _Cls

            return _Cls(self.memory)

        self.drift = _load("DriftManager", _make_drift)
        self.collapse = _load("CollapseManager", _make_collapse)
        self.forecast = _load("ForecastManager", _make_forecast)
        self.loop = _load("LoopLoggerManager", _make_loop_logger)
        self.dream_mgr = _load("DreamManager", _mgr("mind.manager", "DreamManager"))
        self.law = _load("LawManager", _mgr("mind.manager", "LawManager"))
        self.myth = _load("MythManager", _mgr("mind.manager", "MythManager"))
        self.value = _load("ValueManager", _mgr("mind.manager", "ValueManager"))
        self.fusion = _load("FusionManager", _mgr("mind.manager", "FusionManager"))
        self.perception = _load("PerceptionManager", _mgr("mind.manager", "PerceptionManager"))
        self.reflector = _load("ReflectionManager", _make_reflector)

        # -- LLM: manager (mind) wraps LLMBackend engine ----------------
        def _make_llm():
            if not _llm_boot_preflight_ok():
                return CognitiveNull()
            from mind.manager import LLMManager

            _env_gpu = os.environ.get("HAROMA_N_GPU_LAYERS", "").strip()
            _n_gpu = (
                int(_env_gpu)
                if _env_gpu.lstrip("-").isdigit()
                else _llm_cfg.get("n_gpu_layers", -1)
            )
            return LLMManager(
                model_path=_llm_cfg.get("local_gguf"),
                n_ctx=_llm_cfg.get("n_ctx", 2048),
                n_gpu_layers=_n_gpu,
                api_provider=_llm_cfg.get("api_provider"),
                api_model=_llm_cfg.get("api_model"),
                api_max_tokens=_llm_cfg.get("api_max_tokens", 512),
                api_temperature=_llm_cfg.get("api_temperature", 0.7),
                reward_replay_lock=self.locks.training.reward_replay,
                use_programmed=bool(_llm_cfg.get("use_programmed")),
            )

        self.llm_manager = _load("LLMManager", _make_llm)
        if is_cognitive_null(self.llm_manager):
            self.llm_backend = self.llm_manager
        else:
            self.llm_backend = self.llm_manager.engine
            if not is_cognitive_null(self.llm_backend):
                _warm = getattr(self.llm_backend, "warmup_local_inference", None)
                if callable(_warm):
                    try:
                        _warm()
                    except Exception as _wexc:
                        print(
                            f"[SharedResources] LLM warmup error: {_wexc}",
                            flush=True,
                        )

        # -- composition -----------------------------------------------
        def _make_composer():
            from engine.LanguageComposer import LanguageComposer

            return LanguageComposer(
                encoder=self.encoder,
                llm_backend=self.llm_backend,
                data_lock=self.locks.training.language_composer_data,
            )

        self.composer = _load("LanguageComposer", _make_composer)

        try:
            from core.VirtualLLMTree import VirtualLLMTree

            self.virtual_llm_tree = VirtualLLMTree(self.memory, self.llm_backend)
        except Exception as exc:
            print(
                f"[SharedResources] VirtualLLMTree init failed: {exc}",
                flush=True,
            )
            self.virtual_llm_tree = None

        # -- engines ----------------------------------------------------
        def _lazy(module, cls_name, **kwargs):
            """Lazy-import factory for zero-arg or keyword-arg constructors."""

            def _fn():
                mod = __import__(module, fromlist=[cls_name])
                return getattr(mod, cls_name)(**kwargs)

            return _fn

        def _mk_curiosity_mgr():
            from mind.manager import CuriosityManager

            return CuriosityManager(encoder=self.encoder)

        self.curiosity_manager, self.curiosity = _bind_manager(
            "CuriosityManager", _mk_curiosity_mgr
        )
        self.action_generator = _load(
            "ActionGenerator",
            lambda: __import__("core.ActionLoop", fromlist=["ActionGenerator"]).ActionGenerator(
                composer=self.composer
            ),
        )
        self.outcome_evaluator = _load(
            "OutcomeEvaluator", _lazy("core.ActionLoop", "OutcomeEvaluator")
        )
        self.action_memory = _load("ActionMemory", _lazy("core.ActionLoop", "ActionMemory"))

        def _mk_metacog_mgr():
            from mind.manager import MetaCognitionManager

            return MetaCognitionManager()

        self.metacognition_manager, self.metacognition = _bind_manager(
            "MetaCognitionManager", _mk_metacog_mgr
        )
        self.drives = _load(
            "HomeostaticSystem", _lazy("core.HomeostaticDrives", "HomeostaticSystem")
        )

        def _mk_temporal_mgr():
            from mind.manager import TemporalManager

            return TemporalManager()

        self.temporal_manager, self.temporal = _bind_manager("TemporalManager", _mk_temporal_mgr)

        def _mk_knowledge_mgr():
            from mind.manager import KnowledgeGraphManager

            return KnowledgeGraphManager()

        self.knowledge_manager, self.knowledge = _bind_manager(
            "KnowledgeGraphManager", _mk_knowledge_mgr
        )

        def _mk_reasoning_mgr():
            from mind.manager import ReasoningManager

            return ReasoningManager(llm_backend=self.llm_backend, law_manager=self.law)

        self.reasoning_manager, self.reasoning = _bind_manager(
            "ReasoningManager", _mk_reasoning_mgr
        )
        self.interlocutor_model = _load(
            "InterlocutorModel", _lazy("core.InterlocutorModel", "InterlocutorModel")
        )

        def _mk_appraisal_mgr():
            from mind.manager import AppraisalManager

            return AppraisalManager()

        self.appraisal_manager, self.appraisal = _bind_manager(
            "AppraisalManager", _mk_appraisal_mgr
        )

        def _mk_modulation_mgr():
            from mind.manager import ModulationManager

            return ModulationManager()

        self.modulation_manager, self.modulation = _bind_manager(
            "ModulationManager", _mk_modulation_mgr
        )
        _embed_dim = self.encoder._embed_dim if not is_cognitive_null(self.encoder) else 384

        def _mk_self_model_mgr():
            from mind.manager import SelfModelManager

            return SelfModelManager(encoder=self.encoder, embed_dim=_embed_dim)

        self.self_model_manager, self.self_model = _bind_manager(
            "SelfModelManager", _mk_self_model_mgr
        )

        def _mk_attention_mgr():
            from mind.manager import AttentionManager

            return AttentionManager()

        self.attention_manager, self.attention = _bind_manager(
            "AttentionManager", _mk_attention_mgr
        )

        def _mk_goal_synth_mgr():
            from mind.manager import GoalSynthesizerManager

            return GoalSynthesizerManager()

        self.goal_synthesizer_manager, self.goal_synthesizer = _bind_manager(
            "GoalSynthesizerManager", _mk_goal_synth_mgr
        )

        def _make_process_gate():
            from engine.ProcessGate import ProcessGate

            return ProcessGate(pending_lock=self.locks.training.process_gate_pending)

        self.process_gate = _load("ProcessGate", _make_process_gate)
        self.backbone = _load(
            "CognitiveBackbone", _lazy("engine.CognitiveBackbone", "CognitiveBackbone")
        )
        self.discourse = _load(
            "DiscourseProcessor", _lazy("core.DiscourseProcessor", "DiscourseProcessor")
        )
        self.grounder = _load(
            "EnvironmentGrounder", _lazy("environment.EnvironmentGrounder", "EnvironmentGrounder")
        )
        self.action_dispatcher = _load(
            "ActionDispatcher", _lazy("environment.ActionDispatcher", "ActionDispatcher")
        )
        self.training_scheduler = _load(
            "TrainingScheduler", _lazy("engine.TrainingScheduler", "TrainingScheduler")
        )

        # -- conditional modules (skip on low-resource or past deadline)
        def _make_dream_consolidator():
            from core.DreamConsolidator import DreamConsolidator

            return DreamConsolidator(self.memory, encoder=self.encoder)

        def _make_imagination_mgr():
            from mind.manager import ImaginationManager

            return ImaginationManager(
                encoder=self.encoder,
                llm_backend=self.llm_backend,
                buffer_lock=self.locks.training.imagination_buffer,
            )

        self.dream_consolidator = (
            _load("DreamConsolidator", _make_dream_consolidator, optional=True)
            if "dream_consolidator" not in _skip
            else CognitiveNull()
        )
        if "counterfactual" not in _skip:

            def _mk_cf_mgr():
                from mind.manager import CounterfactualManager

                return CounterfactualManager()

            self.counterfactual_manager, self.counterfactual = _bind_manager(
                "CounterfactualManager", _mk_cf_mgr, optional=True
            )
        else:
            self.counterfactual_manager = CognitiveNull()
            self.counterfactual = CognitiveNull()
        if "imagination" not in _skip:
            self.imagination_manager, self.imagination = _bind_manager(
                "ImaginationManager", _make_imagination_mgr, optional=True
            )
        else:
            self.imagination_manager = CognitiveNull()
            self.imagination = CognitiveNull()
        self.mental_simulator = (
            _load(
                "MentalSimulator", _lazy("engine.MentalSimulator", "MentalSimulator"), optional=True
            )
            if "mental_simulator" not in _skip
            else CognitiveNull()
        )
        self.arch_searcher = (
            _load(
                "ArchitectureSearcher",
                _lazy("engine.ArchitectureSearcher", "ArchitectureSearcher"),
                optional=True,
            )
            if "arch_searcher" not in _skip
            else CognitiveNull()
        )

        # -- register training modules ----------------------------------
        if _train_cfg.get("enabled", True):
            _ts = self.training_scheduler
            if not is_cognitive_null(_ts):
                for name, interval in [
                    ("encoder", 3),
                    ("backbone", 3),
                    ("attention", 5),
                    ("process_gate", 5),
                    ("self_model", 5),
                    ("appraisal", 5),
                    ("modulation", 8),
                    ("goal_synth", 8),
                    ("imagination", 8),
                    ("metacog", 5),
                    ("composer", 5),
                    ("generative", 5),
                    ("counterfactual", 8),
                    ("grounder", 10),
                    ("mental_sim", 5),
                    ("arch_search", 10),
                    ("llm_reward", 5),
                ]:
                    _ts.register_module(name, base_interval=interval)

        # -- persistence & soul -----------------------------------------
        def _make_persistence():
            from core.Persistence import CognitivePersistence

            return CognitivePersistence()

        def _make_soul_binder():
            from core.SoulBinder import SoulBinder

            return SoulBinder()

        self.persistence = _load("CognitivePersistence", _make_persistence)
        self.soul_binder = _load("SoulBinder", _make_soul_binder)

        # -- X7 layers (after memory is ready) --------------------------
        def _make_memory_core():
            from core.MemoryCore import MemoryCore

            return MemoryCore(self.memory)

        def _make_reconciliation():
            from core.Reconciliation import SymbolicReconciliationEngine

            return SymbolicReconciliationEngine(self.memory_core)

        self.memory_core = _load("MemoryCore", _make_memory_core)
        self.reconciliation = _load("Reconciliation", _make_reconciliation)
        self.symbolic_queue = _load("SymbolicQueue", _lazy("core.SymbolicQueue", "SymbolicQueue"))
        self.fingerprint_engine = _load(
            "FingerprintEngine", _lazy("core.SymbolicQueue", "SymbolicFingerprintEngine")
        )
        self.organ_registry = _load("OrganRegistry", _lazy("core.OrganRegistry", "OrganRegistry"))

        # Sandbox before persistence.load so autonomy_state restore applies.
        try:
            from core.AutonomySandbox import AutonomySandbox

            self.autonomy_sandbox = AutonomySandbox()
        except Exception:
            self.autonomy_sandbox = None

        # -- soul bind -> persistence -> reassert -----------------------
        self.boot_results["soul"] = _load("soul_bind", lambda: self.soul_binder.bind(self))
        self.boot_results["persistence"] = _load(
            "persistence_load", lambda: self.persistence.load(self)
        )
        try:
            comp = self.composer
            if (
                comp is not None
                and not is_cognitive_null(comp)
                and hasattr(comp, "bootstrap_offline_training_assets")
            ):
                bst = comp.bootstrap_offline_training_assets()
                if bst.get("gen_seeded") or bst.get("vocab_merged"):
                    print(
                        f"[SharedResources] Offline learning bootstrap: "
                        f"vocab+{bst.get('vocab_merged', 0)} "
                        f"gen_seed={bst.get('gen_seeded', 0)} "
                        f"vocab_size={bst.get('vocab_size_after')}",
                        flush=True,
                    )
        except Exception as exc:
            print(
                f"[SharedResources] Offline bootstrap skipped: {exc}",
                flush=True,
            )
        self.boot_results["soul_reassert"] = _load(
            "soul_reassert", lambda: self.soul_binder.reassert(self)
        )
        self.boot_results["fabric"] = (
            self.fabric.stats() if not is_cognitive_null(self.fabric) else {}
        )

        # -- organ registry (after everything is built) -----------------
        self._register_organs()

        self.boot_time = time.time() - t0
        try:
            from mind.lab_research import init_lab_run

            _lr = init_lab_run(self)
            print(f"[SharedResources] lab_run_id={_lr}", flush=True)
        except Exception as _lab_e:
            print(f"[SharedResources] lab research init: {_lab_e}", flush=True)
        print(
            f"[SharedResources] Initialized in {self.boot_time:.1f}s | "
            f"tier={_rc.tier_name} | "
            f"llm={getattr(self.llm_backend, 'model_name', None) or 'none'}",
            flush=True,
        )

    def _register_organs(self):
        """Register live module references into the organ registry."""
        _map = {
            "llm_manager": self.llm_manager,
            "knowledge": self.knowledge,
            "knowledge_manager": self.knowledge_manager,
            "reasoning": self.reasoning,
            "reasoning_manager": self.reasoning_manager,
            "encoder": self.encoder,
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
            "training_scheduler": self.training_scheduler,
            "arch_searcher": self.arch_searcher,
            "soul_binder": self.soul_binder,
            "identity": self.identity,
            "interlocutor_model": self.interlocutor_model,
            "mental_simulator": self.mental_simulator,
            "memory": self.memory,
            "memory_core": self.memory_core,
            "persistence": self.persistence,
            "loop_logger": self.loop,
            "dream_consolidator": self.dream_consolidator,
            "dream": self.dream_mgr,
            "law": self.law,
            "myth": self.myth,
            "value": self.value,
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
            "symbolic_queue": self.symbolic_queue,
            "fingerprint_engine": self.fingerprint_engine,
            "composer": self.composer,
            "virtual_llm_tree": self.virtual_llm_tree,
            "action_generator": self.action_generator,
            "fabric": self.fabric,
        }
        for name, ref in _map.items():
            if ref is not None and not is_cognitive_null(ref):
                self.organ_registry.register_module(name, ref)

    def summary(self) -> Dict[str, Any]:
        mem = self.memory
        mem_nodes = 0
        if mem is not None and not is_cognitive_null(mem):
            try:
                mem_nodes = mem.count_nodes()
            except Exception:
                pass
        return {
            "boot_time": self.boot_time,
            "cycle_count": self.cycle_count,
            "resource_tier": getattr(self.resource_config, "tier_name", "?"),
            "memory_nodes": mem_nodes,
            "organs": (self.organ_registry.summary() if self.organ_registry else {}),
            "boot_results": self.boot_results,
            "autonomy_metrics": self.autonomy_metrics_snapshot(),
            "autonomy_stimulus_queue_len": self.autonomy_stimulus_queue_len(),
            "agent_environment": self.agent_environment_status(),
        }

    def resolve_semantic_model_id(self, persona_id: str) -> str:
        """HF model id for :class:`engine.NeuralEncoder` for this persona.

        Order: ``initial_personas[].semantic_encoder`` → env
        ``HAROMA_SEMANTIC_ENCODER_<PERSONA>`` → ``HAROMA_SEMANTIC_ENCODER`` → default MiniLM.
        """
        try:
            from engine.ModelCache import DEFAULT_SEMANTIC_ENCODER_ID
        except Exception:
            DEFAULT_SEMANTIC_ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"

        pid = (persona_id or "").strip()
        for p in self.agent_config.get("initial_personas", []) or []:
            if str(p.get("id", "")).strip() == pid:
                mid = (p.get("semantic_encoder") or "").strip()
                if mid:
                    return mid
                break
        suf = re.sub(r"[^0-9a-zA-Z]+", "_", pid).strip("_").upper() or "DEFAULT"
        env_key = f"HAROMA_SEMANTIC_ENCODER_{suf}"
        v = str(os.environ.get(env_key, "") or "").strip()
        if v:
            return v
        return os.environ.get("HAROMA_SEMANTIC_ENCODER", "").strip() or DEFAULT_SEMANTIC_ENCODER_ID

    def encoder_for(self, persona_id: str):
        """Per-persona :class:`engine.NeuralEncoder` (cached by resolved model id).

        Matches the global ``self.encoder`` when the resolved id equals the deployment
        default (no extra weights). Background training still uses ``self.encoder``.
        """
        if self.encoder is None or is_cognitive_null(self.encoder):
            return self.encoder
        mid = self.resolve_semantic_model_id(persona_id)
        if mid == getattr(self, "_base_semantic_model_id", "") and not is_cognitive_null(
            self.encoder
        ):
            return self.encoder
        cache = self._persona_encoder_cache
        if mid not in cache:
            _rc = self.resource_config
            _enc_cfg = (
                _rc.encoder
                if _rc is not None and not is_cognitive_null(_rc) and hasattr(_rc, "encoder")
                else {}
            )
            if not isinstance(_enc_cfg, dict):
                _enc_cfg = {}
            from engine.NeuralEncoder import NeuralEncoder

            cache[mid] = NeuralEncoder(
                embed_dim=int(_enc_cfg.get("embed_dim", 256)),
                vocab_size=int(_enc_cfg.get("vocab_size", 8192)),
                force_mode=_enc_cfg.get("mode"),
                semantic_model_id=mid,
            )
        return cache[mid]
