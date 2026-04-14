"""
BootAgent -- system initializer and supervisor.

Sequence (v6):
  1. Create SharedResources (heavy init: resource detection, all modules,
     soul bind, persistence load, soul reassert)
  2. Create MessageBus
  3. Spawn InputAgent
  4. Spawn TrueSelfAgent (sole input receiver, executive consciousness)
  5. Spawn BackgroundAgent
  6. Spawn initial PersonaAgent(s) from soul/agents.json config
  7. Enter supervisor loop: monitor agent health, restart crashed agents
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from agents.base import BaseAgent
from agents.message_bus import MessageBus
from agents.shared_resources import SharedResources

if TYPE_CHECKING:
    from agents.input_agent import InputAgent
    from agents.trueself_agent import TrueSelfAgent
    from agents.background_agent import BackgroundAgent
    from agents.persona_agent import PersonaAgent


class BootAgent(BaseAgent):
    """Initializes Elarion and supervises all child agents."""

    AGENT_TYPE = "boot"

    def __init__(self, tick_interval: float = 10.0):
        super().__init__(
            agent_id="boot",
            tick_interval=tick_interval,
        )
        self.shared: Optional[SharedResources] = None
        self.bus: Optional[MessageBus] = None

        self.input_agent: Optional[InputAgent] = None
        self.trueself_agent: Optional[TrueSelfAgent] = None
        self.background_agent: Optional[BackgroundAgent] = None
        self.action_agent: Optional[Any] = None
        self.persona_agents: List[PersonaAgent] = []
        self._all_agents: List[BaseAgent] = []
        self._persona_lock = threading.Lock()
        self._shutting_down = False

    # -- boot (called once, before start()) ----------------------------

    _BOOT_DEADLINE = 30.0

    def boot(self) -> SharedResources:
        """Run the full initialization sequence and return shared resources.

        Hard deadline: if boot exceeds _BOOT_DEADLINE seconds the remaining
        optional steps are skipped.  Every step is timed and logged.
        """
        print("[BootAgent] === Elarion System Boot ===", flush=True)
        t0 = time.time()

        def _elapsed():
            return time.time() - t0

        def _step(label):
            print(f"  [BOOT] {label} @ {_elapsed():.1f}s", flush=True)

        # 1. SharedResources (heavy init -- has its own per-module timing)
        _step("SharedResources.initialize()")
        self.shared = SharedResources()
        self.shared.initialize()
        _step(f"SharedResources done ({_elapsed():.1f}s)")

        if _elapsed() > self._BOOT_DEADLINE:
            print(
                f"  [BOOT] WARNING: SharedResources alone took "
                f"{_elapsed():.1f}s (>{self._BOOT_DEADLINE}s deadline)",
                flush=True,
            )

        # 2. MessageBus
        cfg = self.shared.agent_config
        # input.dead_letter_timeout_ms: passed through for API compat; see MessageBus.
        dead_letter_ms = cfg.get("input", {}).get("dead_letter_timeout_ms", 2000)
        self.bus = MessageBus(dead_letter_timeout_ms=dead_letter_ms)

        # 3. Spawn InputAgent
        from agents.input_agent import InputAgent

        _step("Spawning InputAgent")
        input_interval = cfg.get("input", {}).get("tick_interval", 0.12)
        _itv_raw = str(os.environ.get("HAROMA_INPUT_TICK_INTERVAL_SEC", "") or "").strip()
        if _itv_raw:
            try:
                input_interval = max(0.02, float(_itv_raw))
            except (TypeError, ValueError):
                print(
                    "[BootAgent] HAROMA_INPUT_TICK_INTERVAL_SEC invalid — using config",
                    flush=True,
                )
        self.input_agent = InputAgent(
            shared=self.shared,
            bus=self.bus,
            tick_interval=input_interval,
        )
        self._all_agents.append(self.input_agent)

        # 4. Spawn TrueSelfAgent
        from agents.trueself_agent import TrueSelfAgent

        _step("Spawning TrueSelfAgent")
        trueself_cfg = cfg.get("trueself", {})
        self.trueself_agent = TrueSelfAgent(
            shared=self.shared,
            bus=self.bus,
            trueself_config=trueself_cfg,
        )
        self._all_agents.append(self.trueself_agent)

        self.shared.persist_emotion = self.trueself_agent.emotion
        self.shared.persist_narrative_source = self.trueself_agent
        pend_e = getattr(self.shared, "_pending_emotion_restore", None)
        if pend_e is not None:
            self.shared._pending_emotion_restore = None
            try:
                from core.Persistence import CognitivePersistence

                CognitivePersistence.apply_emotion_snapshot(self.trueself_agent.emotion, pend_e)
            except Exception as exc:
                print(
                    f"[BootAgent] Deferred emotion restore failed: {exc}",
                    flush=True,
                )
        pend_n = getattr(self.shared, "_pending_narrative_restore", None)
        if pend_n is not None:
            self.trueself_agent._narrative_buffer = list(pend_n)
            self.shared._pending_narrative_restore = None

        # 5. Spawn BackgroundAgent
        from agents.background_agent import BackgroundAgent

        _step("Spawning BackgroundAgent")
        bg_interval = cfg.get("background", {}).get("tick_interval", 5.0)
        self.background_agent = BackgroundAgent(
            shared=self.shared,
            bus=self.bus,
            boot_agent=self,
            tick_interval=bg_interval,
        )
        self._all_agents.append(self.background_agent)

        # 5b. Spawn ActionAgent (CEO — executes board mandate)
        from agents.action_agent import ActionAgent

        _step("Spawning ActionAgent")
        _act_cfg = cfg.get("action_agent", {})
        _act_interval = float(_act_cfg.get("tick_interval", 1.0))
        self.action_agent = ActionAgent(
            shared=self.shared,
            bus=self.bus,
            tick_interval=_act_interval,
        )
        self._all_agents.append(self.action_agent)

        # 6. Spawn initial PersonaAgents
        from agents.persona_agent import PersonaAgent

        _step("Spawning PersonaAgents")
        for persona_cfg in cfg.get("initial_personas", []):
            try:
                persona = PersonaAgent(
                    persona_config=persona_cfg,
                    shared=self.shared,
                    bus=self.bus,
                )
                self.persona_agents.append(persona)
                self._all_agents.append(persona)
            except Exception as exc:
                print(
                    f"[BootAgent] Failed to create persona {persona_cfg.get('name', '?')}: {exc}",
                    flush=True,
                )

        # 7. Wire boot_agent reference
        for agent in self._all_agents:
            if hasattr(agent, "set_boot_agent"):
                agent.set_boot_agent(self)

        # 7b. Restore personality snapshots from persistence
        if self.shared.persistence and self.shared._personality_profiles:
            try:
                res = self.shared.persistence.load_personality_profiles(
                    self.shared._personality_profiles
                )
                print(f"[BootAgent] Personality load: {res}", flush=True)
            except Exception as exc:
                print(f"[BootAgent] Personality load error: {exc}", flush=True)

        # 8. Pre-warm semantic index (fully non-blocking)
        self._prewarm_memory_index()

        elapsed = time.time() - t0
        status = "OK" if elapsed <= self._BOOT_DEADLINE else "SLOW"
        print(
            f"[BootAgent] Boot {status} in {elapsed:.1f}s "
            f"(deadline={self._BOOT_DEADLINE}s) | "
            f"agents: 1 input + 1 trueself + 1 background + 1 action + "
            f"{len(self.persona_agents)} persona(s)",
            flush=True,
        )
        return self.shared

    # -- pre-warm memory index (fully non-blocking) --------------------

    def _prewarm_memory_index(self):
        """Mark semantic index clean at boot -- all rebuilds deferred.

        The BackgroundAgent will incrementally rebuild TF-IDF and dense
        indexes during its normal ticks.  Boot does nothing blocking.
        """
        mem = self.shared.memory
        if mem is None:
            return
        si = mem.semantic_index
        if si is None:
            return
        n_nodes = len(si.nodes)
        print(
            f"  [BOOT] Semantic index: {n_nodes} nodes | all rebuilds deferred to BackgroundAgent",
            flush=True,
        )
        si._dirty = False

    # -- start all agents ----------------------------------------------

    def start_all(self):
        """Start every child agent, then start the supervisor loop."""
        if self.shared.persistence:
            if not self.shared.persistence.wait_for_deferred(timeout=60.0):
                print("[BootAgent] WARNING: deferred loads did not complete within 60s", flush=True)
        with self._persona_lock:
            agents = list(self._all_agents)
        try:
            for agent in agents:
                agent.start()
            self.start()
        except Exception as exc:
            print(
                f"[BootAgent] start_all partial failure: {exc} — stopping already-started agents",
                flush=True,
            )
            self.stop_all()
            raise

    # -- stop all agents -----------------------------------------------

    def stop_all(self):
        """Gracefully stop all agents.

        Phase 1: signal every agent to stop under its lifecycle lock.
        Phase 2: join all threads.
        This prevents agents from generating new work while others drain.
        """
        self._shutting_down = True
        with self._persona_lock:
            agents = list(self._all_agents)
        all_agents = [self] + agents

        for agent in all_agents:
            with agent._lifecycle_lock:
                agent._running = False
                agent._stop_event.set()

        for agent in reversed(all_agents):
            if agent._thread is not None:
                join_timeout = max(agent.tick_interval * 3, 5.0)
                agent._thread.join(timeout=join_timeout)
            print(f"[{agent.agent_type}:{agent.agent_id}] stopped", flush=True)

    # -- supervisor tick -----------------------------------------------

    def _tick(self):
        """Health check: restart any crashed agents."""
        if self._shutting_down:
            return
        with self._persona_lock:
            agents = list(self._all_agents)
        for agent in agents:
            if not agent.is_alive():
                print(
                    f"[BootAgent] Agent {agent.agent_type}:{agent.agent_id} "
                    f"found dead, restarting...",
                    flush=True,
                )
                try:
                    agent.start()
                except Exception as exc:
                    print(
                        f"[BootAgent] Failed to restart {agent.agent_type}:{agent.agent_id}: {exc}",
                        flush=True,
                    )

        # Periodic claim GC
        if self.bus and self._tick_count % 6 == 0:
            self.bus.gc_claims()

    # -- dynamic persona spawning (called by BackgroundAgent) ----------

    def spawn_persona(self, persona_config: Dict[str, Any]) -> Optional[PersonaAgent]:
        """Spawn a new persona agent at runtime."""
        from agents.persona_agent import PersonaAgent

        max_personas = self.shared.agent_config.get("max_personas", 5)

        with self._persona_lock:
            if self._shutting_down:
                return None
            if len(self.persona_agents) >= max_personas:
                print(
                    f"[BootAgent] Cannot spawn persona: at max ({max_personas})",
                    flush=True,
                )
                return None
            persona = PersonaAgent(
                persona_config=persona_config,
                shared=self.shared,
                bus=self.bus,
            )
            persona.set_boot_agent(self)
            self.persona_agents.append(persona)
            self._all_agents.append(persona)
        persona.start()

        print(
            f"[BootAgent] Spawned new persona: {persona.agent_id} "
            f"(total: {len(self.persona_agents)})",
            flush=True,
        )
        return persona

    # -- persona listing (used by router) ------------------------------

    def get_persona_ids(self) -> List[str]:
        """Return IDs of all live persona agents."""
        with self._persona_lock:
            return [p.agent_id for p in self.persona_agents if p.is_alive()]

    # -- introspection -------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        base = super().stats()
        with self._persona_lock:
            agents = list(self._all_agents)
            persona_count = len(self.persona_agents)

        def _safe_stats(agent):
            try:
                return agent.stats()
            except Exception as exc:
                return {"agent_id": agent.agent_id, "error": str(exc)}

        base["child_agents"] = [_safe_stats(a) for a in agents]
        base["persona_count"] = persona_count
        if self.bus:
            base["message_bus"] = self.bus.stats()
        return base

    # -- save on shutdown ----------------------------------------------

    def save_and_shutdown(self):
        """Stop all agents first, then persist state.

        Saves on the current thread so :func:`atexit` handlers and interpreter
        shutdown never spawn a worker thread (disallowed during shutdown on
        Windows).
        """
        self.stop_all()
        if self.shared and self.shared.persistence:
            try:
                _t0 = time.perf_counter()
                self.shared.persistence.save(self.shared)
                self.shared.persistence.write_bulk_cache()
                _dt = time.perf_counter() - _t0
                try:
                    _cm = getattr(self.shared, "cognitive_metrics", None)
                    if _cm is not None:
                        _cm.record_persistence_save_sec(_dt)
                except Exception:
                    pass
                print("[BootAgent] State saved on shutdown.", flush=True)
            except Exception as exc:
                print(f"[BootAgent] Save on shutdown failed: {exc}", flush=True)
