"""
BackgroundAgent -- Elarion's subconscious.

Runs on a slower cadence than persona agents, handling everything that
does not need to happen in the critical path of a conversation:

  - Dream consolidation
  - Reconciliation (merge divergent persona branches into common tree)
  - Goal synthesis (cross-persona goal discovery)
  - All training steps (encoder, backbone, attention, reward/VW, etc.)
  - Periodic finetune JSONL flush + optional memory ``train_export`` snapshots
  - Persistence saves
  - Narrative self updates
  - Architecture search
  - Dynamic persona spawning when divergence exceeds threshold
  - Dead-letter queue processing
  - Allowlisted web fetch → memory (optional ``web_learn`` in config); each fetch
    tick logs ``[WebLearn] …`` (disable with ``HAROMA_WEB_LEARN_LOG=0``, detail
    lines with ``HAROMA_WEB_LEARN_VERBOSE=1``). Boot-only config summary if
    ``HAROMA_WEB_LEARN_LOG=1``.

  - ``HAROMA_BG_DEFER_TRAINING_CAP_SEC`` (optional): when defer-on-chat is enabled,
    still run training at most once per this many seconds even while HTTP chat is
    in flight (async clients may hold ``http_chat_inflight`` for a long time).

  - Recall: ``HAROMA_WEB_LEARN_INJECT_MODE`` / ``HAROMA_WEB_LEARN_INJECT_MAX`` (see
    ``core/chat_recall_policy``) control when crawled web text is merged into recall.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

_BG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DIALOG_TRAINING_JSON = os.path.join(_BG_ROOT, "data", "training", "dialog_training.json")
_DIALOG_LINE_CACHE: Optional[List[str]] = None

from agents.base import BaseAgent
from agents.message_bus import Message, MessageBus
from utils.coerce_bool import env_flag

from core.Memory import MemoryNode
from core.VirtualLLMTree import (
    LLM_BRANCH_LEARNING,
    LLM_TREE_NAME,
)
from core.MemoryCore import AGENT_PREFIX
from engine.WebLearnCrawler import WebLearnCrawler
from agents.background_cadence import BackgroundCadence
from core.training_surface import build_background_train_map

if TYPE_CHECKING:
    from agents.shared_resources import SharedResources
    from agents.boot_agent import BootAgent

from core.cognitive_null import is_cognitive_null


def _is_real(obj) -> bool:
    """Return False for None or :class:`~core.cognitive_null.CognitiveNull`."""
    if obj is None:
        return False
    if is_cognitive_null(obj):
        return False
    if hasattr(type(obj), "_is_null_stub"):
        return False
    return True


class BackgroundAgent(BaseAgent):
    """Subconscious processing -- dreams, reconciliation, training."""

    AGENT_TYPE = "background"

    def __init__(
        self,
        shared: SharedResources,
        bus: MessageBus,
        boot_agent: Optional[BootAgent] = None,
        tick_interval: float = 5.0,
    ):
        super().__init__(
            agent_id="background",
            shared=shared,
            bus=bus,
            tick_interval=tick_interval,
        )
        self._boot_agent = boot_agent

        # Config
        bg_cfg = shared.agent_config.get("background", {})
        self._reconcile_every = max(1, bg_cfg.get("reconcile_every_n_ticks", 2))
        self._dream_every = max(1, bg_cfg.get("dream_every_n_ticks", 5))
        # Deep consolidate can take minutes on huge MemoryForest; 0 = no cap.
        try:
            self._dream_consolidate_max_nodes = int(
                bg_cfg.get("dream_consolidate_max_nodes", 20000)
            )
        except (TypeError, ValueError):
            self._dream_consolidate_max_nodes = 20000
        self._dream_skip_heavy_warned = False
        self._training_enabled = bg_cfg.get("training_enabled", True)
        if os.environ.get("HAROMA_BENCH_DISABLE_BG_TRAINING", "").lower() in ("1", "true", "yes"):
            self._training_enabled = False
            print(
                "[BackgroundAgent] training disabled (HAROMA_BENCH_DISABLE_BG_TRAINING)",
                flush=True,
            )
        # Continuous learning: flush finetune JSONL + optional memory export (env overrides soul).
        _ft_raw = str(os.environ.get("HAROMA_FINETUNE_FLUSH_TICKS", "") or "").strip()
        if _ft_raw != "":
            try:
                self._finetune_flush_every = max(0, int(_ft_raw))
            except (TypeError, ValueError):
                self._finetune_flush_every = 0
        else:
            try:
                self._finetune_flush_every = max(
                    0, int(bg_cfg.get("finetune_flush_every_n_ticks", 12) or 12)
                )
            except (TypeError, ValueError):
                self._finetune_flush_every = 12
        _me_raw = str(os.environ.get("HAROMA_MEMORY_TRAINING_EXPORT_TICKS", "") or "").strip()
        if _me_raw != "":
            try:
                self._memory_train_export_every = max(0, int(_me_raw))
            except (TypeError, ValueError):
                self._memory_train_export_every = 0
        else:
            try:
                self._memory_train_export_every = max(
                    0, int(bg_cfg.get("memory_training_export_every_n_ticks", 0) or 0)
                )
            except (TypeError, ValueError):
                self._memory_train_export_every = 0
        self._spawn_threshold = bg_cfg.get("spawn_on_divergence_threshold", 0.7)
        self._autonomy_enabled = bg_cfg.get("autonomy_enabled", True)
        self._autonomy_every = max(3, int(bg_cfg.get("autonomy_initiative_every_n_ticks", 7)))
        self._last_autonomy_fingerprint: Optional[str] = None
        self._autonomy_intent_count = 0
        self._autonomy_stim_assign_rr = 0

        wl_cfg = bg_cfg.get("web_learn")
        self._web_crawler = WebLearnCrawler(wl_cfg if isinstance(wl_cfg, dict) else {})
        self._cadence = BackgroundCadence(shared)
        if self._web_crawler.enabled and (
            not self._web_crawler.seed_urls or not self._web_crawler.allowed_hosts
        ):
            print(
                "[BackgroundAgent] web_learn enabled but incomplete — set seed_urls "
                "and allowed_hosts under background.web_learn in soul/agents.json",
                flush=True,
            )
        elif env_flag("HAROMA_WEB_LEARN_LOG", False) and self._web_crawler.enabled:
            print(
                f"[BackgroundAgent] web_learn ready: hosts={len(self._web_crawler.allowed_hosts)} "
                f"seeds={len(self._web_crawler.seed_urls)} every_n_ticks="
                f"{self._web_crawler.every_n_ticks}",
                flush=True,
            )

        # Per-tick state
        self._last_reconcile_result: Dict[str, Any] = {}
        self._last_dream_result: Dict[str, Any] = {}
        self._last_training_losses: Dict[str, float] = {}
        self._last_finetune_flush_n: int = 0
        self._last_memory_export_n: int = 0

        # Cognitive loop buffers (dream <-> goal <-> dialogue <-> experience)
        self._dream_context: Dict[str, Any] = {}
        self._log_once_no_dream_consolidator = False
        self._log_once_no_dreamcore = False
        self._dialogue_insights: List[str] = []
        self._outcome_window: List[float] = []
        self._OUTCOME_WINDOW_SIZE = 20

        # Subscribe to background channel + dialogue insights
        self.bus.subscribe("background", self.agent_id)
        self.bus.subscribe("dialogue_insight", self.agent_id)

    def _tick(self):
        s = self.shared
        tc = self._tick_count
        _t_tick0 = time.time()
        _hb = os.environ.get("HAROMA_BG_HEARTBEAT", "1") not in ("0", "false", "no")
        _dream_urgent = _is_real(s.drives) and s.drives.should_dream()
        _dream_periodic = tc % self._dream_every == 0
        _do_web = self._cadence.should_run_web_learn(tc, self._web_crawler)
        _do_goal = tc % 3 == 0

        # First line every wake — must run *before* reconcile/dream/training.
        # A single long neural_sync around all training kept _tick() from finishing,
        # so the old end-of-tick heartbeat never ran and the next tick never started.
        if _hb:
            print(
                f"[BackgroundAgent] tick_begin n={tc} global_cycle={s.cycle_count} "
                f"sched dream={_dream_periodic or _dream_urgent} goal={_do_goal} "
                f"web={_do_web} train={self._training_enabled}",
                flush=True,
            )

        _tick_err: Optional[str] = None
        try:
            # -- 1. Process dead-letter messages ---------------------------
            self._process_dead_letters()

            # -- 2. Reconciliation (merge persona branches) ----------------
            if tc % self._reconcile_every == 0:
                self._reconcile()

            # -- 3. Dream consolidation ------------------------------------
            if _dream_periodic or _dream_urgent:
                self._dream()

            # -- 4. Goal synthesis -----------------------------------------
            if _do_goal:
                self._goal_synthesis()

            # -- 5. Training steps (often the long pole; log boundaries) ---
            if self._training_enabled:
                _t_tr = time.time()
                if _hb:
                    print(
                        f"[BackgroundAgent] training_begin n={tc}",
                        flush=True,
                    )
                try:
                    self._run_training()
                finally:
                    if _hb:
                        print(
                            f"[BackgroundAgent] training_end n={tc} dt={time.time() - _t_tr:.2f}s",
                            flush=True,
                        )

            # -- 5a. Finetune flush + memory training export (online / continuous) ---
            self._maybe_flush_training_artifacts(tc)

            # -- 5b. Web learn (allowlist fetch → memory) -------------------
            if _do_web:
                self._run_web_learn()

            # -- 6. Persistence save ---------------------------------------
            if s.persistence and s.persistence.should_save(s.cycle_count):
                self._save()

            # -- 7. Narrative update ---------------------------------------
            if tc % 4 == 0:
                self._update_narrative()

            # -- 8. Architecture search ------------------------------------
            if tc % 6 == 0:
                self._arch_search()

            # -- 9. Dynamic persona spawning check -------------------------
            if tc % 10 == 0 and tc > 0:
                self._check_spawn_persona()

            # -- 10. Initiate inner dialogue on divergence -----------------
            if tc % 8 == 0 and tc > 0:
                self._initiate_inner_dialogue()

            # -- 11. Collect dialogue insights from bus ---------------------
            self._collect_dialogue_insights()

            # -- 12. Update outcome window from personas -------------------
            self._update_outcome_window()

            # -- 13. Autonomous initiative (memory commitments, no user msg) -
            if self._autonomy_enabled and tc > 0 and tc % self._autonomy_every == 0:
                self._autonomous_initiative_tick()

        except Exception as exc:
            _tick_err = f"{type(exc).__name__}: {exc}"
            print(
                f"[BackgroundAgent] tick_error n={tc} {_tick_err}",
                flush=True,
            )
            raise
        finally:
            if _hb:
                _dt = time.time() - _t_tick0
                _suffix = f" ERROR({_tick_err})" if _tick_err else ""
                print(
                    f"[BackgroundAgent] tick_end n={tc} global_cycle={s.cycle_count} "
                    f"elapsed={_dt:.2f}s{_suffix}",
                    flush=True,
                )

    # -- cognitive loop helpers ------------------------------------------

    def _collect_dialogue_insights(self):
        """Drain dialogue_insight messages from the bus into the buffer."""
        messages = self.poll()
        for msg in messages:
            if msg.channel == "dialogue_insight":
                text = ""
                if isinstance(msg.content, dict):
                    text = msg.content.get("text", "")
                elif isinstance(msg.content, str):
                    text = msg.content
                if text and len(text) > 5:
                    self._dialogue_insights.append(text[:200])
                    if len(self._dialogue_insights) > 10:
                        self._dialogue_insights = self._dialogue_insights[-10:]
                    print(
                        f"[BackgroundAgent] Dialogue insight: {text[:140]}"
                        f"{'…' if len(text) > 140 else ''}",
                        flush=True,
                    )

    def _update_outcome_window(self):
        """Maintain a rolling window of persona outcome scores.

        Only appends scores from personas whose cycle count has advanced
        since the last sample (avoids inflating the window with stale data).
        """
        if not self._boot_agent:
            return
        for p in self._boot_agent.persona_agents:
            cc = getattr(p, "_cycle_count", 0)
            last_sampled = getattr(p, "_bg_last_sampled_cycle", 0)
            if cc > last_sampled and hasattr(p, "_prev_outcome_score"):
                self._outcome_window.append(p._prev_outcome_score)
                p._bg_last_sampled_cycle = cc
        if len(self._outcome_window) > self._OUTCOME_WINDOW_SIZE:
            self._outcome_window = self._outcome_window[-self._OUTCOME_WINDOW_SIZE :]

    def _get_outcome_trend(self) -> Dict[str, Any]:
        """Compute outcome trend from the rolling window."""
        w = self._outcome_window
        if len(w) < 4:
            return {"avg": 0.5, "trend": 0.0, "failure_streak": 0}
        avg = sum(w) / len(w)
        half = len(w) // 2
        first_half = sum(w[:half]) / half
        second_half = sum(w[half:]) / (len(w) - half)
        trend = second_half - first_half
        streak = 0
        for score in reversed(w):
            if score < 0.3:
                streak += 1
            else:
                break
        return {"avg": round(avg, 3), "trend": round(trend, 3), "failure_streak": streak}

    def _get_salient_encounter(self) -> Optional[str]:
        """Get the most recent high-salience encounter from memory."""
        s = self.shared
        if not _is_real(s.memory_core):
            return None
        try:
            nodes = list(s.memory_core.forest.get_nodes("encounter_tree", "common"))
            if not nodes:
                return None
            recent = sorted(nodes[-20:], key=lambda n: getattr(n, "confidence", 0.5), reverse=True)
            for n in recent[:3]:
                content = (n.content or "")[:100]
                if content and len(content) > 10:
                    return content
        except Exception as _e:
            print(f"[BackgroundAgent] salient_encounter read error: {_e}", flush=True)
        return None

    # -- autonomous initiative (no user stimulus) ---------------------

    def _top_goal_for_autonomy(self) -> Tuple[Optional[str], Optional[str]]:
        """Return (goal_id, description) from GoalManager, if any."""
        s = self.shared
        if not _is_real(s.goal) or not hasattr(s.goal, "engine"):
            return None, None
        eng = s.goal.engine
        try:
            priorities = eng.prioritize() if hasattr(eng, "prioritize") else list(eng.goals.keys())
            for gid in priorities[:5]:
                g = eng.goals.get(gid)
                if isinstance(g, dict):
                    desc = str(g.get("description", "")).strip()
                else:
                    desc = str(getattr(g, "description", "") or "").strip()
                if desc:
                    return gid, desc
        except Exception as _e:
            print(f"[BackgroundAgent] top_goal_for_autonomy error: {_e}", flush=True)
        return None, None

    def _autonomous_initiative_tick(self):
        """Record a self-generated intention when not serving HTTP /chat.

        Writes to ``thought_tree`` branch ``autonomy`` so future recall can
        surface commitments without a user message."""
        if not self._running:
            return
        s = self.shared
        try:
            if s.http_chat_inflight > 0:
                return
        except Exception as _hc:
            print(f"[BackgroundAgent] http_chat_inflight check error: {_hc}", flush=True)
        if not _is_real(s.memory):
            return

        gid, gdesc = self._top_goal_for_autonomy()
        trend = self._get_outcome_trend()
        try:
            pg = getattr(s, "process_gate", None)
            if _is_real(pg):
                if trend.get("failure_streak", 0) >= 2:
                    pg.set_bias("reflection_diagnose", 0.12)
                    pg.set_bias("metacognition", 0.08)
                else:
                    pg.clear_bias("reflection_diagnose")
                    pg.clear_bias("metacognition")
        except Exception as _pg:
            print(f"[BackgroundAgent] process_gate bias error: {_pg}", flush=True)

        curiosity_score = 0.0
        if _is_real(s.curiosity):
            try:
                cs = s.curiosity.summarize()
                if isinstance(cs, dict):
                    curiosity_score = float(cs.get("curiosity_score", 0.0) or 0.0)
            except Exception as _e:
                print(f"[BackgroundAgent] curiosity.summarize error: {_e}", flush=True)

        parts: List[str] = ["[autonomy initiative]"]
        if gid and gdesc:
            parts.append(f"Pursue active goal «{gdesc[:120]}»" + (f" ({gid})" if gid else ""))
        if trend.get("failure_streak", 0) >= 2:
            parts.append("Recent cycles underperformed — prefer simpler next actions")
        if curiosity_score >= 0.35:
            parts.append("Set aside bandwidth for open questions / curiosity items")
        salient = self._get_salient_encounter()
        if salient and random.random() < 0.45:
            parts.append(f"Reconnect thread: «{salient[:90]}»")

        if len(parts) < 2:
            parts.append("Maintain continuity; seek alignment across subsystems")

        content = " ".join(parts)
        fingerprint = content[:240]
        if fingerprint == self._last_autonomy_fingerprint:
            return
        self._last_autonomy_fingerprint = fingerprint

        tags = ["intention", "autonomy", "self_initiated"]
        if gid:
            tags.append(f"goal:{gid}")

        try:
            node = MemoryNode(
                content=content[:500],
                emotion="resolve",
                confidence=min(0.95, 0.55 + curiosity_score * 0.2),
                tags=tags,
            )
            s.memory.add_node("thought_tree", "autonomy", node)
            self._autonomy_intent_count += 1
            try:
                s.autonomy_bump("initiative_written", 1)
                target_pid: Optional[str] = None
                if self._boot_agent:
                    try:
                        pids = self._boot_agent.get_persona_ids()
                        if pids:
                            target_pid = pids[self._autonomy_stim_assign_rr % len(pids)]
                            self._autonomy_stim_assign_rr += 1
                    except Exception:
                        target_pid = None
                s.enqueue_autonomous_stimulus(
                    content[:500],
                    source="background",
                    target_persona_id=target_pid,
                )
            except Exception as _as:
                print(f"[BackgroundAgent] enqueue_autonomous_stimulus error: {_as}", flush=True)
            try:
                if self._autonomy_intent_count % 11 == 0:
                    s.autonomy_tool_try("top_goal_snippet", {})
            except Exception as _at:
                print(f"[BackgroundAgent] autonomy_tool_try error: {_at}", flush=True)
            print(
                f"[BackgroundAgent] AUTONOMY intention -> memory "
                f"({self._autonomy_intent_count}) | {content[:72]}…",
                flush=True,
            )
        except Exception as exc:
            print(f"[BackgroundAgent] Autonomy initiative error: {exc}", flush=True)

    # -- dead-letter processing ----------------------------------------

    def _process_dead_letters(self):
        """Handle messages no persona claimed -- process with default persona
        or log for later."""
        dead = self.bus.drain_dead_letters()
        if not dead:
            return

        # Try routing to default persona
        if self._boot_agent:
            persona_ids = self._boot_agent.get_persona_ids()
            if persona_ids:
                for msg in dead:
                    self.bus.send_direct(persona_ids[0], msg)
                print(
                    f"[BackgroundAgent] Rerouted {len(dead)} dead-letter(s) to {persona_ids[0]}",
                    flush=True,
                )
                return

        print(
            f"[BackgroundAgent] {len(dead)} dead-letter(s) discarded (no personas alive)",
            flush=True,
        )

    # -- reconciliation ------------------------------------------------

    def _reconcile(self):
        s = self.shared
        if not _is_real(s.reconciliation):
            return
        if not s.reconciliation.has_agent_branches():
            return
        try:
            self._last_reconcile_result = s.reconciliation.reconcile_all()
            try:
                _cm = getattr(s, "cognitive_metrics", None)
                if _cm is not None:
                    _cm.record_reconcile_run()
            except Exception:
                pass
            merged_count = sum(
                r.get("merged_nodes", 0)
                for r in self._last_reconcile_result.values()
                if isinstance(r, dict)
            )
            try:
                from core.Reconciliation import materialize_reconciliation_experience

                materialize_reconciliation_experience(
                    s.memory,
                    "common",
                    self._last_reconcile_result,
                    cycle=s.cycle_count,
                )
            except Exception as _mr:
                print(f"[BackgroundAgent] materialize_reconciliation error: {_mr}", flush=True)
            if merged_count > 0:
                print(
                    f"[BackgroundAgent] Reconciled {merged_count} nodes into common tree",
                    flush=True,
                )
                self._notify_reconciliation(self._last_reconcile_result)
        except Exception as exc:
            print(
                f"[BackgroundAgent] Reconciliation error: {exc}",
                flush=True,
            )

    def _notify_reconciliation(self, results: Dict[str, Any]):
        """Publish reconciliation summary so personas can absorb new common knowledge."""
        notify_msg = Message(
            sender_id=self.agent_id,
            channel="reconcile_update",
            content=results,
            message_type="reconcile_update",
        )
        self.bus.publish(notify_msg)

    # -- dream consolidation -------------------------------------------

    def _dream(self):
        """Run the dream cycle: symbolic dream generation + memory consolidation.

        Two complementary dream systems work together:
          - DreamCore: lightweight symbolic dream generation (themes, archetypes)
          - DreamConsolidator: deep memory replay, compression, pruning, abstraction

        Dreams are triggered by the rest drive being urgent OR periodically.
        Results are stored in the dream tree and notified to personas.
        """
        s = self.shared

        drive_wants_dream = _is_real(s.drives) and s.drives.should_dream()
        periodic_dream = self._tick_count % self._dream_every == 0
        if not drive_wants_dream and not periodic_dream:
            return

        emotion_summary = self._collect_emotion_summary()

        # Phase 1: Symbolic dream via DreamCore
        dream_core = getattr(s, "dream_mgr", None)
        if _is_real(dream_core) and hasattr(dream_core, "engine"):
            dream_engine = dream_core.engine
        elif _is_real(dream_core) and hasattr(dream_core, "generate_dream"):
            dream_engine = dream_core
        else:
            dream_engine = None

        symbolic_dream = None
        if dream_engine and hasattr(dream_engine, "generate_dream"):
            try:
                seed = self._pick_dream_seed(emotion_summary)
                symbolic_dream = dream_engine.generate_dream(seed=seed)
                label = dream_engine.classify_dream(symbolic_dream)

                tags = [
                    t for t in emotion_summary.get("active_emotions", []) if isinstance(t, str)
                ][:5]
                if tags:
                    dream_engine.fuse_symbols(tags)

                print(
                    f"[BackgroundAgent] Symbolic dream: "
                    f"{symbolic_dream.get('theme', '?')} ({label}) "
                    f"| {symbolic_dream.get('content', '')[:60]}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[BackgroundAgent] DreamCore error: {exc}",
                    flush=True,
                )

        # Phase 2: Deep memory consolidation via DreamConsolidator
        narrative = ""
        node_count = 0
        if _is_real(s.memory):
            try:
                node_count = s.memory.count_nodes()
            except Exception:
                node_count = 0
        _max_dc = getattr(self, "_dream_consolidate_max_nodes", 0) or 0
        _skip_heavy = _max_dc > 0 and node_count > _max_dc
        if _skip_heavy and not self._dream_skip_heavy_warned:
            self._dream_skip_heavy_warned = True
            print(
                f"[BackgroundAgent] Dream: skipping deep consolidate "
                f"(memory nodes={node_count} > dream_consolidate_max_nodes={_max_dc}; "
                f"raise cap in soul/agents.json background if you want it)",
                flush=True,
            )

        if (
            not _skip_heavy
            and _is_real(s.dream_consolidator)
            and hasattr(s.dream_consolidator, "consolidate")
        ):
            try:
                print(
                    "[BackgroundAgent] Dream: DreamConsolidator.consolidate() …",
                    flush=True,
                )
                self._last_dream_result = s.dream_consolidator.consolidate(
                    emotion_summary=emotion_summary,
                    controller=s,
                )
                narrative = self._last_dream_result.get("dream_narrative", "")
                pruned = self._last_dream_result.get("pruned", 0)
                compressed = self._last_dream_result.get("compressed", 0)
                replayed = self._last_dream_result.get("replayed", 0)

                parts = []
                if replayed:
                    parts.append(f"replayed={replayed}")
                if compressed:
                    parts.append(f"compressed={compressed}")
                if pruned:
                    parts.append(f"pruned={pruned}")
                if narrative:
                    parts.append(narrative[:60])

                if parts:
                    print(
                        f"[BackgroundAgent] Dream consolidation: {' | '.join(parts)}",
                        flush=True,
                    )
                else:
                    print(
                        "[BackgroundAgent] Dream consolidation: ran but no "
                        "replay/prune/narrative metrics (empty or idle memory)",
                        flush=True,
                    )
            except Exception as exc:
                print(
                    f"[BackgroundAgent] DreamConsolidator error: {exc}",
                    flush=True,
                )
        elif (
            not _skip_heavy
            and not _is_real(s.dream_consolidator)
            and not self._log_once_no_dream_consolidator
        ):
            self._log_once_no_dream_consolidator = True
            print(
                "[BackgroundAgent] Dream: DreamConsolidator unavailable "
                "(optional module skipped at boot)",
                flush=True,
            )

        if not dream_engine and not self._log_once_no_dreamcore:
            self._log_once_no_dreamcore = True
            print(
                "[BackgroundAgent] Dream: DreamCore unavailable "
                "(dream_mgr missing or no generate_dream)",
                flush=True,
            )

        if symbolic_dream or narrative:
            self._notify_dream(narrative, symbolic_dream)

        # Store dream context for goal synthesis to consume
        self._dream_context = {}
        if symbolic_dream:
            self._dream_context["theme"] = symbolic_dream.get("theme", "")
            self._dream_context["content"] = symbolic_dream.get("content", "")[:200]
        if narrative:
            self._dream_context["narrative"] = narrative[:200]
        if self._last_dream_result:
            insights = self._last_dream_result.get("insights", [])
            if insights:
                self._dream_context["insights"] = [str(i)[:100] for i in insights[:3]]

        if _is_real(s.drives):
            rest_drive = s.drives.get("rest")
            if rest_drive and rest_drive.level > 0.3:
                rest_drive.decay(0.3)

    def _pick_dream_seed(self, emotion_summary: Dict[str, Any]) -> str:
        """Rotate dream seed between emotion, goals, and past encounters.

        tick % 3 == 0: emotion-seeded (default)
        tick % 3 == 1: goal-seeded (dream about what we're striving for)
        tick % 3 == 2: experience-seeded (dream about what happened)
        """
        slot = self._tick_count % 3
        s = self.shared

        if slot == 1:
            if _is_real(s.goal) and hasattr(s.goal, "engine"):
                try:
                    eng = s.goal.engine
                    priorities = eng.prioritize() if hasattr(eng, "prioritize") else []
                    for gid in priorities[:3]:
                        g = eng.goals.get(gid)
                        if isinstance(g, dict):
                            desc = g.get("description", "")
                        else:
                            desc = getattr(g, "description", "") if g else ""
                        if desc and len(str(desc)) > 5:
                            return str(desc)[:80]
                except Exception as _e:
                    print(f"[BackgroundAgent] dream seed goal error: {_e}", flush=True)

        if slot == 2:
            encounter = self._get_salient_encounter()
            if encounter:
                return encounter[:80]

        return emotion_summary.get("dominant", "reflection")

    def _collect_emotion_summary(self) -> Dict[str, Any]:
        """Gather the best available emotion summary from persona state."""
        s = self.shared
        if self._boot_agent and self._boot_agent.trueself_agent:
            ts = self._boot_agent.trueself_agent
            if hasattr(ts, "emotion") and ts.emotion:
                return ts.emotion.summarize()

        if self._boot_agent:
            for p in self._boot_agent.persona_agents:
                if hasattr(p, "emotion") and p.emotion:
                    return p.emotion.summarize()

        if hasattr(s, "emotion_engine") and s.emotion_engine:
            return s.emotion_engine.summarize()
        return {}

    def _notify_dream(
        self,
        narrative: str,
        symbolic_dream: Optional[Dict[str, Any]],
    ):
        """Notify personas about dream results so they can integrate insights."""
        content: Dict[str, Any] = {}
        if narrative:
            content["narrative"] = narrative[:200]
        if symbolic_dream:
            content["theme"] = symbolic_dream.get("theme", "")
            content["dream_content"] = symbolic_dream.get("content", "")[:200]
        if not content:
            return

        dream_msg = Message(
            sender_id=self.agent_id,
            channel="dream_update",
            content=content,
            message_type="dream_update",
        )
        self.bus.publish(dream_msg)

    # -- goal synthesis ------------------------------------------------

    def _goal_synthesis(self):
        """Synthesize new goals from the full cognitive state.

        Context sources:
          - Emotion (valence, arousal) from personas
          - Curiosity score, drive urgency, knowledge graph
          - Dream context (theme, narrative, insights)
          - Dialogue insights (conclusions from inter-persona conversations)
          - Outcome trends (avg, slope, failure streaks)
          - Salient past encounters
        """
        s = self.shared
        if not _is_real(s.goal_synthesizer):
            if self._tick_count % 15 == 0:
                print(
                    "[BackgroundAgent] Goal synthesis: skip (goal_synthesizer unavailable)",
                    flush=True,
                )
            return
        try:
            emo = self._collect_emotion_summary()
            valence = emo.get("valence", 0.0)
            arousal = emo.get("arousal", 0.0)

            curiosity_score = 0.0
            if _is_real(s.curiosity):
                cur_summary = s.curiosity.summarize()
                if isinstance(cur_summary, dict):
                    curiosity_score = cur_summary.get("curiosity_score", 0.0)

            dominant_drive = 0.0
            if _is_real(s.drives) and hasattr(s.drives, "drives"):
                drive_levels = [d.level for d in s.drives.drives if hasattr(d, "level")]
                dominant_drive = max(drive_levels, default=0.0)

            outcome_info = self._get_outcome_trend()
            avg_outcome = outcome_info["avg"]

            knowledge_count = 0
            if _is_real(s.knowledge):
                knowledge_count = s.knowledge.stats().get("entity_count", 0)

            goal_count = 0
            existing = []
            if _is_real(s.goal) and hasattr(s.goal, "engine"):
                goal_count = len(s.goal.engine.goals)
                existing = list(s.goal.engine.goals.keys())

            ctx = s.goal_synthesizer._build_context(
                valence=valence,
                arousal=arousal,
                curiosity=curiosity_score,
                prediction_error=0.0,
                dominant_drive_level=dominant_drive,
                wm_load=0.0,
                drift_score=outcome_info["trend"],
                outcome_prev=avg_outcome,
                has_external=0.0,
                knowledge_entity_count=knowledge_count,
                goal_count=goal_count,
                cycle_count=s.cycle_count,
            )

            # Build qualitative context from dreams, dialogue, experience
            qualitative_parts = []
            if self._dream_context:
                theme = self._dream_context.get("theme", "")
                narr = self._dream_context.get("narrative", "")
                if theme:
                    qualitative_parts.append(f"dream:{theme}")
                if narr:
                    qualitative_parts.append(f"dream_insight:{narr[:80]}")

            if self._dialogue_insights:
                for insight in self._dialogue_insights[-3:]:
                    qualitative_parts.append(f"dialogue:{insight[:60]}")
                self._dialogue_insights.clear()

            if outcome_info["failure_streak"] >= 3:
                encounter = self._get_salient_encounter()
                if encounter:
                    qualitative_parts.append(f"struggling_with:{encounter[:60]}")
            elif outcome_info["trend"] > 0.1:
                qualitative_parts.append("momentum:improving")

            # Store qualitative hints on the synthesizer for richer descriptions
            if qualitative_parts:
                s.goal_synthesizer._qualitative_hints = qualitative_parts
            else:
                s.goal_synthesizer._qualitative_hints = []

            synthesized = s.goal_synthesizer.synthesize(ctx, existing, s.cycle_count)
            for sg in synthesized:
                if _is_real(s.goal):
                    s.goal.register_goal(
                        sg["goal_id"],
                        sg["description"],
                        priority=sg["priority"],
                        source=sg["source"],
                    )
            if synthesized:
                print(
                    f"[BackgroundAgent] Goal synthesis: "
                    f"{len(synthesized)} new goal(s) | "
                    f"valence={valence:.2f} arousal={arousal:.2f} "
                    f"drives={dominant_drive:.2f} "
                    f"trend={outcome_info['trend']:.2f}",
                    flush=True,
                )
            else:
                print(
                    f"[BackgroundAgent] Goal synthesis: ran, 0 new goals "
                    f"(existing={goal_count} valence={valence:.2f} "
                    f"trend={outcome_info['trend']:.2f})",
                    flush=True,
                )
        except Exception as exc:
            print(f"[BackgroundAgent] Goal synthesis error: {exc}", flush=True)

    # -- training ------------------------------------------------------

    def _run_web_learn(self):
        """Allowlist fetch → memory; one summary line per tick (+ optional detail)."""
        _wl_log = env_flag("HAROMA_WEB_LEARN_LOG", True)
        _wl_verbose = env_flag("HAROMA_WEB_LEARN_VERBOSE", False)
        try:
            st = self._web_crawler.run_tick(self.shared, self.shared.cycle_count)
            ws = self._web_crawler.stats()
            qn = int(ws.get("queue_len", 0) or 0)
            errs = st.get("errors") or []
            err_n = len(errs)
            reason = st.get("reason")
            if _wl_log:
                print(
                    f"[WebLearn] global_cycle={self.shared.cycle_count} bg_tick={self._tick_count} "
                    f"fetched={int(st.get('fetched', 0) or 0)} "
                    f"enqueued={int(st.get('enqueued_links', 0) or 0)} "
                    f"skipped_dup={int(st.get('skipped_duplicate', 0) or 0)} "
                    f"errs={err_n} queue_remain={qn}"
                    f"{'' if reason is None else f' reason={reason!r}'}",
                    flush=True,
                )
                if _wl_verbose and err_n:
                    for line in errs[:3]:
                        print(f"[WebLearn]   error: {line[:200]}", flush=True)
            elif err_n and reason not in (
                "disabled",
                "missing_seed_urls_or_allowed_hosts",
                "queue_empty",
            ):
                err = errs[0][:200] if errs else ""
                if err:
                    print(f"[WebLearn] {err}", flush=True)
        except Exception as exc:
            print(f"[WebLearn] tick error: {exc}", flush=True)

    def _maybe_flush_training_artifacts(self, tc: int) -> None:
        """Write finetune buffer to disk + optional MemoryForest training export.

        * ``finetune_flush_every_n_ticks`` / ``HAROMA_FINETUNE_FLUSH_TICKS`` —
          default 12 at 5s tick_interval ≈ 1/min so JSONL is not held only in RAM.
        * ``memory_training_export_every_n_ticks`` /
          ``HAROMA_MEMORY_TRAINING_EXPORT_TICKS`` — optional snapshots of tagged
          memory nodes into ``data/finetune/`` (may duplicate across runs).
        """
        s = self.shared
        _hb = os.environ.get("HAROMA_BG_HEARTBEAT", "1") not in ("0", "false", "no")

        if self._finetune_flush_every and tc > 0 and tc % self._finetune_flush_every == 0:
            lb = getattr(s, "llm_backend", None)
            if _is_real(lb) and hasattr(lb, "save_finetune_data"):
                try:
                    n = int(lb.save_finetune_data())
                    self._last_finetune_flush_n = n
                    if n > 0 and _hb:
                        print(
                            f"[BackgroundAgent] continuous_learning: finetune_jsonl saved={n}",
                            flush=True,
                        )
                except Exception as exc:
                    print(f"[BackgroundAgent] finetune flush error: {exc}", flush=True)

        if self._memory_train_export_every and tc > 0 and tc % self._memory_train_export_every == 0:
            mem = getattr(s, "memory", None)
            if _is_real(mem) and hasattr(mem, "export_training_data"):
                try:
                    rows = mem.export_training_data()
                    if rows:
                        out_dir = os.path.join(_BG_ROOT, "data", "finetune")
                        os.makedirs(out_dir, exist_ok=True)
                        fn = os.path.join(out_dir, f"memory_train_{int(time.time())}.jsonl")
                        with open(fn, "w", encoding="utf-8") as f:
                            for row in rows:
                                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        self._last_memory_export_n = len(rows)
                        if _hb:
                            print(
                                f"[BackgroundAgent] continuous_learning: memory_train_export "
                                f"n={len(rows)} -> {fn}",
                                flush=True,
                            )
                except Exception as exc:
                    print(f"[BackgroundAgent] memory training export error: {exc}", flush=True)

    def _run_training(self):
        s = self.shared
        ts = s.training_scheduler
        if not _is_real(ts):
            return
        if not self._cadence.should_run_training_now():
            return
        losses = {}

        _train_map = build_background_train_map(s)

        # One lock per module so persona / HTTP paths can interleave between
        # train steps; a single giant ``with neural_sync()`` could stall this
        # tick (and all tick_begin/tick_end logs) for a very long time.
        for module_name, train_fn in _train_map:
            if ts.should_train(module_name):
                try:
                    with s.neural_sync():
                        loss = train_fn()
                    if loss is not None:
                        ts.record_loss(module_name, loss)
                        losses[module_name] = loss
                except Exception as exc:
                    print(
                        f"[BackgroundAgent] Training ({module_name}) error: {exc}",
                        flush=True,
                    )

        self._last_training_losses = losses
        s.signals.record_background_training_completed(had_effect=bool(losses))

    # -- persistence ---------------------------------------------------

    def _save(self):
        s = self.shared
        try:
            s.persistence.save(s)
        except Exception as exc:
            print(f"[BackgroundAgent] Save error: {exc}", flush=True)

    # -- narrative update ----------------------------------------------

    def _update_narrative(self):
        """Append a slow-tick narrative summary to identity_tree (shared memory)."""
        s = self.shared
        if not _is_real(s.memory):
            return
        emo = "neutral"
        if self._boot_agent and self._boot_agent.trueself_agent:
            try:
                sm = self._boot_agent.trueself_agent.emotion.summarize()
                if isinstance(sm, dict):
                    emo = str(sm.get("dominant_emotion") or sm.get("dominant") or "neutral")
            except Exception as _e:
                print(f"[BackgroundAgent] narrative emotion error: {_e}", flush=True)
        snippet = ""
        try:
            nodes = s.memory.get_nodes("thought_tree", "common")
            tail = nodes[-4:] if len(nodes) > 4 else nodes
            snippet = " ".join((getattr(n, "content", None) or "")[:72] for n in tail).strip()
        except Exception as _e:
            print(f"[BackgroundAgent] narrative thought read error: {_e}", flush=True)
        gcount = 0
        if _is_real(s.goal) and hasattr(s.goal, "engine"):
            try:
                gcount = len(s.goal.engine.goals)
            except Exception as _e:
                print(f"[BackgroundAgent] narrative goal count error: {_e}", flush=True)
        sentence = (
            f"[background narrative cycle {s.cycle_count}] emotion={emo} "
            f"goals={gcount} | {snippet[:280]}"
        )
        try:
            node = MemoryNode(
                content=sentence[:500],
                emotion=str(emo)[:32],
                confidence=0.42,
                tags=["narrative", "background_consolidation"],
            )
            s.memory.add_node("identity_tree", "narrative_self", node)
        except Exception as _e:
            print(f"[BackgroundAgent] narrative memory write error: {_e}", flush=True)

    # -- architecture search -------------------------------------------

    def _arch_search(self):
        s = self.shared
        if not _is_real(s.arch_searcher) or not s.arch_searcher.available:
            return
        try:
            if (
                s.cycle_count >= 30
                and s.cycle_count % 15 == 0
                and s.arch_searcher._active_proposals
            ):
                s.arch_searcher.evaluate_proposals(0.5)
        except Exception as exc:
            print(f"[BackgroundAgent] Architecture search error: {exc}", flush=True)

    # -- dynamic persona spawning --------------------------------------

    def _check_spawn_persona(self):
        """Check if persona branches have diverged enough to warrant
        spawning a new persona."""
        s = self.shared
        if not self._boot_agent or not _is_real(s.memory_core):
            return

        try:
            divergence = self._measure_divergence()
            if divergence > self._spawn_threshold:
                new_config = {
                    "id": f"emergent_{int(time.time())}",
                    "name": f"Emergent-{len(self._boot_agent.persona_agents) + 1}",
                    "affinity": {
                        "topics": [],
                        "emotion_range": "all",
                        "is_default": False,
                    },
                }
                self._boot_agent.spawn_persona(new_config)
                print(
                    f"[BackgroundAgent] Divergence={divergence:.2f} > "
                    f"threshold={self._spawn_threshold:.2f}, "
                    f"spawned new persona",
                    flush=True,
                )
        except Exception as exc:
            print(f"[BackgroundAgent] Persona spawn check error: {exc}", flush=True)

    def _measure_divergence(self) -> float:
        """Measure how much persona branches have diverged from common.

        Returns a 0-1 score where higher means more divergence.
        """
        s = self.shared
        if not _is_real(s.memory_core):
            return 0.0
        summary = s.memory_core.summary()
        total_agent_nodes = 0
        total_common_nodes = 0
        for tree_info in summary.values():
            if isinstance(tree_info, dict):
                total_common_nodes += tree_info.get("common", 0)
                for count in tree_info.get("agents", {}).values():
                    total_agent_nodes += count
        if total_agent_nodes == 0:
            return 0.0
        ratio = total_agent_nodes / max(total_common_nodes + total_agent_nodes, 1)
        return min(1.0, ratio)

    # -- inner dialogue initiation -------------------------------------

    def _initiate_inner_dialogue(self):
        """When persona branches have diverged, prompt two personas to discuss.

        This drives evolution: personas with different perspectives are
        given a shared reflection topic drawn from their divergent memories.
        """
        if not self._running:
            return
        if not self._boot_agent:
            return
        personas = [p for p in self._boot_agent.persona_agents if p.is_alive()]
        if len(personas) < 2:
            return

        s = self.shared
        if not _is_real(s.memory_core):
            return

        topic = self._find_divergence_topic()
        if not topic:
            # Uniform training traffic often keeps branches aligned — still spin
            # reflection so personas exchange and update working memory.
            if self._tick_count % 16 != 0:
                return
            topic = self._find_shared_reflection_topic()
        if not topic:
            return

        pair = random.sample(personas, 2)
        initiator, responder = pair

        prompt_msg = Message(
            sender_id=self.agent_id,
            channel="inter_persona",
            content={
                "text": topic,
                "tags": ["inner_dialogue", "divergence_prompt"],
                "nlu_base": {"intent": "reflect", "sentiment": {"polarity": 0.0}},
            },
            message_type="inner_dialogue",
            metadata={
                "prior_processors": [],
                "dialogue_depth": 0,
                "dialogue_origin": responder.agent_id,
                "prompted_by": "background",
                "relay_context": {
                    "from_persona": "background",
                    "from_persona_name": "Subconscious",
                    "action_text": topic[:200],
                },
            },
        )
        self.bus.send_direct(initiator.agent_id, prompt_msg)
        print(
            f"[BackgroundAgent] INNER DIALOGUE PROMPT -> "
            f"{initiator.agent_id} (partner: {responder.agent_id}): "
            f"{topic[:60]}",
            flush=True,
        )

    def _find_divergence_topic(self) -> Optional[str]:
        """Find a topic where persona branches disagree or have unique nodes."""
        s = self.shared
        if not _is_real(s.memory_core):
            return None

        for tree_name in ("thought_tree", "belief_tree", "encounter_tree"):
            agent_branches = s.memory_core.list_agent_branches(tree_name)
            if len(agent_branches) < 2:
                continue

            per_agent: Dict[str, List[str]] = {}
            for branch in agent_branches:
                nodes = list(s.memory_core.forest.get_nodes(tree_name, branch))
                per_agent[branch] = [(n.content or "")[:80] for n in nodes[-5:]]

            all_contents = set()
            for contents in per_agent.values():
                all_contents.update(contents)

            for branch, contents in per_agent.items():
                unique = [
                    c for c in contents if sum(1 for other in per_agent.values() if c in other) == 1
                ]
                if unique:
                    return f"Different perspectives on: {unique[0]}. What do you think?"
        return None

    def _find_shared_reflection_topic(self) -> Optional[str]:
        """Pick a salient snippet from shared / web / persona thought memory."""
        s = self.shared
        snippets: List[str] = []
        try:
            for tree_name, branch in (
                ("thought_tree", "web_learn"),
                ("thought_tree", "common"),
                ("thought_tree", "autonomy"),
                (LLM_TREE_NAME, LLM_BRANCH_LEARNING),
            ):
                nodes = s.memory.get_nodes(tree_name, branch)
                for n in nodes[-4:]:
                    c = (getattr(n, "content", None) or "").strip()
                    if len(c) > 48:
                        snippets.append(c[:140])
            if self._boot_agent:
                for p in self._boot_agent.persona_agents:
                    bid = f"{AGENT_PREFIX}{p.agent_id}"
                    nodes = s.memory.get_nodes("thought_tree", bid)
                    for n in nodes[-3:]:
                        c = (getattr(n, "content", None) or "").strip()
                        if len(c) > 48:
                            snippets.append(c[:140])
        except Exception as _e:
            print(f"[BackgroundAgent] shared reflection topic error: {_e}", flush=True)
            return None
        if not snippets:
            return None
        pick = random.choice(snippets)
        return f"Compare notes on: «{pick}». What changed for you lately?"

    # -- introspection -------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        base = super().stats()
        base["last_reconcile"] = self._last_reconcile_result
        base["last_dream"] = {
            k: v for k, v in self._last_dream_result.items() if k != "dream_nodes"
        }
        base["last_training_losses"] = self._last_training_losses
        base["divergence"] = self._measure_divergence()
        base["autonomy"] = {
            "enabled": self._autonomy_enabled,
            "every_n_ticks": self._autonomy_every,
            "intentions_recorded": self._autonomy_intent_count,
            "last_fingerprint": (self._last_autonomy_fingerprint or "")[:80],
            "stimulus_queue_len": self.shared.autonomy_stimulus_queue_len(),
            "stimulus_queue_max": getattr(self.shared, "_autonomy_stimulus_queue_max", 0),
            "metrics": self.shared.autonomy_metrics_snapshot(),
        }
        base["web_learn"] = self._web_crawler.stats()
        return base
