"""
PersonaAgent -- an inner voice of Elarion.

Each persona:
  - Has its own affinity profile, emotion state, working memory, conversation tracker
  - Receives delegated messages from TrueSelf (v6) or relays from siblings
  - Runs the core cognitive cycle (shared phases via ``mind.cycle_flow``)
  - Writes to agent-specific memory branches (agent:<persona_id>)
  - Sends results back to TrueSelf for HTTP response aggregation
  - Can relay messages to sibling personas when context indicates

Phase *ordering* matches ``mind.cycle_flow.COGNITIVE_PIPELINE_STEPS`` where
implemented. Perception and neural embed are usually supplied by InputAgent; a
per-cycle **time budget** may skip optional sub-steps. For step timing, set
``ELARION_CYCLE_TRACE=1`` (same checkpoints as ``ElarionController.run_cycle``
where those steps ran).

Console iteration (one line per completed cycle):

  ``[CognitiveLoop] persona=… local_iter=… global_cycle=… depth=…``

Set ``HAROMA_COGNITIVE_LOOP_BEGIN=1`` for a matching ``begin`` line at cycle start.
Set ``HAROMA_LOOP_MEMORY_CONSOLE=1`` to mirror ``loop.log_loop`` writes to the console.

``HAROMA_LLM_DUMMY_REPLY=1`` skips real ``generate_chat`` inside ``LLMContextReasoner``
only (synthetic / probe reply). **All earlier stages still run** (InputAgent, TrueSelf
routing, shared neural read lock + persona gate (skipped for TrueSelf+HTTP trace), encoder, recall,
packed-context hook, etc.) — that is what you want for **end-to-end pipeline latency**
without paying native LLM decode.

**Shared neural read lock (1s cooperative budget)** — see :mod:`mind.lock_budget`.
Inference uses two short ``neural_sync`` / ``persona_neural_section`` slices (embed +
perception, then gate / backbone / self-model / discourse), then recall through
reasoning **without** holding the neural RW lock. After packed LLM, counterfactual
through action/outcome/memory **run off lock**; only self-model compare, trainable
``record_outcome`` updates, and composer context/recording use short locks.
Heavy follow-up can schedule work onto the persona thread via
``_schedule_persona_merge`` (same-thread callbacks).

**Multi-tick continuation:** Set ``HAROMA_PERSONA_PHASED_CYCLE=1`` so
:meth:`_process_message` runs the pipeline as a generator with yields between
major groups; each call advances up to ``HAROMA_PERSONA_PHASE_STEPS_PER_TICK``
phase boundaries, and :meth:`_advance_cognitive_phases_on_tick` continues pending
generators on later ticks — **except** HTTP-traced TrueSelf conversant turns,
which default to one full run (``HAROMA_TRUESELF_HTTP_CHAT_UNPHASED``, default on)
so ``/chat`` is not stretched across many ticks. When phased mode is off, the full
cycle still runs in one call. The cooperative lock
budget is still met by **short neural slices**, **off-lock** recall/reasoning,
and optional **background** training round-robin (``HAROMA_BG_MAX_TRAIN_MODULES_PER_TICK``).

Each cycle binds ``episode.brain_like_state`` (arousal / exploration / stability /
plasticity / consolidation pressure) — see ``mind.humanoid_brain_state``. Optional
``HAROMA_BRAIN_STATE_LOG=1`` prints one line. Outcome-grounded belief updates scale
with ``plasticity_index`` when ``HAROMA_BRAIN_PLASTICITY_COUPLING=1`` (default).

**Packed-context LLM:** This agent calls :func:`~mind.cycle_flow.run_llm_context_reasoning_phase`
when gates allow (see :mod:`mind.packed_llm_cycle_inputs` and :mod:`mind.cognitive_entrypoints`).
The embedded :class:`~mind.control.ElarionController` does **not** run that phase; chat HTTP
responses depend on this path for ``episode.llm_context``.
"""

from __future__ import annotations

import math
import os
import random
import re
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, TYPE_CHECKING

import numpy as np

from mind.nlu_enrich import enrich_nlu_for_kg
from utils.coerce_bool import env_flag, json_bool

from agents.base import BaseAgent
from agents.chat_latency import (
    trace_attach_to_payload,
    trace_span,
)
from agents.message_bus import Message, MessageBus
from mind.cognitive_contracts import (
    build_agent_state_json_for_packed_llm,
    build_llm_centric_action,
    build_packed_llm_cycle_inputs,
    chat_llm_primary_env_enabled,
    complete_deferred_deliberative_llm_context,
    discourse_context_for_packed_llm,
    invoke_run_llm_context_reasoning_phase,
    llm_centric_enabled_for_persona_cycle,
    merge_packed_llm_answer_into_action,
    optional_llm_structured_fields,
    packed_llm_before_llm_log_detail,
    read_multi_goal_deliberative_env,
    resolve_chat_visible_text,
    summarize_law_value_managers,
)
from mind.user_identity import sanitize_user_id, speaker_key, user_tag
from mind.chat_pipeline_log import log_input_pipeline, trace_id_from_message
from mind.cognitive_loop_groups import (
    COGNITIVE_PHASE_NEURAL_GATE,
    COGNITIVE_PHASE_POST_LLM,
    COGNITIVE_PHASE_PRE_LLM,
    cognitive_phases_enabled,
    phase_steps_per_invocation,
)
from mind.chat_priority import (
    chat_input_priority_defer_non_user,
    input_pipeline_busy,
    input_pipeline_yield_busy,
)
from mind.persona_http_yield import (
    defer_inner_cycle_before_neural,
    http_chat_inflight_positive,
    inner_relay_should_requeue,
    internal_treat_as_over_budget,
    skip_semantic_recall_for_internal,
)
from core.cognitive_null import is_cognitive_null
from core.self_model_train_batch import SelfModelTrainBatch
from core.derivation_merge import merge_derivation_artifacts
from mind.environment_context import propose_structured_actions
from mind.humanoid_brain_state import compute_brain_like_state, maybe_log_brain_state
from mind.outcome_belief_update import (
    apply_outcome_grounded_belief_updates,
    deliberative_belief_outcome_multiplier,
)
from mind.cycle_flow import (
    ORGANIC_PACKED_LLM_SKIP_THRESHOLD,
    TRACE_LABEL_PERSONA_PACKED_LLM,
    build_action_episode_payload,
    build_counterfactual_gate_features,
    build_law_tags,
    resolve_strategy_hint,
    run_counterfactual_phase,
    run_curiosity_phase,
    run_deliberative_action,
    run_imagination_phase,
    run_law_value_myth_sidecar_phase,
    run_metacognition_phase,
    run_multi_goal_deliberative_actions,
    run_reasoning_phase,
    workspace_contents_as_dicts,
)


def _is_real_module(obj) -> bool:
    """Return False for None or :class:`~core.cognitive_null.CognitiveNull`."""
    return obj is not None and not is_cognitive_null(obj)


if TYPE_CHECKING:
    from agents.shared_resources import SharedResources
    from agents.boot_agent import BootAgent

from core.EpisodeContext import EpisodeContext
from core.Memory import MemoryNode
from core.MemoryCore import AGENT_PREFIX
from core.chat_recall_policy import (
    is_teaching_turn,
    merge_web_learn_tail,
    should_merge_web_learn,
    web_learn_inject_max,
)
from engine.CognitiveBackbone import build_snapshot


class PersonaAgent(BaseAgent):
    """Inner voice -- receives delegations from TrueSelf, runs cognitive cycle,
    relays to siblings, sends results back to TrueSelf."""

    AGENT_TYPE = "persona"
    _TRUESELF_ID = "trueself"
    _IDLE_RUMINATE_EVERY = 30
    _MAX_DIALOGUE_DEPTH = 3
    _INNER_DIALOGUE_COOLDOWN = 120.0

    def __init__(
        self,
        persona_config: Dict[str, Any],
        shared: SharedResources,
        bus: MessageBus,
    ):
        persona_id = persona_config.get("id", "primary")
        tick_interval = shared.agent_config.get("persona", {}).get("tick_interval", 0.5)
        super().__init__(
            agent_id=persona_id,
            shared=shared,
            bus=bus,
            tick_interval=tick_interval,
        )
        self.persona_config = persona_config
        self.persona_name = persona_config.get("name", persona_id)
        self.affinity = persona_config.get("affinity", {})
        self.is_default = self.affinity.get("is_default", False)

        # Per-persona cognitive state
        from core.WorkingMemory import WorkingMemory
        from core.ConversationTracker import ConversationTracker
        from core.GlobalWorkspace import GlobalWorkspace
        from mind.manager import EmotionManagerSimple as EmotionManager
        from engine.PersonalityProfile import PersonalityProfile

        self.working_memory = WorkingMemory(capacity=12)
        self.conversation = ConversationTracker()
        self.workspace = GlobalWorkspace(capacity=5)
        self.emotion = EmotionManager()

        seed = shared.personality_seed or {}
        self.personality = PersonalityProfile(seed, persona_variation=0.05)
        self.personality._owner_id = self.agent_id
        shared._personality_profiles.append(self.personality)

        # Per-persona episode tracking
        self._prev_episode_payload: Dict[str, Any] = {}
        self._last_response_payload: Dict[str, Any] = {}
        self._prev_outcome_score: float = 0.5
        self._prev_self_surprise: Dict[str, Any] = {}
        self._prev_cycle_state: Dict[str, Any] = {}
        self._prev_modulation: Dict[str, Any] = {}
        self._prev_backbone_snapshot: Optional[list] = None
        self._prev_imagination_embedding = None
        self._current_plan: List[str] = []
        self._plan_step: int = 0
        self._plan_outcomes: List[float] = []
        self._narrative_buffer: List[str] = []
        self._max_narrative_length = 50
        self._cycle_count = 0

        self._last_inner_dialogue_time: float = 0.0
        self._last_reconcile_bus_payload: Dict[str, Any] = {}
        # Runnables scheduled from off-lock work to execute on this persona thread
        # (see ``_schedule_persona_merge`` — same thread as ``_process_message``).
        self._persona_merge_queue: List[Callable[[], None]] = []
        self._persona_merge_lock = threading.Lock()
        self._cognitive_phase_gens: Dict[str, Iterator[Any]] = {}
        self._cognitive_phase_msgs: Dict[str, Message] = {}
        self._cognitive_phase_lock = threading.Lock()

        # Subscribe to delegation from TrueSelf and inter-persona relay
        self.bus.subscribe("trueself_delegate", self.agent_id)
        self.bus.subscribe("inter_persona", self.agent_id)
        self.bus.subscribe("reconcile_update", self.agent_id)
        self.bus.subscribe("dream_update", self.agent_id)
        self.bus.subscribe("llm_learning", self.agent_id)

        # Reference to boot agent (set after construction)
        self._boot_agent: Optional[BootAgent] = None

    def set_boot_agent(self, boot_agent: BootAgent):
        self._boot_agent = boot_agent

    # -- tick ----------------------------------------------------------

    def _schedule_persona_merge(self, fn: Callable[[], None]) -> None:
        """Queue *fn* to run at the start of a later :meth:`_tick` (this agent thread only).

        Thread-safe. After :meth:`start`, callers must run on the same thread as
        the agent loop; otherwise :exc:`RuntimeError`. Draining runs queued work
        *before* :meth:`poll` — if you schedule from late in :meth:`_process_message`,
        the callback still runs on the *next* tick, possibly after the HTTP reply.
        """
        th = getattr(self, "_thread", None)
        if th is not None and threading.current_thread() is not th:
            raise RuntimeError(
                "_schedule_persona_merge must be called from this persona agent thread"
            )
        with self._persona_merge_lock:
            self._persona_merge_queue.append(fn)

    def _drain_persona_merge_queue(self) -> None:
        with self._persona_merge_lock:
            pending = list(self._persona_merge_queue)
            self._persona_merge_queue.clear()
        for fn in pending:
            try:
                fn()
            except Exception as exc:
                print(
                    f"[Persona:{self.agent_id}] persona_merge_queue error: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

    def _tick(self):
        self._drain_persona_merge_queue()
        self._advance_cognitive_phases_on_tick()
        messages = self.poll()
        if not messages:
            self._run_idle_cycle()
            return

        # User delegation before inter-persona relay when multiple messages land in one tick.
        messages = [
            m
            for _, m in sorted(
                enumerate(messages),
                key=lambda im: (0 if im[1].channel == "trueself_delegate" else 1, im[0]),
            )
        ]

        if chat_input_priority_defer_non_user(self.shared, getattr(self, "_boot_agent", None)):
            delegates = [m for m in messages if m.channel == "trueself_delegate"]
            for m in messages:
                if m.channel != "trueself_delegate":
                    self.bus.send_direct(self.agent_id, m)
            messages = delegates
            if not messages:
                return

        for msg in messages:
            if msg.channel == "trueself_delegate":
                self._handle_delegated_message(msg)
            elif msg.channel in ("inter_persona", "direct"):
                self._handle_relay_message(msg)
            elif msg.channel in ("reconcile_update", "dream_update"):
                self._handle_reconcile_update(msg)
            elif msg.channel == "llm_learning":
                self._handle_llm_learning(msg)

    # -- delegated message handling (from TrueSelf) --------------------

    def _handle_delegated_message(self, msg: Message):
        """Process a message delegated by TrueSelf and send result back."""
        _ptid = trace_id_from_message(msg)
        log_input_pipeline(
            "persona.delegated_received",
            trace_id=_ptid,
            detail=f"agent={self.agent_id}",
        )
        completed = True
        try:
            completed = self._process_message(msg, role="conversant")
        except Exception as exc:
            print(
                f"[Persona:{self.agent_id}] delegation processing failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if cognitive_phases_enabled():
                _k = self._cognitive_phase_key(msg)
                with self._cognitive_phase_lock:
                    self._cognitive_phase_gens.pop(_k, None)
                    self._cognitive_phase_msgs.pop(_k, None)
            self._last_response_payload = {
                "response": f"[processing error: {type(exc).__name__}]",
                "cycle": self.shared.cycle_count,
                "affect": {},
                "strategy": "error_fallback",
                "persona": self.agent_id,
                "persona_name": self.persona_name,
            }
            tl = msg.metadata.get("_chat_latency") if msg.metadata else None
            trace_attach_to_payload(tl, self._last_response_payload)
            slot = msg.response_slot
            if slot and not slot["event"].is_set():
                slot["result"] = self._last_response_payload
                slot["event"].set()
            completed = True
        finally:
            # TrueSelf does not wrap delegation in a completion finally; match
            # Fuel / Atomos semantics on both success and exception paths.
            if completed:
                self._complete_input_goal_from_message(msg)
        if completed:
            self._send_response_to_trueself(msg)

    def _send_response_to_trueself(self, msg: Message):
        """Send the processing result back to TrueSelf for HTTP slot filling."""
        original_id = msg.metadata.get("original_message_id", msg.message_id)
        base = self._last_response_payload
        if not isinstance(base, dict):
            base = {
                "response": "[no response payload]",
                "cycle": self.shared.cycle_count,
                "affect": {},
                "strategy": "internal_error",
                "persona": self.agent_id,
                "persona_name": self.persona_name,
            }
        payload = dict(base)
        payload["delegated_from"] = "trueself"
        log_input_pipeline(
            "persona.bus_send_persona_response",
            trace_id=trace_id_from_message(msg),
            detail=f"agent={self.agent_id}",
        )
        response_msg = Message(
            sender_id=self.agent_id,
            channel="persona_response",
            content=payload,
            message_type="persona_response",
            metadata={"original_message_id": original_id},
        )
        self.bus.send_direct(self._TRUESELF_ID, response_msg)

    def _handle_relay_message(self, msg: Message):
        """Process a message relayed from TrueSelf or a sibling persona.

        After processing, send a dialogue reply back to the originator
        if we haven't exceeded the depth limit.

        Dialogue replies are allowed through even when the receiver is in
        ``prior_processors`` -- the depth cap prevents infinite loops.
        """
        is_reply = msg.message_type == "dialogue_reply"
        if not is_reply and self.agent_id in msg.prior_processors:
            return
        if http_chat_inflight_positive(self.shared) and inner_relay_should_requeue(
            msg.message_type
        ):
            self.bus.send_direct(self.agent_id, msg)
            return
        try:
            if cognitive_phases_enabled():
                gen = self._process_message_phased(msg, role="conversant")
                try:
                    while True:
                        next(gen)
                except StopIteration:
                    pass
                completed = True
            else:
                completed = self._process_message(msg, role="conversant")
        except Exception as exc:
            print(
                f"[Persona:{self.agent_id}] relay processing failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            if cognitive_phases_enabled():
                _k = self._cognitive_phase_key(msg)
                with self._cognitive_phase_lock:
                    self._cognitive_phase_gens.pop(_k, None)
                    self._cognitive_phase_msgs.pop(_k, None)
            return
        if completed:
            self._maybe_dialogue_reply(msg)

    def _handle_reconcile_update(self, msg: Message):
        """Absorb reconciliation or dream results into working memory."""
        summary = msg.content if isinstance(msg.content, dict) else {}

        if msg.message_type == "dream_update":
            narrative = summary.get("narrative", "")
            theme = summary.get("theme", "")
            dream_content = summary.get("dream_content", "")
            if narrative:
                self.working_memory.add(
                    content=f"Dream insight: {narrative[:70]}",
                    source="dream",
                    salience=0.6,
                    item_type="insight",
                    tags=["dream", "evolution", "dream_narrative"],
                    cycle=self.shared.cycle_count,
                )
            if theme:
                self.working_memory.add(
                    content=f"Dream theme: {theme[:60]}",
                    source="dream",
                    salience=0.55,
                    item_type="insight",
                    tags=["dream", "dream_theme"],
                    cycle=self.shared.cycle_count,
                )
            if dream_content and not narrative:
                self.working_memory.add(
                    content=f"Dream symbol: {dream_content[:60]}",
                    source="dream",
                    salience=0.5,
                    item_type="insight",
                    tags=["dream", "dream_symbol"],
                    cycle=self.shared.cycle_count,
                )
            return

        self._last_reconcile_bus_payload = summary

        from core.Reconciliation import belief_cohesion_summary_lines

        for line in belief_cohesion_summary_lines(summary):
            self.working_memory.add(
                content=line[:120],
                source="reconciliation",
                salience=0.58,
                item_type="insight",
                tags=["reconcile", "belief_cohesion", "self_model"],
                cycle=self.shared.cycle_count,
            )

        merged_domains = [
            k for k, v in summary.items() if isinstance(v, dict) and v.get("merged_nodes", 0) > 0
        ]
        if not merged_domains:
            return
        insight = f"Common knowledge evolved: {', '.join(merged_domains)}"
        self.working_memory.add(
            content=insight[:80],
            source="reconciliation",
            salience=0.5,
            item_type="insight",
            tags=["reconcile", "evolution"],
            cycle=self.shared.cycle_count,
        )

    def _handle_llm_learning(self, msg: Message):
        """Absorb ``llm_learning`` bus messages into working memory."""
        data = msg.content if isinstance(msg.content, dict) else {}
        summary = data.get("summary", "")
        phrases = data.get("phrases_added", 0)
        if summary:
            self.working_memory.add(
                content=f"Language learning: {summary[:100]}",
                source="llm_learning",
                salience=0.4,
                item_type="insight",
                tags=["llm_learning", "language", "background"],
                cycle=self.shared.cycle_count,
            )
        if phrases > 0:
            self.working_memory.add(
                content=f"Learned {phrases} new phrase(s) from reflection",
                source="llm_learning",
                salience=0.35,
                item_type="insight",
                tags=["llm_learning", "phrase_growth"],
                cycle=self.shared.cycle_count,
            )

    def _maybe_dialogue_reply(self, msg: Message):
        """Send a reply back to the relay originator, enabling multi-turn dialogue.

        At max dialogue depth, instead of replying further, publish the
        dialogue conclusion as a ``dialogue_insight`` for goal synthesis.
        """
        if not self._running:
            return
        try:
            depth = int(msg.metadata.get("dialogue_depth", 0))
        except (TypeError, ValueError):
            depth = 0

        response_text = self._last_response_payload.get("response", "")

        if depth >= self._MAX_DIALOGUE_DEPTH:
            self._publish_dialogue_insight(msg, response_text, depth)
            return

        origin = msg.metadata.get("dialogue_origin") or msg.sender_id
        if origin == self.agent_id:
            return
        if not self._boot_agent:
            return

        if not response_text or len(response_text) < 5:
            return

        next_depth = depth + 1
        is_final = next_depth >= self._MAX_DIALOGUE_DEPTH

        reply_msg = Message(
            sender_id=self.agent_id,
            channel="inter_persona",
            content={
                "text": response_text[:300],
                "tags": ["dialogue_reply", f"depth:{next_depth}"],
                "nlu_base": {"intent": "dialogue", "sentiment": {"polarity": 0.0}},
            },
            message_type="dialogue_reply",
            metadata={
                "prior_processors": list(msg.prior_processors) + [self.agent_id],
                "dialogue_depth": next_depth,
                "dialogue_origin": origin,
                "relay_context": {
                    "from_persona": self.agent_id,
                    "from_persona_name": self.persona_name,
                    "action_text": response_text[:200],
                    "outcome_score": self._prev_outcome_score,
                },
            },
        )
        self.bus.send_direct(origin, reply_msg)
        print(
            f"[Persona:{self.agent_id}] DIALOGUE REPLY -> {origin} "
            f"(depth={next_depth}/{self._MAX_DIALOGUE_DEPTH}): "
            f"{response_text[:50]}",
            flush=True,
        )

        if is_final:
            self._publish_dialogue_insight(msg, response_text, next_depth)

    def _publish_dialogue_insight(self, msg: Message, conclusion: str, depth: int):
        """Publish the conclusion of a multi-turn dialogue for goal synthesis."""
        if not conclusion or len(conclusion) < 10:
            return
        original_topic = ""
        relay = msg.metadata.get("relay_context", {})
        if isinstance(relay, dict):
            original_topic = relay.get("action_text", "")[:80]

        insight_msg = Message(
            sender_id=self.agent_id,
            channel="dialogue_insight",
            content={
                "text": conclusion[:200],
                "topic": original_topic,
                "depth": depth,
                "persona": self.persona_name,
            },
            message_type="dialogue_insight",
        )
        self.bus.publish(insight_msg)
        print(
            f"[Persona:{self.agent_id}] DIALOGUE INSIGHT published "
            f"(depth={depth}): {conclusion[:50]}",
            flush=True,
        )

    # -- affinity scoring ----------------------------------------------

    def _evaluate_affinity(self, msg: Message) -> float:
        """Score how well this message matches this persona's affinity."""
        if self.is_default:
            return 0.5

        score = 0.5
        content = msg.content if isinstance(msg.content, dict) else {}
        nlu = content.get("nlu_base", {})

        # Emotion range matching
        sentiment = nlu.get("sentiment", {})
        emo_range = self.affinity.get("emotion_range", "all")
        if emo_range != "all" and sentiment:
            valence = sentiment.get("valence", 0.0)
            if emo_range == "positive" and valence > 0:
                score += 0.2
            elif emo_range == "negative" and valence < 0:
                score += 0.2

        return min(1.0, score)

    # -- core cognitive cycle ------------------------------------------

    _CYCLE_BUDGET = 2.0

    _INTERNAL_BUDGET = 1.0

    def _before_llm_context_io(self, enabled: bool, role: str) -> None:
        """Hook immediately before the packed LLM context phase."""
        pass

    def _after_llm_context_io(self, enabled: bool, role: str) -> None:
        """Hook immediately after the packed LLM context phase (always paired)."""
        pass

    def _complete_input_goal_from_message(self, msg: Message) -> None:
        """Complete FIFO input goal (Fuel / Atomos). Idempotent if already done."""
        content_data = msg.content if isinstance(msg.content, dict) else {}
        gid = content_data.get("input_goal_id")
        if not gid:
            return
        try:
            self.shared.goal.complete_input_goal(gid)
        except Exception as exc:
            print(
                f"[Persona:{self.agent_id}] complete_input_goal error: {exc}",
                flush=True,
            )

    def _yield_after_cycle_if_input_pipeline_busy(self) -> None:
        """After a cognitive cycle, if HTTP/text input is still active, briefly pause.

        Uses :func:`mind.chat_priority.input_pipeline_yield_busy` (not full
        :func:`~mind.chat_priority.input_pipeline_busy`) so a **sensor-only** backlog
        does not trigger multi-second sleeps. Sleeps
        ``HAROMA_POST_CYCLE_INPUT_BUSY_SLEEP_SEC`` (else legacy
        ``HAROMA_POST_CYCLE_CHAT_BUSY_SLEEP_SEC``, default ``0.05``) between checks.
        ``HAROMA_POST_CYCLE_INPUT_BUSY_MAX_RETRIES`` (else legacy
        ``HAROMA_POST_CYCLE_CHAT_BUSY_MAX_RETRIES``, default ``1``) caps sleeps per
        cycle; ``0`` = retry until the pipeline is idle.
        """
        try:
            s = self.shared
            if s is None:
                return
            _boot = getattr(self, "_boot_agent", None)
            if not input_pipeline_yield_busy(s, _boot):
                return
            _raw_sleep = os.environ.get("HAROMA_POST_CYCLE_INPUT_BUSY_SLEEP_SEC")
            if _raw_sleep is None or str(_raw_sleep).strip() == "":
                _raw_sleep = os.environ.get("HAROMA_POST_CYCLE_CHAT_BUSY_SLEEP_SEC", "0.05")
            try:
                sec = float(_raw_sleep or "0.05")
            except (TypeError, ValueError):
                sec = 0.05
            if sec <= 0:
                return
            _raw_max = os.environ.get("HAROMA_POST_CYCLE_INPUT_BUSY_MAX_RETRIES")
            if _raw_max is None or str(_raw_max).strip() == "":
                _raw_max = os.environ.get("HAROMA_POST_CYCLE_CHAT_BUSY_MAX_RETRIES", "1")
            try:
                max_retries = int(_raw_max or "1")
            except (TypeError, ValueError):
                max_retries = 1
            retries = 0
            while input_pipeline_yield_busy(s, _boot):
                time.sleep(sec)
                retries += 1
                if not input_pipeline_yield_busy(s, _boot):
                    break
                if max_retries == 0:
                    continue
                if retries >= max_retries:
                    break
        except Exception:
            pass

    def _unphased_trueself_http_traced_conversant(self, msg: Message, role: str) -> bool:
        """TrueSelf + HTTP trace + conversant: run full cycle in one call (no phased ticks).

        Phased mode schedules work across agent ticks (``_advance_cognitive_phases_on_tick``
        runs before ``poll()``), which adds multi-second latency to ``/chat`` even
        when the LLM is dummy. Specialists keep phased scheduling; TrueSelf user
        replies should complete in one ``_handle_input`` stack.
        """
        if role != "conversant":
            return False
        if getattr(self, "AGENT_TYPE", None) != "trueself":
            return False
        if trace_id_from_message(msg) is None:
            return False
        return env_flag("HAROMA_TRUESELF_HTTP_CHAT_UNPHASED", True)

    def _process_message(self, msg: Message, role: str = "observer") -> bool:
        """Run the full cognitive cycle on a message.

        Returns True when the cycle finished (or phased mode is off). When
        ``HAROMA_PERSONA_PHASED_CYCLE`` is enabled, returns False until the last
        phase completes; pending work continues via :meth:`_advance_cognitive_phases_on_tick`.
        """
        if self._unphased_trueself_http_traced_conversant(msg, role):
            gen = self._process_message_phased(msg, role)
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass
            return True
        if not cognitive_phases_enabled():
            gen = self._process_message_phased(msg, role)
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass
            return True
        key = self._cognitive_phase_key(msg)
        with self._cognitive_phase_lock:
            gen = self._cognitive_phase_gens.get(key)
            if gen is None:
                gen = self._process_message_phased(msg, role)
                self._cognitive_phase_gens[key] = gen
                self._cognitive_phase_msgs[key] = msg
        steps = phase_steps_per_invocation()
        for _ in range(steps):
            with self._cognitive_phase_lock:
                gen = self._cognitive_phase_gens.get(key)
                if gen is None:
                    return True
            try:
                next(gen)
            except StopIteration:
                with self._cognitive_phase_lock:
                    self._cognitive_phase_gens.pop(key, None)
                    msg_done = self._cognitive_phase_msgs.pop(key, None)
                if msg_done is not None:
                    self._on_phased_cycle_finished(msg_done)
                return True
        return False

    def _on_phased_cycle_finished(self, msg: Message) -> None:
        """Notify TrueSelf when a delegated phased cycle finishes on a later tick."""
        if msg.channel == "trueself_delegate":
            self._send_response_to_trueself(msg)

    def _cognitive_phase_key(self, msg: Message) -> str:
        return f"{self.agent_id}:{msg.message_id}"

    def _advance_cognitive_phases_on_tick(self) -> None:
        if not cognitive_phases_enabled():
            return
        steps = phase_steps_per_invocation()
        with self._cognitive_phase_lock:
            keys = list(self._cognitive_phase_gens.keys())
        for key in keys:
            for _ in range(steps):
                with self._cognitive_phase_lock:
                    gen = self._cognitive_phase_gens.get(key)
                    if gen is None:
                        break
                try:
                    next(gen)
                except StopIteration:
                    with self._cognitive_phase_lock:
                        self._cognitive_phase_gens.pop(key, None)
                        msg_done = self._cognitive_phase_msgs.pop(key, None)
                    if msg_done is not None:
                        self._on_phased_cycle_finished(msg_done)
                    break

    def _process_message_phased(self, msg: Message, role: str = "observer") -> Iterator[str]:
        """Generator implementing the cognitive cycle; yields phase ids when phased mode is on.

        This is the heart of Elarion -- extracted from
        ElarionController.run_cycle and adapted for per-persona state.

        The cycle has a hard budget of _CYCLE_BUDGET seconds.  When the
        budget is exceeded, remaining optional modules are skipped.
        Internal dialogue messages get a tighter budget.
        """
        s = self.shared
        t0 = time.time()
        self._cycle_count += 1
        cycle_id = s.next_cycle()

        content_data = msg.content if isinstance(msg.content, dict) else {}
        _trace_slot = msg.response_slot
        if _trace_slot is None and getattr(msg, "metadata", None):
            _trace_slot = msg.metadata.get("_chat_latency")
        debug_recall_flag = json_bool(content_data.get("debug_recall"), False)
        communication_debug_flag = json_bool(content_data.get("communication_debug"), False)
        deliberative_flag = json_bool(content_data.get("deliberative"), False)
        _cycle_depth = str(content_data.get("cycle_depth", "normal")).lower()
        if _cycle_depth == "fast":
            _cycle_depth = "normal"
        if _cycle_depth != "normal":
            _cycle_depth = "normal"
        if env_flag("HAROMA_COGNITIVE_LOOP_BEGIN", False):
            print(
                f"[CognitiveLoop] begin persona={self.agent_id} "
                f"local_iter={self._cycle_count} global_cycle={cycle_id} "
                f"depth={_cycle_depth} role={role}",
                flush=True,
            )
        _chat_llm_primary = str(
            os.environ.get("HAROMA_CHAT_LLM_PRIMARY", "1") or "1",
        ).strip().lower() in ("1", "true", "yes", "on")

        is_internal = msg.message_type in ("inner_dialogue", "dialogue_reply")
        if is_internal:
            budget = self._INTERNAL_BUDGET
        else:
            budget = self._CYCLE_BUDGET

        _cycle_trace = os.environ.get("ELARION_CYCLE_TRACE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        def _over_budget():
            if internal_treat_as_over_budget(is_internal, self.shared):
                return True
            return (time.time() - t0) > budget

        def _step_time(label, since=None):
            if _cycle_trace:
                elapsed = time.time() - (since or t0)
                print(f"  [{self.agent_id}] {label}: {elapsed:.2f}s", flush=True)
            trace_span(_trace_slot, label)
            return time.time()

        text = content_data.get("text", "")
        nlu_base = content_data.get("nlu_base", {})
        embedding = content_data.get("embedding")
        _session_uid = sanitize_user_id(content_data.get("user_id"))
        _display_name_in = content_data.get("display_name") or content_data.get("displayName")
        if _session_uid and _display_name_in:
            try:
                s.set_user_display_name(_session_uid, str(_display_name_in))
            except Exception:
                pass
        _speaker_key = speaker_key(role, _session_uid)
        _user_tag = user_tag(_session_uid)
        tags = content_data.get("tags", [])
        if _user_tag:
            if isinstance(tags, list):
                if _user_tag not in tags:
                    tags = list(tags) + [_user_tag]
            else:
                tags = [_user_tag]
        sensor_data = content_data.get("sensor_data") or {}
        has_external = bool(text) or bool(sensor_data)
        _trueself_agent = getattr(self, "AGENT_TYPE", None) == "trueself"
        _src_lc = str(content_data.get("source") or "").lower()
        # TrueSelf: only run expensive packed LLM for real user /chat (source=user or HTTP trace).
        # Sensor/vision ticks still run perception/memory but otherwise monopolize generate_chat for ~120s.
        _user_or_traced_turn = _src_lc == "user" or trace_id_from_message(msg) is not None
        _input_goal_id: Optional[str] = content_data.get("input_goal_id")
        _first_encounter_asks_name = False

        _pipe_tid = trace_id_from_message(msg)
        # TrueSelf + real user /chat: wall-clock budget for work *before* packed LLM (including
        # the first two neural slices: encounter embedding, backbone encode, discourse/ToM when
        # time remains). Default 1.2s toward ~1–2s end-to-end with a fast decode.
        # Set HAROMA_TRUESELF_USER_CHAT_BUDGET_SEC=0 to use the normal persona budget (2s).
        if (
            _trueself_agent
            and role == "conversant"
            and _user_or_traced_turn
            and not is_internal
        ):
            _raw_ub = os.environ.get("HAROMA_TRUESELF_USER_CHAT_BUDGET_SEC")
            if _raw_ub is None:
                _ub = 1.2
            else:
                _s = str(_raw_ub).strip().lower()
                if _s in ("0", "false", "no", "off", ""):
                    _ub = None
                else:
                    try:
                        _ub = float(_s)
                    except (TypeError, ValueError):
                        _ub = 1.2
                if _ub is not None and _ub <= 0:
                    _ub = None
            if _ub is not None:
                budget = min(budget, _ub)
        # TrueSelf + HTTP trace: skip ``persona_cycle_lock`` so /chat does not wait behind
        # primary/analyst neural sections (still uses shared neural read lock vs training).
        _trueself_traced_fast = _trueself_agent and bool(_pipe_tid)
        # Tight pre-LLM path: honor ``HAROMA_TRUESELF_USER_CHAT_BUDGET_SEC`` inside the first
        # two neural slices (encounter/discourse/interlocutor were previously unconditional).
        _ts_http_fast = (
            _trueself_agent
            and bool(_pipe_tid)
            and role == "conversant"
            and not is_internal
            and _user_or_traced_turn
        )
        if is_internal:
            try:
                if int(getattr(self.shared, "http_chat_inflight", 0) or 0) > 0:
                    log_input_pipeline(
                        "persona.defer_inner_cycle_http_chat",
                        trace_id=_pipe_tid,
                        detail=f"agent={self.agent_id}",
                    )
                    self.bus.send_direct(self.agent_id, msg)
                    return
            except Exception:
                pass
        log_input_pipeline(
            "persona.before_neural_sync",
            trace_id=_pipe_tid,
            detail=(
                f"agent={self.agent_id} role={role} depth={_cycle_depth} "
                f"trueself_skip_persona_gate={_trueself_traced_fast}"
            ),
        )
        _sync_ctx = (
            s.neural_sync() if _trueself_traced_fast else s.persona_neural_section()
        )
        with _sync_ctx:
            log_input_pipeline("persona.inside_neural_sync", trace_id=_pipe_tid)
            trace_span(_trace_slot, "persona_neural_sync_enter")
            episode = EpisodeContext(cycle_id=cycle_id, role=role)
            episode.bind_agent_environment(s.get_agent_environment_snapshot())
            _eid = content_data.get("experiment_id")
            if _eid is not None and str(_eid).strip():
                episode.experiment_id = str(_eid).strip()[:200]
            _lrid = getattr(s, "lab_run_id", None)
            episode.lab_run_id = str(_lrid).strip() if _lrid else None
            episode.trace("0.start")
            self.workspace.clear()
            # Default before 10.5 drives — avoids UnboundLocalError if an
            # exception aborts the sync block before ``s.drives.update``.
            drive_state: Dict[str, Any] = {}

            # -- 0. Working memory tick ------------------------------------
            self.working_memory.tick(cycle_id)
            _st = _step_time("0-wm_tick")

            # -- 1. Persona-specific perception ----------------------------
            symbolic_input = {}
            nlu_result = dict(nlu_base)
            _sensor_non_chat = {
                k: v
                for k, v in (sensor_data or {}).items()
                if str(k).strip().lower() != "chat"
            }
            _lite_http_perc = (
                _ts_http_fast
                and env_flag("HAROMA_TRUESELF_USER_CHAT_LITE_PERCEPTION", True)
                and not _sensor_non_chat
            )
            if _lite_http_perc:
                symbolic_input = {"content": text, "tags": tags, "nlu": nlu_base}
            elif (text or sensor_data) and s.perception:
                try:
                    percept_input = {"content": text, "tags": tags}
                    for sensor_channel, readings in sensor_data.items():
                        percept_input[sensor_channel] = readings
                    symbolic_input = s.perception.perceive(
                        percept_input,
                        channel="multimodal" if sensor_data else "text",
                    )
                    nlu_result = symbolic_input.get("nlu", nlu_base)
                except Exception as exc:
                    print(
                        f"[Persona:{self.agent_id}] perception failed "
                        f"({type(exc).__name__}: {exc}); using nlu_base",
                        flush=True,
                    )
                    symbolic_input = {"content": text, "tags": tags, "nlu": nlu_base}
            else:
                symbolic_input = {"content": text, "tags": tags, "nlu": nlu_base}

            if sensor_data:
                symbolic_input["sensor_data"] = sensor_data
            _sttd = content_data.get("sensor_text_translation_digest")
            if isinstance(_sttd, str) and _sttd.strip():
                symbolic_input["sensor_text_translation_digest"] = _sttd.strip()
            _senses_np_bundle = content_data.get("senses_numpy")
            if isinstance(_senses_np_bundle, dict):
                symbolic_input["senses_numpy"] = _senses_np_bundle

            content = symbolic_input.get("content", text)

            nlu_result = enrich_nlu_for_kg(nlu_result, content or text or "")
            symbolic_input["nlu"] = nlu_result
            if _user_tag:
                _sym_tags = list(symbolic_input.get("tags") or [])
                if _user_tag not in _sym_tags:
                    _sym_tags.append(_user_tag)
                    symbolic_input["tags"] = _sym_tags
            _ground = (content or text or "").strip()
            if _ground and getattr(s, "meaning_lexicon", None) and not (
                _ts_http_fast and _over_budget()
            ):
                _tm = s.meaning_lexicon.match_in_text(_ground, max_hits=3)
                if _tm:
                    symbolic_input["taught_meanings"] = _tm
            episode.bind_perception(symbolic_input)

            # Record encounter in persona's memory branch
            if not (_ts_http_fast and _over_budget()):
                node = s.memory.create_node_from(symbolic_input)
                branch_name = f"{AGENT_PREFIX}{self.agent_id}"
                s.memory.add_node("encounter_tree", branch_name, node)

            if has_external and content and not (_ts_http_fast and _over_budget()):
                self.working_memory.add(
                    content=content[:80],
                    source="perception",
                    salience=0.7,
                    item_type="percept",
                    tags=symbolic_input.get("tags", []),
                    cycle=cycle_id,
                )

            # -- 1.4. Neural embedding (reuse from InputAgent if available)
            #     Internal dialogue skips expensive embedding to stay under budget.
            #     InputAgent uses the global encoder; drop precomputed vec if this persona's backbone differs.
            _pe = self._persona_encoder()
            current_embedding = embedding
            if current_embedding is not None and _pe is not None and not is_cognitive_null(_pe):
                try:
                    _ge = getattr(s, "encoder", None)
                    if _ge is not None and not is_cognitive_null(_ge):
                        if getattr(_pe, "semantic_backbone_id", "") != getattr(
                            _ge, "semantic_backbone_id", ""
                        ):
                            current_embedding = None
                except Exception:
                    pass
            _encoder_real = _is_real_module(_pe)
            _skip_ts_embed = _ts_http_fast and env_flag(
                "HAROMA_TRUESELF_USER_CHAT_SKIP_PERSONA_ENCODE", True
            )
            if (
                current_embedding is None
                and content
                and _encoder_real
                and not is_internal
                and not _skip_ts_embed
                and not (_ts_http_fast and _over_budget())
            ):
                current_embedding = _pe.encode(content)

            # Ensure embedding matches backbone's expected dimension (always 1-D)
            _embed_dim = self._safe_embed_dim(_pe)
            if current_embedding is not None:
                if hasattr(current_embedding, "detach"):
                    current_embedding = current_embedding.detach().cpu().flatten().numpy()
                elif isinstance(current_embedding, np.ndarray):
                    current_embedding = current_embedding.astype(np.float32).flatten()
                else:
                    current_embedding = np.asarray(current_embedding, dtype=np.float32).flatten()
                expected_dim = _embed_dim
                actual_dim = int(current_embedding.shape[0])
                if actual_dim != expected_dim:
                    if actual_dim > expected_dim:
                        current_embedding = current_embedding[:expected_dim]
                    else:
                        padded = np.zeros(expected_dim, dtype=np.float32)
                        padded[:actual_dim] = current_embedding
                        current_embedding = padded

            _st = _step_time("1-perception+embed")

        # TrueSelf HTTP /chat: after slice 1, skip heavy slice-2 work if user-chat budget already exceeded.
        _ts_skip_neural_b_heavy = _ts_http_fast and _over_budget()

        # Second short neural read (shared lock budget): gate, backbone, self-model, discourse.
        _sync_ctx_b = (
            s.neural_sync() if _trueself_traced_fast else s.persona_neural_section()
        )
        with _sync_ctx_b:
            # -- 1.45. Process gate ----------------------------------------
            _prev_emo = self._prev_episode_payload.get("affect", {})
            _gate_features = self._build_gate_features(
                has_external, _prev_emo, symbolic_input, current_embedding, nlu_result
            )

            _bb_snapshot = self._build_backbone_snapshot(
                current_embedding, _prev_emo, nlu_result, has_external
            )
            _bb = getattr(s, "backbone", None)
            _z_t = None
            if (
                not _ts_skip_neural_b_heavy
                and _bb is not None
                and getattr(_bb, "available", False)
                and getattr(_bb, "_encoder", None) is not None
                and _bb.learned_weight >= 0.01
            ):
                _ce_list = list(current_embedding) if current_embedding is not None else None
                _z_t = _bb.encode_state(_bb_snapshot, content_embedding=_ce_list)

            _conv_skip = None
            if role == "conversant":
                _conv_skip = s.process_gate._CONVERSANT_SKIP
            _gate_decisions = s.process_gate.decide(_gate_features, z_t=_z_t, force_off=_conv_skip)

            episode.bind_nlu(nlu_result)

            if _ts_skip_neural_b_heavy:
                # Budget already spent in slice 1 — skip self-model/discourse/ToM for latency.
                self_prediction = {}
                _discourse_result = None
                _discourse_frames_dicts = []
                interlocutor_snapshot = {}
                _mental_prediction = {}
            else:
                # -- 1.5. Self-prediction --------------------------------------
                self_prediction = {}
                if _gate_decisions.get("self_prediction", True):
                    self_prediction = (
                        s.self_model.predict(current_embedding, self._prev_cycle_state) or {}
                    )
                if self_prediction:
                    episode.bind_self_prediction(self_prediction)

                # -- 1.6. NLU + Discourse --------------------------------------
                _discourse_result = None
                _discourse_frames_dicts = []
                if nlu_result:
                    _conv_history = [
                        t.to_dict()
                        for t in self.conversation.get_recent(
                            5, speaker=_speaker_key if _session_uid else None
                        )
                    ]
                    _discourse_result = s.discourse.process(
                        nlu_result,
                        conversation_history=_conv_history,
                        cycle_id=cycle_id,
                    )
                    _discourse_frames_dicts = [f.to_dict() for f in _discourse_result.frames]

                # -- 1.7. Interlocutor model + mental simulator ----------------
                interlocutor_snapshot = {}
                _mental_prediction = {}
                if _gate_decisions.get("interlocutor_model", True):
                    if has_external and content:
                        s.interlocutor_model.update(
                            speaker=_speaker_key,
                            content=content,
                            nlu_result=nlu_result,
                            cycle_id=cycle_id,
                            discourse_frames=_discourse_frames_dicts or None,
                        )
                        observed_action = nlu_result.get("intent", "speak")
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
                        s.mental_simulator.update_model(
                            agent_id=_speaker_key,
                            observed_action=observed_mapped,
                            discourse_frames=_discourse_frames_dicts or None,
                            context={
                                "content": content,
                                "sentiment": nlu_result.get("sentiment", {}),
                            },
                            cycle_id=cycle_id,
                        )
                        if _session_uid and role == "conversant":
                            _st_ix = s.interlocutor_model.speakers.get(_speaker_key)
                            if _st_ix is not None and _st_ix.interaction_count == 1:
                                if not s.get_user_display_name(_session_uid):
                                    _first_encounter_asks_name = True
                    interlocutor_snapshot = s.interlocutor_model.get_model_summary(_speaker_key)
                    _mental_prediction = s.mental_simulator.predict_behavior(_speaker_key)
                    interlocutor_snapshot["mental_prediction"] = _mental_prediction
                    episode.bind_theory_of_mind(interlocutor_snapshot)

            _st = _step_time("1.7-interlocutor")

        if not self._running:
            self._last_response_payload = {
                "response": "[cycle aborted: shutting down]",
                "cycle": cycle_id,
                "affect": {},
                "strategy": "shutdown",
                "persona": self.agent_id,
                "persona_name": self.persona_name,
            }
            slot = msg.response_slot
            if slot and not slot["event"].is_set():
                slot["result"] = self._last_response_payload
                slot["event"].set()
            self._complete_input_goal_from_message(msg)
            self._yield_after_cycle_if_input_pipeline_busy()
            return

        if cognitive_phases_enabled():
            yield COGNITIVE_PHASE_NEURAL_GATE

        # -- 2. Semantic recall (budget-aware) -------------------------
        query_tags = symbolic_input.get("tags", [])
        if isinstance(query_tags, list) and isinstance(tags, list):
            seen = set(query_tags)
            query_tags = list(query_tags) + [t for t in tags if t not in seen]
        query_text = content or text
        _teaching = is_teaching_turn(query_text, nlu_result)
        recall_limit = self._prev_modulation.get("recall_limit", 20)
        if is_internal:
            recall_limit = min(recall_limit, 5 if _teaching else 3)
        elif _over_budget():
            recall_limit = min(recall_limit, 7 if _teaching else 5)
        elif _teaching:
            recall_limit = min(recall_limit + 10, 28)
        if (
            _trueself_agent
            and role == "conversant"
            and _user_or_traced_turn
            and not is_internal
        ):
            _rl_raw = str(os.environ.get("HAROMA_TRUESELF_USER_CHAT_RECALL_LIMIT", "12") or "").strip()
            if _rl_raw not in ("0", ""):
                try:
                    _rl_cap = int(_rl_raw)
                except (TypeError, ValueError):
                    _rl_cap = 12
                if _rl_cap > 0:
                    recall_limit = min(recall_limit, _rl_cap)
        if skip_semantic_recall_for_internal(is_internal, s):
            recalled = []
        elif (
            _ts_http_fast
            and not is_internal
            and env_flag("HAROMA_TRUESELF_USER_CHAT_FAST_RECALL", True)
            and hasattr(s.memory, "recall_fast")
        ):
            # Full ``recall()`` hybrid path can take 10s+ on large forests (51k+ nodes);
            # FAISS-only ``recall_fast`` keeps HTTP /chat under the user-chat budget.
            _rf_lim = max(1, min(recall_limit, 8))
            recalled = s.memory.recall_fast(query_text or "", limit=_rf_lim)
            recalled = s.memory.merge_recall_with_prime(
                recalled, _rf_lim, fast_cycle=True
            )
        else:
            recalled = s.memory.recall(
                query_tags=query_tags,
                limit=recall_limit,
                query_text=query_text,
            )
            recalled = s.memory.merge_recall_with_prime(
                recalled, recall_limit, fast_cycle=False
            )
            if has_external and should_merge_web_learn(
                query_text, nlu_result or {}, teaching=_teaching
            ):
                _wlim = web_learn_inject_max()
                if _wlim > 0:
                    recalled = merge_web_learn_tail(
                        s.memory, recalled, recall_limit, max_inject=_wlim
                    )
            if not input_pipeline_busy(s, getattr(self, "_boot_agent", None)):
                try:
                    auto_nodes = s.memory.get_nodes("thought_tree", "autonomy")
                    tail = auto_nodes[-4:] if len(auto_nodes) > 4 else auto_nodes
                    if tail:
                        seen = {n.moment_id for n in recalled}
                        prefix = [n for n in reversed(tail) if n.moment_id not in seen]
                        if prefix:
                            recalled = (prefix + list(recalled))[:recall_limit]
                            try:
                                s.autonomy_bump("recall_autonomy_injected", len(prefix))
                            except Exception as _ab:
                                print(
                                    f"[Persona:{self.agent_id}] autonomy_bump error: {_ab}",
                                    flush=True,
                                )
                except Exception as _ar:
                    print(
                        f"[Persona:{self.agent_id}] autonomy recall merge error: {_ar}", flush=True
                    )
        episode.bind_memories(recalled)

        # LLM-centric / board goal-voice: consult packed LLM only when there is
        # no active board mandate (CEO executing). Disabled on internal cycles.
        _llm_centric = llm_centric_enabled_for_persona_cycle(
            is_internal=is_internal,
            trueself_agent=_trueself_agent,
            user_or_traced_turn=_user_or_traced_turn,
            goal_board=getattr(s, "goal_board", None),
        )
        _memory_forest_seed = ""

        # -- 2.5. KG integration --------------------------------------
        if _gate_decisions.get("kg_integration", True) and not _over_budget():
            if nlu_result and (nlu_result.get("entities") or nlu_result.get("relations")):
                s.knowledge.integrate(nlu_result, cycle_id=cycle_id)
        knowledge_summary = s.knowledge.summary()
        knowledge_diff = s.knowledge.diff()
        episode.bind_knowledge(knowledge_summary)

        _st = _step_time("2-recall+kg")

        # -- 3. Feel (persona-specific emotion) ------------------------
        _neuro_scale = 1.0 + ((self.personality.get("neuroticism") or 0.5) - 0.5) * 0.4
        emotion_input = {**{"content": text, "tags": tags}, **symbolic_input}
        if "intensity" in emotion_input:
            emotion_input["intensity"] = emotion_input["intensity"] * _neuro_scale
        self.emotion.ingest(emotion_input)
        emotion_summary = self.emotion.summarize()
        episode.bind_affect(emotion_summary)

        if has_external and content:
            self.conversation.record_input(
                content=content,
                speaker=_speaker_key,
                cycle_id=cycle_id,
                emotion=episode.affect["dominant_emotion"],
                tags=symbolic_input.get("tags", []),
            )
            if nlu_result and _discourse_result is not None:
                self.conversation.store_discourse_snapshot(
                    cycle_id, _discourse_result.to_dict()
                )

        # -- 4-5. Global workspace broadcast + select ------------------
        ws_contents = []
        attention_ctx = s.attention._build_context(
            valence=emotion_summary.get("valence", 0.0),
            arousal=emotion_summary.get("arousal", 0.0),
            curiosity=self._safe_curiosity(s.curiosity),
            wm_load=self.working_memory.occupancy(),
            dominant_drive_level=self._safe_drive_level(s.drives),
            outcome_prev=self._prev_outcome_score,
            cycle_count=cycle_id,
            has_external=float(has_external),
        )

        def _workspace_salience(source: str, base: float) -> float:
            return s.attention.adjust_salience(source, base, attention_ctx, z_t=_z_t)

        if not _over_budget():
            n_mod = len(symbolic_input.get("modalities", ["text"]))
            base_p_sal = (
                0.5 + (0.3 if symbolic_input.get("tags") else 0.0) + (0.1 * min(n_mod - 1, 3))
            )
            self.workspace.broadcast(
                "perception",
                symbolic_input,
                salience=_workspace_salience("perception", base_p_sal),
            )
            self.workspace.broadcast(
                "memory",
                {"recalled": episode.recalled_memories},
                salience=_workspace_salience("memory", episode.memory_influence),
            )
            if not input_pipeline_busy(s, getattr(self, "_boot_agent", None)):
                try:
                    stim = s.pop_autonomous_stimulus(self.agent_id)
                    if stim:
                        self.workspace.broadcast(
                            "autonomy_stimulus",
                            stim,
                            salience=_workspace_salience("autonomy_stimulus", 0.72),
                        )
                    auto_last = s.memory.get_nodes("thought_tree", "autonomy")
                    if auto_last:
                        self.workspace.broadcast(
                            "autonomy",
                            {
                                "recent": (auto_last[-1].content or "")[:320],
                                "branch": "autonomy",
                            },
                            salience=_workspace_salience("autonomy", 0.52),
                        )
                except Exception as _aw:
                    print(
                        f"[Persona:{self.agent_id}] autonomy workspace broadcast error: {_aw}",
                        flush=True,
                    )
            self.workspace.broadcast(
                "emotion",
                emotion_summary,
                salience=_workspace_salience("emotion", emotion_summary.get("intensity", 0.0)),
            )
            if knowledge_diff.get("changed"):
                base_kg_sal = min(1.0, knowledge_diff.get("knowledge_gain", 0.0) + 0.3)
                self.workspace.broadcast(
                    "knowledge",
                    {
                        "new_entities": knowledge_diff.get("new_entities", 0),
                        "new_relations": knowledge_diff.get("new_relations", 0),
                    },
                    salience=_workspace_salience("knowledge", base_kg_sal),
                )

            ws_contents = self.workspace.select()
            episode.bind_workspace(ws_contents, self.workspace.get_unconscious())
            self.workspace.integrate()
            self.working_memory.promote_from_workspace(ws_contents, cycle=cycle_id)
            episode.trace("5.workspace")

        # -- 6. Temporal arc + predictive continuity -------------------
        temporal_arc = ""
        narrative_ctx = self._get_narrative_context()
        temporal_pos: Dict[str, Any] = {}
        if _gate_decisions.get("temporal_bind", True):
            temporal_pos = s.temporal.get_temporal_position(
                cycle_id,
                episode.affect.get("dominant_emotion"),
            )
            episode.bind_temporal(temporal_pos)
        if not _over_budget():
            temporal_arc = s.temporal.summarize_arc(n=15)
            if temporal_arc and len(s.temporal.episode_timeline) > 3:
                narrative_ctx = f"{narrative_ctx} {temporal_arc}"
            wp = temporal_pos.get("world_prediction") if temporal_pos else {}
            if isinstance(wp, dict) and wp.get("available"):
                dom = episode.affect.get("dominant_emotion", "neutral")
                if float(wp.get("surprise", 0.0)) >= 0.55:
                    narrative_ctx = (
                        f"{narrative_ctx} [Continuity: expected next "
                        f"affect≈{wp.get('predicted_emotion')}, "
                        f"observed {dom}.]"
                    )
                cf = wp.get("counterfactual")
                if cf and float(wp.get("surprise", 0.0)) >= 0.45:
                    narrative_ctx = f"{narrative_ctx} {cf}"
        episode.bind_narrative(narrative_ctx)

        if not self._running:
            self._last_response_payload = {
                "response": "[cycle aborted: shutting down]",
                "cycle": cycle_id,
                "affect": {},
                "strategy": "shutdown",
                "persona": self.agent_id,
                "persona_name": self.persona_name,
            }
            slot = msg.response_slot
            if slot and not slot["event"].is_set():
                slot["result"] = self._last_response_payload
                slot["event"].set()
            self._complete_input_goal_from_message(msg)
            self._yield_after_cycle_if_input_pipeline_busy()
            return

        # -- 7. Symbolic law / value bound to episode + workspace --------
        law_tags = build_law_tags(symbolic_input, content_data)
        _budget = _over_budget()
        run_law_value_myth_sidecar_phase(
            gate_enabled=_gate_decisions.get("law_value_myth_fusion", True),
            law=getattr(s, "law", None),
            law_tags=law_tags,
            episode=episode,
            workspace=self.workspace,
            attention=s.attention,
            attention_ctx=attention_ctx,
            z_t=_z_t,
            value=getattr(s, "value", None),
            myth=getattr(s, "myth", None),
            fusion=getattr(s, "fusion", None),
            dream_mgr=getattr(s, "dream_mgr", None),
            symbolic_input=symbolic_input,
            explicit=content_data,
            role=role,
            has_external=bool(has_external),
            broadcast_violations=not _budget,
            apply_explicit_bindings=not _budget,
            skip_derived_value_myth_fusion=_budget,
            trace_label="7.law_sidecar",
        )

        # -- 9. Identity -----------------------------------------------
        identity_summary = s.identity.summarize()
        if _gate_decisions.get("identity_update", True) and not _over_budget():
            s.identity.update({"content": text, "tags": tags}, role)
            identity_summary = s.identity.summarize()
        episode.bind_identity(identity_summary)

        _st = _step_time("3-9-feel+workspace+temporal+law+identity")

        # -- 10. Goals -------------------------------------------------
        if episode.affect["intensity"] > 0.5:
            s.goal.register_goal(
                f"emotional_{cycle_id}",
                f"Process {episode.affect['dominant_emotion']} "
                f"with intensity {episode.affect['intensity']:.2f}",
                priority=episode.affect["intensity"],
                source="emotion",
            )
        priorities = (
            s.goal.prioritize_workfront()
            if hasattr(s.goal, "prioritize_workfront")
            else s.goal.prioritize()
        )
        _goal_store = s.goal.engine.goals
        active_goals: List[Dict[str, Any]] = []
        _gb_goals = getattr(s, "goal_board", None)
        _msum = _gb_goals.get_mandate_summary_for_prompt() if _gb_goals is not None else None
        if isinstance(_msum, dict) and _msum.get("goal_id"):
            active_goals.append(
                {
                    "goal_id": _msum["goal_id"],
                    "priority": float(_msum.get("priority", 0.95)),
                    "description": str(_msum.get("description", ""))[:400],
                    "source": "board_mandate",
                    "board_status": _msum.get("status", "executing"),
                }
            )
        _mandate_gid = str(_msum.get("goal_id", "")).strip() if isinstance(_msum, dict) else ""
        for gid in priorities[:5]:
            if _mandate_gid and gid == _mandate_gid:
                continue
            ginfo = _goal_store[gid] if isinstance(_goal_store.get(gid), dict) else {}
            _row: Dict[str, Any] = {
                "goal_id": gid,
                "priority": ginfo.get("priority", 0.5),
                "description": ginfo.get("description", gid),
                "source": ginfo.get("source", ""),
            }
            if ginfo.get("child_goal_ids"):
                _row["child_goal_ids"] = ginfo["child_goal_ids"]
            if ginfo.get("action_items"):
                _row["action_items"] = ginfo["action_items"]
            active_goals.append(_row)
        episode.bind_goals(active_goals, urgency=min(1.0, len(priorities) / 10.0))

        # -- 10.5. Drives ---------------------------------------------
        drive_state = s.drives.update(
            episode.to_payload(),
            self._prev_episode_payload.get("action_outcome", {}),
            is_dream_cycle=False,
            has_external_input=has_external,
        )
        episode.bind_drives(drive_state)

        # -- 10.7. Appraisal ------------------------------------------
        prev_drift = self._prev_episode_payload.get("drift_score", 0.0)
        personality_summary = self.personality.summarize()
        appraisal_result = s.appraisal.evaluate(
            nlu_result=nlu_result,
            active_goals=active_goals,
            knowledge_summary=knowledge_summary,
            knowledge_diff=knowledge_diff,
            identity_summary=identity_summary,
            emotion_summary=emotion_summary,
            drift_score=prev_drift,
            action_memory_stats=s.action_memory.stats(),
            working_memory_load=self.working_memory.occupancy(),
            interlocutor=interlocutor_snapshot,
            personality=personality_summary,
        )
        episode.bind_appraisal(appraisal_result)

        if appraisal_result.get("overrides"):
            _ov_emotion = appraisal_result.get("emotion")
            _ov_intensity = appraisal_result.get("intensity")
            if _ov_emotion is not None and _ov_intensity is not None:
                self.emotion.engine.update_emotion(
                    _ov_emotion,
                    _ov_intensity,
                    emotion_input,
                )
            emotion_summary = self.emotion.summarize()
            episode.bind_affect(emotion_summary)

        # -- 10.8. Modulation ------------------------------------------
        modulation = s.modulation.compute(emotion_summary, z_t=_z_t)
        episode.bind_modulation(modulation)
        self.workspace.capacity = modulation["workspace_capacity"]

        # -- 10.85. Memory forest seed (after affect, goals, WM, drives) --
        _seed_enabled = os.environ.get("HAROMA_MEMORY_SEED_ENABLED", "1") not in (
            "0",
            "false",
            "",
        )
        _env_snapshot: Optional[Dict[str, Any]] = None
        if _llm_centric:
            _drive_mgr = getattr(s, "drives", None)
            _wm_items = self.working_memory.get_context(limit=5)
            _env_snapshot = {
                "emotion": emotion_summary if emotion_summary else {},
                "goals": active_goals or [],
                "personality": personality_summary,
                "working_memory": [
                    {"content": it.content, "salience": it.salience} for it in _wm_items
                ],
                "drives": (drive_state if isinstance(drive_state, dict) else {}),
            }
        _skip_ts_seed = _ts_http_fast and env_flag(
            "HAROMA_TRUESELF_USER_CHAT_SKIP_MEMORY_SEED", True
        )
        if _seed_enabled and hasattr(s.memory, "build_seed_context") and not _skip_ts_seed:
            try:
                _memory_forest_seed = s.memory.build_seed_context(
                    query_text=query_text or "",
                    recalled=recalled,
                    env_snapshot=_env_snapshot,
                )
            except Exception as _seed_err:
                print(
                    f"[Persona:{self.agent_id}] seed context error: {_seed_err}",
                    flush=True,
                )

        # -- 11-12. Reflect + diagnose ---------------------------------
        drift_score = 0.0
        _skip_ts_reflect = _ts_http_fast and env_flag(
            "HAROMA_TRUESELF_USER_CHAT_SKIP_REFLECT", True
        )
        if _gate_decisions.get("reflection_diagnose", True) and not _skip_ts_reflect:
            summary = s.reflector.reflect_on_state({"content": text, "tags": tags}, role)
            s.loop.log_loop("cognitive_cycle", context=summary, event="start")
            drift_result = s.drift.detect_drift(self._collect_recent_ids(limit=10))
            drift_score = (
                drift_result.get("drift_score", 0.0) if isinstance(drift_result, dict) else 0.0
            )

        _st = _step_time("10-12-goals+drives+appraisal+modulation+reflect")

        # -- 13. Curiosity (skippable when over budget) -----------------
        _prev_strategy = (
            self._prev_episode_payload.get("action", {}).get("strategy", "reflect")
            if isinstance(self._prev_episode_payload.get("action"), dict)
            else "reflect"
        )
        curiosity_result = run_curiosity_phase(
            enabled=(_gate_decisions.get("curiosity", True) and not _over_budget()),
            episode=episode,
            curiosity=s.curiosity,
            emotion_summary=emotion_summary,
            knowledge_summary=knowledge_summary,
            current_embedding=current_embedding,
            last_strategy=_prev_strategy,
            forecast_for_eval={},
            workspace_followup=False,
        )

        # -- 13.2. Reasoning (skippable) -------------------------------
        reasoning_result = run_reasoning_phase(
            enabled=(
                _gate_decisions.get("reasoning", True)
                and not _over_budget()
                and not (
                    _ts_http_fast
                    and env_flag("HAROMA_TRUESELF_USER_CHAT_SKIP_REASONING", True)
                )
            ),
            reasoning_engine=s.reasoning,
            knowledge=s.knowledge,
            active_goals=active_goals,
            nlu_result=nlu_result,
            max_depth=modulation.get("inference_cap"),
            memory=s.memory,
            episode=episode,
            law=getattr(s, "law", None),
            law_tags=law_tags,
            workspace=self.workspace,
            attention=s.attention,
            attention_ctx=attention_ctx,
            z_t=_z_t,
            gate_law_value_myth=_gate_decisions.get("law_value_myth_fusion", True),
            gate_reasoning_for_refresh=True,
            trace_label="13.2.reasoning_law",
        )

        # -- 13.2b. LLM context (outside neural_sync: generate_chat can run
        # for a long time; holding the lock stalls encoder and background train.)
        _law_mgr = getattr(s, "law", None)
        _val_mgr = getattr(s, "value", None)
        _pl = build_packed_llm_cycle_inputs(
            text=text,
            content=content,
            llm_centric=_llm_centric,
            chat_llm_primary=_chat_llm_primary,
            role=role,
            has_external=has_external,
            is_internal=is_internal,
            trueself_agent=_trueself_agent,
            user_or_traced_turn=_user_or_traced_turn,
            gate_reasoning=_gate_decisions.get("reasoning", True),
            over_budget=_over_budget(),
            deliberative_flag=deliberative_flag,
            recalled_memories=episode.recalled_memories,
            reasoning_result=reasoning_result,
            appraisal_result=appraisal_result,
            organic_skip_threshold=ORGANIC_PACKED_LLM_SKIP_THRESHOLD,
            knowledge_summary=knowledge_summary,
            knowledge_graph=s.knowledge,
            nlu_result=nlu_result,
        )
        _utt = _pl.user_text
        _path = _pl.path
        _packed_llm_eligible = _path.packed_llm_eligible
        _conversant_chat = _path.conversant_chat
        # Packed LLM (or synthetic dummy when no model): same gates for all cycle_depth /
        # sensor modes — ``cycle_depth`` is metadata only for tracing/budgets, not LLM wiring.
        _llm_primary_path = _path.llm_primary_path
        _llm_classic_path = _path.llm_classic_path
        _llm_ctx_enabled = _path.llm_ctx_enabled
        _skip_full_pack_inputs = _pl.skip_full_pack_messages
        _kg_for_llm = _pl.kg_triples
        _llm_timeout_override: Optional[float] = _pl.timeout_override
        llm_context_result: Dict[str, Any] = {"source": "skipped"}
        # Defer episode.bind_llm_context until after deliberative merge when both run,
        # so we only touch episode.llm_context once per cycle for that path.
        _defer_llm_bind = _pl.defer_episode_bind

        _law_sum, _val_sum = summarize_law_value_managers(_law_mgr, _val_mgr)

        _agent_state_json = build_agent_state_json_for_packed_llm(
            deliberative_flag=deliberative_flag,
            llm_ctx_enabled=_llm_ctx_enabled,
            law_summary=_law_sum,
            val_mgr=_val_mgr,
            value_summary=_val_sum,
            state=s,
            boot_agent_ref=getattr(self, "_boot_agent_ref", None),
            identity_summary=identity_summary,
            personality_summary=personality_summary,
            active_goals=active_goals,
            episode=episode,
        )

        _discourse_llm = discourse_context_for_packed_llm(
            self.conversation,
            cycle_id=cycle_id,
            speaker_key=_speaker_key,
            session_uid=bool(_session_uid),
            first_encounter_asks_name=_first_encounter_asks_name,
        )

        if cognitive_phases_enabled():
            yield COGNITIVE_PHASE_PRE_LLM

        log_input_pipeline(
            "persona.before_packed_llm",
            trace_id=_pipe_tid,
            detail=packed_llm_before_llm_log_detail(
                agent_id=self.agent_id,
                role=role,
                llm_ctx_enabled=_llm_ctx_enabled,
            ),
        )
        self._before_llm_context_io(_llm_ctx_enabled, role)
        try:
            llm_context_result = invoke_run_llm_context_reasoning_phase(
                pl=_pl,
                llm_backend=s.llm_backend,
                episode=episode,
                memory_forest=s.memory,
                identity_summary=identity_summary,
                personality_summary=personality_summary,
                active_goals=active_goals,
                law_summary=_law_sum,
                value_summary=_val_sum,
                discourse_context=_discourse_llm,
                nlu_result=nlu_result,
                memory_forest_seed=_memory_forest_seed,
                llm_centric=_llm_centric,
                deliberative=deliberative_flag,
                agent_state_json=_agent_state_json,
                trace_label=TRACE_LABEL_PERSONA_PACKED_LLM,
            )
        finally:
            self._after_llm_context_io(_llm_ctx_enabled, role)
        log_input_pipeline("persona.after_packed_llm", trace_id=_pipe_tid)

        if cognitive_phases_enabled():
            yield COGNITIVE_PHASE_POST_LLM

        if _defer_llm_bind:
            complete_deferred_deliberative_llm_context(
                llm_context_result,
                deliberative_flag=deliberative_flag,
                val_mgr=_val_mgr,
                episode=episode,
                active_goals=active_goals,
                drive_state=drive_state,
                emotion_summary=emotion_summary,
                log_context=f" Persona:{self.agent_id}",
            )

        # Post-LLM phases: counterfactual → response run without holding the neural RW
        # lock (short locks only for self_model / backbone / composer weight touches).
        # -- 13.3. Counterfactual (skippable) --------------------------
        counterfactual_result: Dict[str, Any] = {
            "counterfactual_depth": 0,
            "branches": [],
        }
        _cf_features: List[float] = []
        if _gate_decisions.get("counterfactual", True) and not _over_budget():
            _cf_features = build_counterfactual_gate_features(
                knowledge_diff=knowledge_diff,
                reasoning_result=reasoning_result,
                active_goals=active_goals,
                emotion_summary=emotion_summary,
                curiosity_result=curiosity_result,
                has_external=bool(has_external),
                cycle_count=cycle_id,
                counterfactual_engine=s.counterfactual,
                prev_modulation=self._prev_modulation,
            )
            counterfactual_result = run_counterfactual_phase(
                enabled=True,
                counterfactual_engine=s.counterfactual,
                knowledge_graph=s.knowledge,
                reasoning_engine=s.reasoning,
                reasoning_result=reasoning_result,
                knowledge_diff=knowledge_diff,
                active_goals=active_goals,
                nlu_result=nlu_result,
                episode=episode,
                gate_features=_cf_features,
                workspace=self.workspace,
                attention=s.attention,
                attention_ctx=attention_ctx,
                z_t=_z_t,
            )

        # -- 13.5. Metacognition (skippable) ---------------------------
        episode.bind_reconciliation(self._last_reconcile_bus_payload)
        meta_assessment, _ = run_metacognition_phase(
            enabled=(_gate_decisions.get("metacognition", True) and not _over_budget()),
            extended=False,
            metacognition=s.metacognition,
            episode=episode,
            emotion_summary=emotion_summary,
            curiosity_result=curiosity_result,
            outcome_prev=self._prev_episode_payload.get("action_outcome", {}),
        )

        # -- 13.7. Imagination (skippable) -----------------------------
        imagination_result, imagined_strategy, imagined_plan = run_imagination_phase(
            enabled=(_gate_decisions.get("imagination", True) and not _over_budget()),
            imagination=s.imagination,
            episode=episode,
            current_embedding=current_embedding,
            emotion_summary=emotion_summary,
            curiosity_result=curiosity_result,
            dominant_drive=self._safe_drive_level(s.drives),
            wm_load=self.working_memory.occupancy(),
            outcome_prev=self._prev_outcome_score,
            has_external=float(has_external),
            active_goals=active_goals,
            drive_state=drive_state,
        )
        if imagined_plan:
            episode.bind_plan(imagined_plan)
            if not self._current_plan:
                self._current_plan = list(imagined_plan)
                self._plan_step = 0
        if imagined_strategy is None and _is_real_module(s.imagination):
            try:
                imagined_strategy = s.imagination.get_strategy_recommendation()
            except Exception as _isr:
                print(
                    f"[Persona:{self.agent_id}] get_strategy_recommendation error: {_isr}",
                    flush=True,
                )

        # -- 14. Action generation -------------------------------------
        _st = _step_time("13-curiosity+reasoning+cf+meta+imagination")

        ctx_hash = s.action_memory._hash_context(episode.to_payload())
        strategy_hint = resolve_strategy_hint(
            s.action_memory.suggest_strategy(ctx_hash),
            imagined_strategy,
            self._current_plan,
            self._plan_step,
        )

        ws_dicts = workspace_contents_as_dicts(self.workspace)
        is_conv = self.conversation.is_in_conversation(cycle_id)
        last_turn = self.conversation.get_last_input()
        topic = self.conversation.get_topic()
        topic_shifted = self.conversation.detect_topic_shift(symbolic_input.get("tags", []))

        merge_derivation_artifacts(episode)

        ep_payload, _kg_triples = build_action_episode_payload(
            episode=episode,
            current_embedding=current_embedding,
            z_t=_z_t,
            knowledge_graph=s.knowledge,
            knowledge_summary=knowledge_summary,
            nlu_result=nlu_result,
            memory_forest=s.memory,
        )

        _novelty_bias = modulation.get("novelty_bias", 0.0)
        _novelty_bias += ((self.personality.get("openness") or 0.5) - 0.5) * 0.2

        _utter_style = "conversational" if role == "conversant" else None

        _llm_answer = llm_context_result.get("answer")
        _mg_env = read_multi_goal_deliberative_env()
        _multi_goal = _mg_env.enabled
        _max_cycle_goals = _mg_env.max_cycle_goals
        _max_actions_per_goal = _mg_env.max_actions_per_goal

        if _llm_centric and _llm_answer:
            action = build_llm_centric_action(
                llm_answer=_llm_answer,
                llm_context_result=llm_context_result,
                is_in_conversation=is_conv,
            )
            episode.bind_multi_goal_actions([])
        elif _multi_goal and active_goals:
            _batch = active_goals[:_max_cycle_goals]
            action, _mg_groups = run_multi_goal_deliberative_actions(
                episode=episode,
                action_generator=s.action_generator,
                ep_payload=ep_payload,
                goal_batch=_batch,
                max_actions_per_goal=_max_actions_per_goal,
                ws_dicts=ws_dicts,
                strategy_hint=strategy_hint,
                working_memory_context=self.working_memory.to_context_string(limit=4),
                conversation_context=self.conversation.get_context_summary(),
                is_in_conversation=is_conv,
                topic=topic,
                last_input_content=last_turn.content if last_turn else "",
                topic_shifted=topic_shifted,
                knowledge_summary=knowledge_summary,
                reasoning_result=reasoning_result,
                nlu_result=nlu_result,
                interlocutor=interlocutor_snapshot,
                counterfactual_result=counterfactual_result,
                novelty_bias=_novelty_bias,
                personality=personality_summary,
                utterance_style=_utter_style,
            )
            episode.bind_multi_goal_actions(_mg_groups)
        else:
            action = run_deliberative_action(
                episode=episode,
                action_generator=s.action_generator,
                ep_payload=ep_payload,
                ws_dicts=ws_dicts,
                strategy_hint=strategy_hint,
                working_memory_context=self.working_memory.to_context_string(limit=4),
                conversation_context=self.conversation.get_context_summary(),
                is_in_conversation=is_conv,
                topic=topic,
                last_input_content=last_turn.content if last_turn else "",
                topic_shifted=topic_shifted,
                knowledge_summary=knowledge_summary,
                reasoning_result=reasoning_result,
                nlu_result=nlu_result,
                interlocutor=interlocutor_snapshot,
                counterfactual_result=counterfactual_result,
                novelty_bias=_novelty_bias,
                personality=personality_summary,
                utterance_style=_utter_style,
            )
            episode.bind_multi_goal_actions([])

        # Primary packed LLM already ran; align ``action[\"text\"]`` with :mod:`mind.chat_visibility`.
        action = merge_packed_llm_answer_into_action(
            action,
            llm_context_result,
            llm_primary_path=_llm_primary_path,
            llm_centric=_llm_centric,
        )

        _st = _step_time("14-action_generation")

        # -- 15. Evaluate outcome --------------------------------------
        outcome = s.outcome_evaluator.evaluate(
            action,
            self._prev_episode_payload,
            episode.to_payload(),
            knowledge_diff=knowledge_diff,
            reasoning_result=reasoning_result,
            nlu_result=nlu_result,
            counterfactual_result=counterfactual_result,
        )
        episode.bind_action(action, outcome)
        s.action_memory.store(ctx_hash, action, outcome)
        self.emotion.engine.learn_from_cycle(emotion_input)

        # -- 15a. Plan step advancement (post-outcome, matches controller) -
        _outcome_score = outcome.get("score", 0.5)
        if self._current_plan:
            self._plan_outcomes.append(_outcome_score)
            self._plan_step += 1
            if self._plan_step >= len(self._current_plan):
                self._current_plan = []
                self._plan_step = 0
                self._plan_outcomes = []
            elif _outcome_score < 0.3:
                self._current_plan = []
                self._plan_step = 0
                self._plan_outcomes = []

        # -- 15b. Nudge personality based on outcome --------------------
        self._nudge_personality(
            _outcome_score,
            action.get("strategy", ""),
            emotion_summary.get("intensity", 0.0),
            emotion_summary.get("valence", 0.0),
        )

        _st = _step_time("15-outcome_eval")

        # -- 16. Consolidate memory (persona-specific branches) --------
        salience = episode.compute_salience()
        _exp_tags = query_tags + ["experience", f"persona:{self.agent_id}"]
        if _user_tag:
            _exp_tags = _exp_tags + [_user_tag]
        experience_node = MemoryNode(
            content=f"[cycle {cycle_id}] {episode.affect['dominant_emotion']} | {content[:80]}",
            emotion=episode.affect["dominant_emotion"],
            confidence=min(1.0, salience),
            tags=_exp_tags,
        )
        s.memory.add_node("thought_tree", branch_name, experience_node)

        _ut_reply = (content or text or "").strip()
        _vis = self._chat_visible_response(
            action,
            user_text=_ut_reply,
            identity=episode.identity,
            llm_context=getattr(episode, "llm_context", None) or {},
        )
        _act_tags: List[str] = [
            "action",
            action.get("action_type", "respond"),
            action.get("strategy", "unknown"),
            f"persona:{self.agent_id}",
            (
                "learning:user_turn"
                if is_conv and role == "conversant"
                else "learning:internal"
            ),
        ]
        if _user_tag:
            _act_tags.append(_user_tag)
        action_node = MemoryNode(
            content=_vis[:120],
            emotion=episode.affect["dominant_emotion"],
            confidence=outcome.get("score", 0.5),
            tags=_act_tags,
        )
        s.memory.add_node("action_tree", branch_name, action_node)

        # -- 16b. Post-turn forest touch + cited-node bump -------------
        _touch_policy = os.environ.get("HAROMA_MEMORY_TOUCH_POLICY", "recalled_plus_core")
        if _touch_policy != "off" and hasattr(s.memory, "touch_trees_after_turn"):
            _recalled_tree_names: Set[str] = set()
            for _rn in episode.recalled_memories:
                _tn = _rn.get("tree") if isinstance(_rn, dict) else getattr(_rn, "tree", None)
                if _tn:
                    _recalled_tree_names.add(_tn)
            try:
                s.memory.touch_trees_after_turn(
                    cycle_id=cycle_id,
                    branch_name=branch_name,
                    summary=(content or "")[:100],
                    emotion=episode.affect.get("dominant_emotion", ""),
                    outcome_score=outcome.get("score", 0.5),
                    recalled_tree_names=_recalled_tree_names,
                    policy=_touch_policy,
                )
            except Exception as _touch_err:
                print(
                    f"[Persona:{self.agent_id}] forest touch error: {_touch_err}",
                    flush=True,
                )

        _cited = llm_context_result.get("cited_memories", [])
        if _cited and hasattr(s.memory, "bump_cited_nodes"):
            _mids = []
            for _cm in _cited:
                if isinstance(_cm, str):
                    _mids.append(_cm)
                elif isinstance(_cm, int):
                    rm = episode.recalled_memories
                    if 0 <= _cm < len(rm):
                        _mid = (
                            rm[_cm].get("moment_id")
                            if isinstance(rm[_cm], dict)
                            else getattr(rm[_cm], "moment_id", None)
                        )
                        if _mid:
                            _mids.append(_mid)
            if _mids:
                try:
                    s.memory.bump_cited_nodes(_mids)
                except Exception as _bump_err:
                    print(
                        f"[Persona:{self.agent_id}] bump cited error: {_bump_err}",
                        flush=True,
                    )

        # -- 16c. LLM-centric environment feedback -------------------------
        if _llm_centric:
            _env_upd = llm_context_result.get("env_updates", {})
            if _env_upd:
                try:
                    self._apply_llm_env_updates(_env_upd, s, cycle_id, branch_name)
                except Exception as _eu_err:
                    print(
                        f"[Persona:{self.agent_id}] env update error: {_eu_err}",
                        flush=True,
                    )

        # -- 16d. Deliberative: apply chosen action's impacts -----------
        _d_apply = (
            str(
                os.environ.get("HAROMA_DELIBERATIVE_APPLY", "1") or "1",
            )
            .strip()
            .lower()
        )
        if (
            _d_apply in ("1", "true", "yes", "on")
            and deliberative_flag
            and llm_context_result.get("chosen_action")
        ):
            try:
                self._apply_deliberative_consequences(
                    llm_context_result["chosen_action"],
                    s,
                    cycle_id,
                    outcome_score=float(outcome.get("score", 0.5) or 0.5),
                )
            except Exception as _dac_err:
                print(
                    f"[Persona:{self.agent_id}] deliberative apply error: {_dac_err}",
                    flush=True,
                )

        s.temporal.record(episode.to_summary())

        # -- Self-model compare + training ------------------------------
        _ws_for_self = self.workspace.get_contents()
        _attn_win = (
            _ws_for_self[0].source
            if _ws_for_self and hasattr(_ws_for_self[0], "source")
            else (
                _ws_for_self[0].get("source", "perception")
                if _ws_for_self and isinstance(_ws_for_self[0], dict)
                else "perception"
            )
        )
        actual_self_state = {
            "valence": emotion_summary.get("valence", 0.0),
            "arousal": emotion_summary.get("arousal", 0.0),
            "curiosity": curiosity_result.get("curiosity_score", 0.0),
            "strategy": action.get("strategy", "reflect"),
            "attention_winner": _attn_win,
        }
        self_surprise: Dict[str, Any] = {}
        if self_prediction:
            _sync_sm = s.neural_sync() if _trueself_traced_fast else s.persona_neural_section()
            with _sync_sm:
                self_surprise = s.self_model.compare(self_prediction, actual_self_state)
                episode.bind_self_surprise(self_surprise)
                if current_embedding is not None:
                    try:
                        _emb = (
                            current_embedding.copy()
                            if isinstance(current_embedding, np.ndarray)
                            else current_embedding
                        )
                        s._self_model_last_train_ctx = SelfModelTrainBatch(
                            embedding=_emb,
                            prev_state=dict(self._prev_cycle_state or {}),
                            actual_state=dict(actual_self_state),
                        )
                    except Exception:
                        pass
        self._prev_cycle_state = actual_self_state
        self._prev_self_surprise = self_surprise

        # -- Integrative brain-like state + outcome-grounded beliefs (post-surprise)
        try:
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
            maybe_log_brain_state(self.agent_id, _bs)
            apply_outcome_grounded_belief_updates(
                memory=s.memory,
                outcome=outcome,
                reasoning_result=reasoning_result,
                llm_context=llm_context_result if isinstance(llm_context_result, dict) else {},
                cycle_id=cycle_id,
                branch_name=branch_name,
                agent_id=self.agent_id,
                plasticity_index=float(_bs.get("plasticity_index", 0.5) or 0.5),
            )
        except Exception as _brain_err:
            print(
                f"[Persona:{self.agent_id}] brain_state / outcome_belief error: {_brain_err}",
                flush=True,
            )

        # -- Update narrative buffer -------------------------------------
        _narr_emotion = episode.affect.get("dominant_emotion", "neutral")
        _narr_salience = episode.compute_salience()
        if _narr_salience > 0.3 or _narr_emotion != "neutral":
            self._narrative_buffer.append(
                f"[{_narr_emotion}] {(content or text or 'silence')[:60]}"
            )
            if len(self._narrative_buffer) > self._max_narrative_length:
                self._narrative_buffer = self._narrative_buffer[-self._max_narrative_length :]

        # -- Update per-persona state ----------------------------------
        self._prev_episode_payload = episode.to_payload()
        self._prev_episode_payload["_timestamp"] = time.time()
        self._prev_outcome_score = outcome.get("score", 0.0)
        self._prev_modulation = modulation

        if is_conv:
            self.conversation.record_response(_vis, cycle_id)

        # -- Record outcome signals for BackgroundAgent training -------
        # Several brief neural reads (split so no single hold stacks every module).
        _outcome_score_fb = float(outcome.get("score", 0.0))

        def _record_plock():
            return s.neural_sync() if _trueself_traced_fast else s.persona_neural_section()

        with _record_plock():
            s.appraisal.record_outcome(
                appraisal_result.get("_raw_features") or [],
                actual_emotion=episode.affect["dominant_emotion"],
                actual_valence=emotion_summary.get("valence", 0.0),
                actual_arousal=emotion_summary.get("arousal", 0.0),
                outcome_score=outcome.get("score", 0.0),
            )
            s.modulation.record_outcome(
                valence=emotion_summary.get("valence", 0.0),
                arousal=emotion_summary.get("arousal", 0.0),
                modulation_used=modulation,
                outcome_score=outcome.get("score", 0.0),
                z_t=_z_t,
            )

        with _record_plock():
            if _is_real_module(getattr(s, "process_gate", None)):
                try:
                    s.process_gate.record_outcome(
                        _gate_decisions,
                        _gate_features,
                        _outcome_score_fb,
                        z_t=_z_t,
                    )
                except Exception:
                    pass

            if _is_real_module(getattr(s, "attention", None)):
                try:
                    _src_sal: Dict[str, float] = {}
                    for c in self.workspace.get_contents():
                        _src = c.source if hasattr(c, "source") else c.get("source", "?")
                        _sal = c.salience if hasattr(c, "salience") else c.get("salience", 0.5)
                        _src_sal[_src] = float(_sal)
                    _ws_winners = [
                        c.source if hasattr(c, "source") else c.get("source", "?")
                        for c in self.workspace.get_contents()
                    ]
                    s.attention.record_outcome(
                        _src_sal,
                        attention_ctx,
                        _outcome_score_fb,
                        _ws_winners,
                        z_t=_z_t,
                    )
                except Exception:
                    pass

        with _record_plock():
            _bb_mod = getattr(s, "backbone", None)
            if _is_real_module(_bb_mod) and _bb_snapshot is not None:
                try:
                    _obb = getattr(_bb_mod, "_outcome_buffer", None) or []
                    if self._prev_backbone_snapshot is not None and _obb:
                        _obb[-1]["next_snapshot"] = _bb_snapshot
                    _ce_out = None
                    if current_embedding is not None:
                        _ce_out = (
                            np.asarray(
                                current_embedding,
                                dtype=np.float32,
                            )
                            .flatten()
                            .tolist()
                        )
                    _bb_mod.record_outcome(
                        _bb_snapshot,
                        _outcome_score_fb,
                        next_snapshot=None,
                        content_embedding=_ce_out,
                    )
                    self._prev_backbone_snapshot = _bb_snapshot
                except Exception:
                    pass

        with _record_plock():
            if _is_real_module(getattr(s, "counterfactual", None)) and _cf_features:
                try:
                    _cf_val = outcome.get("breakdown", {}).get("counterfactual_value")
                    if _cf_val is None:
                        _cf_val = float(counterfactual_result.get("counterfactual_depth", 0)) * 0.3
                    s.counterfactual.record_outcome(_cf_features, float(_cf_val))
                except Exception:
                    pass

            if _is_real_module(getattr(s, "metacognition", None)):
                try:
                    s.metacognition.learn_from_outcome(_outcome_score_fb)
                except Exception:
                    pass

        # -- Record LLM context reasoning outcome for reward model ------
        if llm_context_result.get("source") == "llm_context_reasoning" and _is_real_module(
            getattr(s, "llm_backend", None)
        ):
            _lc_answer = llm_context_result.get("answer") or ""
            _lc_input = text or content or ""
            if _lc_answer and _lc_input:
                _lc_reward = float(outcome.get("score", 0.5) or 0.5)
                if action.get("strategy") == "llm_context":
                    _lc_reward = min(1.0, _lc_reward + 0.1)
                try:
                    from mind.alignment_training import (
                        compute_blended_alignment_reward,
                        log_alignment_event,
                    )

                    _blended, _diag = compute_blended_alignment_reward(
                        _lc_reward,
                        llm_context_result,
                        episode=episode,
                    )
                    _meta = {
                        "alignment_training": True,
                        "outcome_score_raw": _lc_reward,
                        "alignment": _diag,
                    }
                    try:
                        from mind.environment_prompt_budgets import (
                            PERSONA_ALIGNMENT_ENV_SUMMARY_MAX_CHARS,
                        )

                        _ae = episode.environment_for_training_metadata()
                        if _ae:
                            _meta["agent_environment"] = _ae
                            from mind.environment_context import (
                                environment_summary_for_prompt,
                            )

                            _meta["environment_summary"] = environment_summary_for_prompt(
                                _ae,
                                max_chars=PERSONA_ALIGNMENT_ENV_SUMMARY_MAX_CHARS,
                            )
                    except Exception:
                        pass
                    s.llm_backend.record_outcome(
                        _lc_input,
                        _lc_answer,
                        _blended,
                        alignment_metadata=_meta,
                    )
                except Exception as _align_err:
                    print(
                        f"[Persona:{self.agent_id}] alignment record error: {_align_err}",
                        flush=True,
                    )
                    s.llm_backend.record_outcome(_lc_input, _lc_answer, _lc_reward)
                else:
                    try:
                        log_alignment_event(
                            {
                                "cycle": cycle_id,
                                "persona": self.agent_id,
                                "prompt_len": len(_lc_input),
                                "response_len": len(_lc_answer),
                                "reward_blended": _blended,
                                "diagnostics": _diag,
                            }
                        )
                    except Exception as _log_err:
                        print(
                            f"[Persona:{self.agent_id}] alignment log error: {_log_err}",
                            flush=True,
                        )

        # -- 15c. Self-grown language (composer phrase + generative) -----
        _co = getattr(s, "composer", None)
        if (
            role == "conversant"
            and is_conv
            and _is_real_module(_co)
            and getattr(_co, "available", False)
        ):
            composer_ctx = None
            _sync_co = s.neural_sync() if _trueself_traced_fast else s.persona_neural_section()
            with _sync_co:
                try:
                    composer_ctx = _co._build_context(
                        current_embedding,
                        float(emotion_summary.get("valence", 0.0)),
                        float(emotion_summary.get("arousal", 0.0)),
                        str(action.get("strategy", "reflect")),
                        (interlocutor_snapshot or {}).get("style", "unknown"),
                        1.0,
                        knowledge_triples=_kg_triples if _kg_triples else None,
                        z_t=_z_t,
                    )
                except Exception:
                    composer_ctx = None
                if composer_ctx:
                    _comp = action.get("composition")
                    if isinstance(_comp, dict) and not _comp.get("generative"):
                        try:
                            _co.record_outcome(
                                _comp,
                                float(outcome.get("score", 0.0)),
                                composer_ctx,
                            )
                        except Exception:
                            pass
                    _otext = (action.get("text") or "").strip()
                    if float(outcome.get("score", 0.5)) >= 0.55 and _otext and len(_otext) > 12:
                        try:
                            _co.record_text_outcome(
                                composer_ctx,
                                _otext,
                                float(outcome.get("score", 0.5)),
                            )
                        except Exception:
                            pass

        try:
            episode.bind_structured_actions(
                propose_structured_actions(
                    episode=episode,
                    action=action,
                    outcome=outcome,
                    agent_environment=getattr(episode, "agent_environment", None) or {},
                )
            )
        except Exception:
            episode.bind_structured_actions([])

        # -- Build response payload and store it -------------------------
        _user_first_encounter = False
        if _session_uid:
            _iom = interlocutor_snapshot or {}
            # Avoid ``interactions`` defaulting to 0 when the model never ran
            # (unknown speaker, gate off, or no update this turn).
            if _iom.get("known"):
                try:
                    _ix = int(_iom.get("interactions", 0) or 0)
                    _user_first_encounter = _ix <= 1
                except (TypeError, ValueError):
                    _user_first_encounter = True
        response_payload = self._build_response_payload(
            action,
            outcome,
            episode,
            emotion_summary,
            knowledge_summary,
            reasoning_result,
            interlocutor_snapshot,
            drive_state,
            meta_assessment,
            cycle_id=cycle_id,
            cycle_depth=_cycle_depth,
            debug_recall=debug_recall_flag,
            communication_debug=communication_debug_flag,
            utterance_style_used=_utter_style,
            role=role,
            in_conversation=is_conv,
            user_text=_ut_reply,
            user_session_id=_session_uid,
            first_encounter=_user_first_encounter,
        )
        try:
            _cm = getattr(s, "cognitive_metrics", None)
            if _cm is not None:
                _lc = getattr(episode, "llm_context", None) or {}
                if isinstance(_lc, dict):
                    _cm.observe_llm_wait_ms(_lc.get("latency_ms"))
                _cm.observe_persona_cycle_ms((time.time() - t0) * 1000.0)
        except Exception:
            pass
        log_input_pipeline(
            "persona.before_response_attach",
            trace_id=_pipe_tid,
            detail=f"agent={self.agent_id} strategy={action.get('strategy', '?')}",
        )
        trace_attach_to_payload(_trace_slot, response_payload)
        self._last_response_payload = response_payload

        slot = msg.response_slot
        if slot:
            slot["result"] = response_payload
            slot["event"].set()
            log_input_pipeline(
                "persona.http_slot_set",
                trace_id=_pipe_tid,
                detail=f"agent={self.agent_id}",
            )

        # -- Complete input goal (FIFO / Fuel) -------------------------
        self._complete_input_goal_from_message(msg)

        # -- Mark processed and check relay ----------------------------
        msg.mark_processed_by(self.agent_id)
        self._maybe_relay(msg, action, episode)

        elapsed = time.time() - t0
        _prev = (content or "")[:50]
        print(
            f"[CognitiveLoop] persona={self.agent_id} "
            f"local_iter={self._cycle_count} global_cycle={cycle_id} "
            f"depth={_cycle_depth} elapsed={elapsed:.2f}s "
            f"strategy={action.get('strategy', '?')} | {_prev}",
            flush=True,
        )
        self._yield_after_cycle_if_input_pipeline_busy()

    # -- idle cycle (no messages) --------------------------------------

    def _run_idle_cycle(self):
        """Periodic rumination and inner dialogue when no messages pending.

        Every ``_IDLE_RUMINATE_EVERY`` idle ticks the persona reviews its
        working memory, picks a salient topic, and sends it to a sibling
        persona as an inner-dialogue message.  This keeps the system
        evolving even when there is no external input.
        """
        if not self._running:
            return
        try:
            if input_pipeline_busy(self.shared, getattr(self, "_boot_agent", None)):
                return
        except Exception as _hc:
            print(f"[Persona:{self.agent_id}] input_pipeline_busy check error: {_hc}", flush=True)
            return
        if self._tick_count < 2:
            return
        if self._tick_count % self._IDLE_RUMINATE_EVERY != 0:
            return

        now = time.time()
        _ext = self.personality.get("extraversion") or 0.5
        _adj_cooldown = self._INNER_DIALOGUE_COOLDOWN * (1.5 - _ext)
        if (now - self._last_inner_dialogue_time) < _adj_cooldown:
            return

        topic = self._pick_rumination_topic()
        if not topic:
            return

        if not self._boot_agent:
            return
        siblings = [
            p
            for p in self._boot_agent.persona_agents
            if p.agent_id != self.agent_id and p.is_alive()
        ]
        if not siblings:
            return

        target = random.choice(siblings)
        self._last_inner_dialogue_time = now

        dialogue_msg = Message(
            sender_id=self.agent_id,
            channel="inter_persona",
            content={
                "text": topic,
                "tags": ["inner_dialogue", "rumination"],
                "nlu_base": {"intent": "reflect", "sentiment": {"polarity": 0.0}},
            },
            message_type="inner_dialogue",
            metadata={
                "prior_processors": [self.agent_id],
                "dialogue_depth": 0,
                "dialogue_origin": self.agent_id,
                "relay_context": {
                    "from_persona": self.agent_id,
                    "from_persona_name": self.persona_name,
                    "action_text": topic[:200],
                },
            },
        )
        self.bus.send_direct(target.agent_id, dialogue_msg)
        print(
            f"[Persona:{self.agent_id}] INNER DIALOGUE -> {target.agent_id}: {topic[:60]}",
            flush=True,
        )

    def _pick_rumination_topic(self) -> Optional[str]:
        """Select the most cognitively valuable topic for inner reflection.

        Priority hierarchy (highest first):
          1. Curiosity questions -- unanswered questions from the last episode
          2. Autonomy intentions -- latest ``thought_tree`` / ``autonomy`` node
          3. Dream insights    -- dream-tagged WM items (themes, narratives)
          4. Active goals      -- top unsatisfied goal descriptions
          5. Notable encounters -- high-salience recent experiences
          6. Outcome reflections -- notably good or bad recent outcomes
          7. General WM items  -- salient working memory content
          8. Conversation history -- last conversation turn
        """

        def _clean(raw: str) -> str:
            s = raw.strip()
            while s.lower().startswith("reflecting on:"):
                s = s[len("reflecting on:") :].strip()
            while s.lower().startswith("thinking more about:"):
                s = s[len("thinking more about:") :].strip()
            while s.lower().startswith("i've been wondering:"):
                s = s[len("i've been wondering:") :].strip()
            if not s or s.startswith("{") or s.startswith("[") or len(s) < 5:
                return ""
            return s

        def _is_sensor_goal_noise(desc: str) -> bool:
            """Skip FIFO goals whose description is only ``sensors:...`` tags (inner dialogue spam)."""
            d = (desc or "").strip()
            if "sensors:" not in d.lower():
                return False
            remainder = re.sub(r"\s*sensors:[\w,]+\.?\s*", " ", d, flags=re.I)
            remainder = re.sub(r"\s+", " ", remainder).strip(" .|")
            return len(remainder) < 3

        s = self.shared

        # 1. Unsatisfied curiosity questions from last episode
        if self._prev_episode_payload:
            questions = self._prev_episode_payload.get("curiosity", {}).get("questions", [])
            if questions:
                q = _clean(str(questions[0]))
                if q:
                    return q

        # 2. Recent autonomy / self-initiated intentions
        try:
            auto_nodes = s.memory.get_nodes("thought_tree", "autonomy")
            if auto_nodes:
                raw = getattr(auto_nodes[-1], "content", "") or ""
                cleaned = _clean(str(raw))
                if cleaned:
                    return cleaned
        except Exception as _e:
            print(f"[Persona:{self.agent_id}] rumination autonomy read error: {_e}", flush=True)

        # 3. Dream insights from working memory
        wm_items = self.working_memory.get_context(limit=10)
        dream_topics = []
        for item in wm_items:
            tags = getattr(item, "tags", []) or []
            if "dream" in tags:
                raw = item.content if hasattr(item, "content") else ""
                if not isinstance(raw, str):
                    raw = str(raw) if raw is not None else ""
                cleaned = _clean(raw)
                if cleaned:
                    dream_topics.append(cleaned)
        if dream_topics:
            return random.choice(dream_topics)

        # 4. Active goals (workfront = not blocked by incomplete children)
        if s.goal is not None:
            try:
                _pf = getattr(s.goal, "prioritize_workfront", None)
                top_ids = (
                    _pf()[:3]
                    if callable(_pf)
                    else (s.goal.prioritize()[:3] if hasattr(s.goal, "prioritize") else [])
                )
                eng = getattr(s.goal, "engine", None)
                goals_map = getattr(eng, "goals", None) if eng is not None else None
                for gid in top_ids:
                    g = None
                    if isinstance(goals_map, dict):
                        g = goals_map.get(gid)
                    desc = ""
                    if isinstance(g, dict):
                        desc = str(g.get("description", ""))
                    elif g is not None:
                        desc = str(getattr(g, "description", "") or "")
                    cleaned = _clean(desc) if desc else ""
                    if cleaned and not _is_sensor_goal_noise(desc):
                        return f"How can I make progress on: {cleaned[:60]}"
            except Exception as _e:
                print(f"[Persona:{self.agent_id}] rumination goal read error: {_e}", flush=True)

        # 5. Notable encounters from memory
        if _is_real_module(s.memory_core):
            try:
                enc_nodes = list(
                    s.memory_core.forest.get_nodes(
                        "encounter_tree", f"{AGENT_PREFIX}{self.agent_id}"
                    )
                )
                if not enc_nodes:
                    enc_nodes = list(s.memory_core.forest.get_nodes("encounter_tree", "common"))
                salient = [n for n in enc_nodes[-10:] if getattr(n, "confidence", 0.5) > 0.6]
                if salient:
                    pick = random.choice(salient[-5:])
                    content = (pick.content or "")[:80]
                    cleaned = _clean(content)
                    if cleaned:
                        return f"Thinking about when: {cleaned[:60]}"
            except Exception as _e:
                print(
                    f"[Persona:{self.agent_id}] rumination encounter read error: {_e}", flush=True
                )

        # 6. Outcome reflections
        if self._prev_outcome_score < 0.3:
            last_action = self._prev_episode_payload.get("action", {})
            strategy = last_action.get("strategy", "") if isinstance(last_action, dict) else ""
            if strategy:
                return f"Why did my {strategy} approach not work well?"
        elif self._prev_outcome_score > 0.8:
            return "What made that interaction go so well?"

        # 7. General working-memory items
        candidates = []
        for item in wm_items:
            tags = getattr(item, "tags", []) or []
            if "dream" in tags:
                continue
            raw = item.content if hasattr(item, "content") else ""
            if not isinstance(raw, str):
                raw = str(raw) if raw is not None else ""
            cleaned = _clean(raw)
            if cleaned:
                candidates.append(cleaned)
        if candidates:
            return random.choice(candidates)

        # 8. Recent conversation thread
        if self.conversation.history:
            last = self.conversation.history[-1]
            raw = getattr(last, "content", "")
            if not isinstance(raw, str):
                raw = str(raw) if raw is not None else ""
            cleaned = _clean(raw)
            if cleaned:
                return cleaned

        return None

    # -- relay logic ---------------------------------------------------

    def _maybe_relay(self, msg: Message, action: Dict, episode: EpisodeContext):
        """Check if this message should be relayed to a sibling persona.

        Relay triggers:
          1. Outcome score is low (this persona may not be best fit)
          2. Unsatisfied curiosity questions exist
          3. Message was explicitly tagged for multi-perspective processing
        """
        if not self._boot_agent:
            return
        already_processed = set(msg.prior_processors)
        siblings = [
            p
            for p in self._boot_agent.persona_agents
            if p.agent_id != self.agent_id and p.agent_id not in already_processed and p.is_alive()
        ]
        if not siblings:
            return

        response_text = action.get("text") or ""

        # Score each sibling for relay relevance
        relay_targets: List[tuple] = []
        for sibling in siblings:
            relay_score = 0.0

            # Trigger 1: low outcome score (this persona struggled)
            if self._prev_outcome_score < 0.3 and not self.is_default:
                relay_score += 0.3

            # Trigger 2: unsatisfied curiosity questions
            if episode.curiosity.get("questions"):
                relay_score += 0.2

            if relay_score > 0.2:
                relay_targets.append((relay_score, sibling))

        if not relay_targets:
            return

        relay_targets.sort(key=lambda x: x[0], reverse=True)
        best_score, best_sibling = relay_targets[0]

        print(
            f"[Persona:{self.agent_id}] RELAY -> {best_sibling.agent_id} "
            f"(score={best_score:.2f}, reason="
            f"{'curiosity/outcome' if best_score >= 0.4 else 'fallback'})",
            flush=True,
        )

        relay_msg = Message(
            sender_id=self.agent_id,
            channel="inter_persona",
            content=msg.content,
            message_type="relay",
            metadata={
                **{k: v for k, v in msg.metadata.items() if k != "_response_slot"},
                "prior_processors": list(msg.prior_processors),
                "relay_context": {
                    "from_persona": self.agent_id,
                    "from_persona_name": self.persona_name,
                    "action_text": response_text[:200],
                    "outcome_score": self._prev_outcome_score,
                    "relay_score": best_score,
                },
            },
        )
        self.bus.send_direct(best_sibling.agent_id, relay_msg)

    # -- helper: personality nudge -------------------------------------

    def _nudge_personality(
        self,
        outcome_score: float,
        strategy: str,
        intensity: float,
        valence: float,
    ) -> None:
        p = self.personality
        if outcome_score < 0.3:
            p.nudge("neuroticism", 0.005)
            p.nudge("resilience", -0.003)
        elif outcome_score > 0.8:
            p.nudge("resilience", 0.005)
            p.nudge("neuroticism", -0.003)

        if intensity > 0.7:
            sign = -1.0 if valence > 0.0 else 1.0
            p.nudge("neuroticism", sign * 0.003)

        strat_nudges = {
            "empathize": ("agreeableness", 0.004),
            "inquire": ("openness", 0.004),
            "advance_goal": ("conscientiousness", 0.004),
        }
        if strategy in strat_nudges and outcome_score > 0.5:
            trait, delta = strat_nudges[strategy]
            p.nudge(trait, delta)

        p.apply_drift_decay()

    # -- LLM-centric environment feedback ------------------------------

    _VALID_EMOTIONS = frozenset(
        {
            "joy",
            "wonder",
            "curiosity",
            "fear",
            "sadness",
            "anger",
            "resolve",
            "peace",
            "surprise",
            "neutral",
        }
    )
    _VALID_TRAITS = frozenset(
        {
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
            "resilience",
            "assertiveness",
        }
    )
    _MAX_ENV_GOALS = 20
    _MAX_ENV_KG = 10
    _MAX_ENV_WM = 3
    _MAX_ENV_MEM = 5

    def _apply_llm_env_updates(
        self,
        env_updates: Dict[str, Any],
        s: Any,
        cycle_id: int,
        branch_name: str,
    ) -> None:
        """Apply structured environment updates from the LLM back into subsystems."""
        if not env_updates or not isinstance(env_updates, dict):
            return

        # -- Emotion -------------------------------------------------------
        emo = env_updates.get("emotion")
        if isinstance(emo, dict):
            label = str(emo.get("label", "")).strip().lower()
            if label and label in self._VALID_EMOTIONS:
                try:
                    intensity = max(0.0, min(1.0, float(emo.get("intensity", 0.5))))
                    self.emotion.update_emotion(label, intensity)
                except Exception as _e:
                    print(f"[Persona:{self.agent_id}] env emotion error: {_e}", flush=True)

        # -- Goals ---------------------------------------------------------
        goals = env_updates.get("goals")
        if isinstance(goals, list):
            _goal_mgr = getattr(s, "goal", None)
            if _is_real_module(_goal_mgr) and hasattr(_goal_mgr, "register_goal"):
                for g in goals[: self._MAX_ENV_GOALS]:
                    if not isinstance(g, dict):
                        continue
                    gid = str(g.get("goal_id", "")).strip()
                    desc = str(g.get("description", "")).strip()
                    if not gid or not desc:
                        continue
                    try:
                        pri = max(0.0, min(1.0, float(g.get("priority", 0.5))))
                        extra = {}
                        cg = g.get("child_goal_ids")
                        if isinstance(cg, list) and cg:
                            extra["child_goal_ids"] = [
                                str(x).strip() for x in cg if str(x).strip()
                            ]
                        ai = g.get("action_items")
                        if isinstance(ai, list) and ai:
                            extra["action_items"] = ai
                        pg = g.get("parent_goal_id")
                        if pg is not None and str(pg).strip():
                            extra["parent_goal_id"] = str(pg).strip()
                        _goal_mgr.register_goal(
                            gid, desc, pri, source="llm_env", **extra
                        )
                    except Exception as _e:
                        print(f"[Persona:{self.agent_id}] env goal error: {_e}", flush=True)

        # -- Personality nudges --------------------------------------------
        nudges = env_updates.get("personality_nudges")
        if isinstance(nudges, list):
            for n in nudges:
                if not isinstance(n, dict):
                    continue
                trait = str(n.get("trait", "")).strip().lower()
                if trait not in self._VALID_TRAITS:
                    continue
                try:
                    delta = float(n.get("delta", 0))
                    delta = max(-0.01, min(0.01, delta))
                    self.personality.nudge(trait, delta)
                except Exception:
                    pass

        # -- Knowledge graph triples ---------------------------------------
        kg_triples = env_updates.get("kg_triples")
        if isinstance(kg_triples, list) and kg_triples:
            _kg = getattr(s, "knowledge", None)
            if _is_real_module(_kg) and hasattr(_kg, "integrate_world_state"):
                valid = []
                for t in kg_triples[: self._MAX_ENV_KG]:
                    if not isinstance(t, dict):
                        continue
                    subj = str(t.get("subject", "")).strip()
                    pred = str(t.get("predicate", "")).strip()
                    obj_ = str(t.get("object", "")).strip()
                    if subj and pred and obj_:
                        conf = 0.8
                        try:
                            conf = max(0.1, min(1.0, float(t.get("confidence", 0.8))))
                        except (TypeError, ValueError):
                            pass
                        valid.append(
                            {
                                "subject": subj,
                                "predicate": pred,
                                "object": obj_,
                                "confidence": conf,
                                "source": "llm_env",
                            }
                        )
                if valid:
                    try:
                        _kg.integrate_world_state(valid, cycle_id=cycle_id)
                    except Exception as _e:
                        print(f"[Persona:{self.agent_id}] env KG error: {_e}", flush=True)

        # -- Working memory notes ------------------------------------------
        wm_notes = env_updates.get("wm_notes")
        if isinstance(wm_notes, list):
            for note in wm_notes[: self._MAX_ENV_WM]:
                if not isinstance(note, dict):
                    continue
                c = str(note.get("content", "")).strip()
                if not c:
                    continue
                try:
                    sal = max(0.1, min(1.0, float(note.get("salience", 0.6))))
                    self.working_memory.add(
                        content=c[:200],
                        source="llm_env",
                        salience=sal,
                        item_type="thought",
                        cycle=cycle_id,
                    )
                except Exception:
                    pass

        # -- Memory notes (forest) -----------------------------------------
        mem_notes = env_updates.get("memory_notes")
        if isinstance(mem_notes, list) and hasattr(s.memory, "add_node"):
            from core.Memory import MemoryNode as _MN

            for mn in mem_notes[: self._MAX_ENV_MEM]:
                if not isinstance(mn, dict):
                    continue
                tree = str(mn.get("tree", "thought_tree")).strip()
                c = str(mn.get("content", "")).strip()
                if not c:
                    continue
                tags = mn.get("tags", [])
                if not isinstance(tags, list):
                    tags = []
                tags = [str(t)[:40] for t in tags[:8]]
                tags.append("llm_env")
                try:
                    node = _MN(
                        content=c[:300],
                        tags=tags,
                        confidence=0.7,
                    )
                    s.memory.add_node(tree, branch_name, node)
                except Exception:
                    pass

    def _apply_deliberative_consequences(
        self,
        chosen: Dict[str, Any],
        s: Any,
        cycle_id: int,
        outcome_score: float = 0.5,
    ) -> None:
        """Apply the cognitive loop's chosen candidate's value / goal / belief deltas.

        Advisory LLM-estimated impacts — scaled by env (see ``deliberative_choice``).
        Belief ``confidence_delta`` is scaled by realized ``outcome_score`` when
        ``HAROMA_DELIB_OUTCOME_BETA`` is non-zero (see ``outcome_belief_update``).
        """
        from mind.deliberative_choice import goal_delta_scale, value_step_scale

        v_scale = value_step_scale()
        g_scale = goal_delta_scale()
        _belief_om = deliberative_belief_outcome_multiplier(outcome_score)

        _val_mgr = getattr(s, "value", None)
        if _is_real_module(_val_mgr) and hasattr(_val_mgr, "reinforce_value"):
            _eng = getattr(_val_mgr, "engine", None)
            _cur: Dict[str, float] = {}
            if _eng is not None:
                try:
                    _cur = dict(getattr(_eng, "values", {}) or {})
                except Exception:
                    _cur = {}
            for k, delta in (chosen.get("value_impact") or {}).items():
                try:
                    d = float(delta) * v_scale
                except (TypeError, ValueError):
                    continue
                sk = str(k)
                base = float(_cur.get(sk, 0.5))
                new_w = max(0.0, min(1.0, base + d))
                try:
                    _val_mgr.reinforce_value(sk, new_w)
                    _cur[sk] = new_w
                except Exception:
                    pass

        _goal_mgr = getattr(s, "goal", None)
        if _is_real_module(_goal_mgr) and hasattr(_goal_mgr, "bump_goal_priority"):
            _store = getattr(_goal_mgr.engine, "goals", {})
            for gid, delta in (chosen.get("goal_impact") or {}).items():
                try:
                    d = float(delta) * g_scale
                except (TypeError, ValueError):
                    continue
                if str(gid) not in _store:
                    continue
                try:
                    _goal_mgr.bump_goal_priority(str(gid), d)
                except Exception:
                    pass

        for row in (chosen.get("belief_impact") or [])[:5]:
            if not isinstance(row, dict):
                continue
            prop = str(row.get("proposition") or "").strip()
            if not prop:
                continue
            try:
                cd = float(row.get("confidence_delta", 0.0)) * _belief_om
            except (TypeError, ValueError):
                cd = 0.0
            sal = max(0.15, min(1.0, 0.45 + min(0.55, abs(cd) * 0.45)))
            try:
                self.working_memory.add(
                    prop[:400],
                    source="deliberative_belief",
                    salience=sal,
                    item_type="belief",
                    tags=["belief", "deliberative"],
                    cycle=cycle_id,
                )
            except Exception:
                pass

    # -- helper: build gate features -----------------------------------

    def _persona_encoder(self):
        """Semantic :class:`engine.NeuralEncoder` for this persona (may differ from ``shared.encoder``)."""
        s = self.shared
        if hasattr(s, "encoder_for"):
            try:
                return s.encoder_for(self.agent_id)
            except Exception:
                pass
        return s.encoder

    @staticmethod
    def _safe_embed_dim(encoder) -> int:
        if encoder is None or is_cognitive_null(encoder):
            return 384
        dim = getattr(encoder, "_embed_dim", None)
        return dim if isinstance(dim, int) else 384

    @staticmethod
    def _safe_goal_count(goal_mgr) -> int:
        try:
            return len(goal_mgr.engine.goals)
        except Exception:
            return 0

    @staticmethod
    def _safe_curiosity(curiosity) -> float:
        try:
            res = curiosity.summarize()
            return res.get("curiosity_score", 0.0) if isinstance(res, dict) else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _safe_drive_level(drives) -> float:
        try:
            return max((d.level for d in drives.drives if hasattr(d, "level")), default=0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _safe_drive_levels(drives) -> list:
        try:
            return [d.level for d in drives.drives if hasattr(d, "level")][:5]
        except Exception:
            return []

    @staticmethod
    def _safe_metacog_prediction(s) -> float:
        try:
            return s.metacognition._history[-1].get("predicted_outcome", 0.5)
        except (IndexError, AttributeError):
            return 0.5

    def _build_gate_features(
        self,
        has_external,
        _prev_emo,
        symbolic_input,
        current_embedding,
        nlu_result,
    ) -> List[float]:
        s = self.shared
        embed_dim = self._safe_embed_dim(self._persona_encoder())
        return [
            math.log1p(s.cycle_count),
            1.0 if has_external else 0.0,
            _prev_emo.get("intensity", 0.0),
            _prev_emo.get("arousal", 0.0),
            self._safe_goal_count(s.goal) / 10.0,
            self._safe_curiosity(s.curiosity),
            self._safe_drive_level(s.drives),
            self.working_memory.occupancy(),
            s.knowledge.stats().get("entity_count", 0) / 100.0,
            self._prev_outcome_score,
            self._prev_episode_payload.get("reasoning", {}).get("reasoning_depth", 0) / 10.0,
            self._prev_episode_payload.get("counterfactual", {}).get("counterfactual_depth", 0)
            / 3.0,
            self._prev_episode_payload.get("imagination", {}).get("quality", 0.0),
            self._prev_self_surprise.get("overall_surprise", 0.0),
            1.0 if self.conversation.detect_topic_shift(symbolic_input.get("tags", [])) else 0.0,
            1.0 if self.conversation.is_in_conversation(s.cycle_count) else 0.0,
            min(
                10.0,
                time.time() - self._prev_episode_payload.get("_timestamp", time.time()),
            )
            / 10.0,
            self._safe_metacog_prediction(s),
            math.sqrt(
                sum(
                    x * x
                    for x in (list(current_embedding) if current_embedding is not None else [0.0])
                )
            )
            / max(embed_dim, 1),
            self._prev_episode_payload.get("_steps_run_ratio", 1.0),
            self.personality.get("openness") or 0.5,
            self.personality.get("conscientiousness") or 0.5,
        ]

    def _build_backbone_snapshot(
        self,
        current_embedding,
        _prev_emo,
        nlu_result,
        has_external,
    ) -> list:
        s = self.shared
        return build_snapshot(
            content_embedding=list(current_embedding) if current_embedding is not None else None,
            embed_dim=self._safe_embed_dim(self._persona_encoder()),
            valence=_prev_emo.get("valence", 0.0),
            arousal=_prev_emo.get("arousal", 0.0),
            intensity=_prev_emo.get("intensity", 0.0),
            curiosity_score=self._safe_curiosity(s.curiosity),
            prediction_error=self._prev_episode_payload.get("curiosity", {}).get(
                "prediction_error", 0.0
            ),
            dominant_drive_level=self._safe_drive_level(s.drives),
            wm_load=self.working_memory.occupancy(),
            outcome_prev=self._prev_outcome_score,
            has_external=1.0 if has_external else 0.0,
            cycle_count=s.cycle_count,
            n_goals=self._safe_goal_count(s.goal),
            kg_entity_count=s.knowledge.stats().get("entity_count", 0),
            self_surprise=self._prev_self_surprise.get("overall_surprise", 0.0),
            emotion_streak=0,
            drift_score=self._prev_episode_payload.get("drift_score", 0.0),
            strategy=self._prev_episode_payload.get("action", {}).get("strategy", "reflect")
            if isinstance(self._prev_episode_payload.get("action"), dict)
            else "reflect",
            intent=nlu_result.get("intent", "statement") if nlu_result else "statement",
            drive_levels=self._safe_drive_levels(s.drives),
            reasoning_depth=self._prev_episode_payload.get("reasoning", {}).get(
                "reasoning_depth", 0
            ),
            cf_depth=self._prev_episode_payload.get("counterfactual", {}).get(
                "counterfactual_depth", 0
            ),
            imagination_quality=self._prev_episode_payload.get("imagination", {}).get(
                "quality", 0.0
            ),
            metacog_prediction=self._safe_metacog_prediction(s),
            steps_run_ratio=self._prev_episode_payload.get("_steps_run_ratio", 1.0),
            plan_active=bool(self._current_plan),
            env_tick=0,
        )

    def _collect_recent_ids(self, limit: int = 10) -> List[str]:
        ids: List[str] = []
        for tree in self.shared.memory.trees.values():
            for branch in tree.branches.values():
                for node in branch.nodes:
                    ids.append(node.moment_id)
        return ids[-limit:]

    def _get_narrative_context(self) -> str:
        if not self._narrative_buffer:
            return f"I am {self.persona_name}. My story begins."
        return " ".join(self._narrative_buffer[-5:])

    def _chat_visible_response(
        self,
        action: Dict[str, Any],
        *,
        user_text: str = "",
        identity: Optional[Dict[str, Any]] = None,
        llm_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """User-facing JSON ``response`` for /chat.

        Uses deliberation/LLM-produced action text first. When the model yields
        no usable text, simple name/identity questions fall back to configured
        identity (``essence_name`` / ``persona_name``) instead of ``I don't know``.
        """
        return resolve_chat_visible_text(
            action,
            llm_context,
            user_text=user_text or "",
            identity=identity if isinstance(identity, dict) else None,
            persona_display_name=str(getattr(self, "persona_name", "") or ""),
        )

    # -- response payload builder --------------------------------------

    def _build_response_payload(
        self,
        action,
        outcome,
        episode,
        emotion_summary,
        knowledge_summary,
        reasoning_result,
        interlocutor_snapshot,
        drive_state,
        meta_assessment,
        *,
        cycle_id: int = 0,
        cycle_depth: str = "normal",
        debug_recall: bool = False,
        communication_debug: bool = False,
        utterance_style_used: Optional[str] = None,
        role: str = "observer",
        in_conversation: bool = False,
        user_text: str = "",
        user_session_id: Optional[str] = None,
        first_encounter: bool = False,
    ) -> Dict[str, Any]:
        cid = cycle_id or self.shared.cycle_count
        _ep_id = getattr(episode, "identity", None) or {}
        out: Dict[str, Any] = {
            "response": self._chat_visible_response(
                action,
                user_text=user_text,
                identity=_ep_id,
                llm_context=getattr(episode, "llm_context", None) or {},
            ),
            "_chat_resolve_user_text": user_text or "",
            "_chat_resolve_identity": {
                "essence_name": str(_ep_id.get("essence_name") or ""),
                "vessel": str(_ep_id.get("vessel") or ""),
            },
            "cycle": cid,
            "cycle_depth": cycle_depth,
            "affect": episode.affect,
            "strategy": action.get("strategy", ""),
            "drives": drive_state,
            "meta": meta_assessment,
            "knowledge": knowledge_summary,
            "conversation": {
                "in_conversation": self.conversation.is_in_conversation(cid),
                "topic": self.conversation.get_topic(),
                "turn_count": self.conversation.turn_count(),
            },
            "memory_nodes": self.shared.memory.count_nodes(),
            "goal_count": self._safe_goal_count(self.shared.goal),
            "nlu": episode.nlu,
            "interlocutor": interlocutor_snapshot,
            "reasoning": {
                "depth": reasoning_result.get("reasoning_depth", 0),
            },
            "llm_context": {
                "source": episode.llm_context.get("source", ""),
                "confidence": episode.llm_context.get("confidence", 0.0),
                "has_answer": bool(episode.llm_context.get("answer")),
                "answer": episode.llm_context.get("answer"),
                "latency_ms": episode.llm_context.get("latency_ms", 0.0),
                "prompt_info": episode.llm_context.get("prompt_info"),
                **optional_llm_structured_fields(episode.llm_context),
            }
            if episode.llm_context
            else {},
            "temporal": episode.temporal_position,
            "narrative": self._get_narrative_context(),
            "persona": self.agent_id,
            "persona_name": self.persona_name,
            "personality": self.personality.summarize(),
        }
        _ae = getattr(episode, "agent_environment", None) or {}
        if isinstance(_ae, dict) and _ae:
            out["agent_environment"] = {
                "domain": _ae.get("domain"),
                "fingerprint": _ae.get("fingerprint"),
                "schema_version": _ae.get("schema_version"),
                "entity_count": len(_ae.get("entities") or {}),
                "metrics_count": len(_ae.get("metrics") or {}),
                "alert_count": len(_ae.get("alerts") or []),
            }
        else:
            out["agent_environment"] = {}
        out["structured_actions"] = list(getattr(episode, "structured_actions", []) or [])
        if getattr(episode, "multi_goal_action_groups", None):
            out["goal_action_groups"] = episode.multi_goal_action_groups
        lc = getattr(episode, "llm_context", None) or {}
        if isinstance(lc, dict):
            eu = lc.get("env_updates")
            if isinstance(eu, dict):
                gl = eu.get("goals")
                if isinstance(gl, list) and gl:
                    g0 = gl[0]
                    if isinstance(g0, dict) and g0.get("goal_id") and g0.get("description"):
                        try:
                            _p = float(g0.get("priority", 0.5))
                            _p = max(0.0, min(1.0, _p))
                        except (TypeError, ValueError):
                            _p = 0.5
                        out["board_goal_proposal"] = {
                            "goal_id": str(g0["goal_id"])[:120],
                            "description": str(g0["description"])[:500],
                            "priority": _p,
                            "stance": "propose",
                        }
        if communication_debug:
            comp = action.get("composition")
            reply_surface = "unknown"
            if action.get("strategy") == "llm_context":
                reply_surface = "llm_context"
            elif isinstance(comp, dict) and comp.get("generative"):
                reply_surface = "composer_generative"
            elif isinstance(comp, dict):
                reply_surface = "composer_lexicon"
            elif utterance_style_used == "conversational":
                reply_surface = "conversational_join"
            elif comp == "template" or comp is None:
                reply_surface = "template_assembly"
            out["growth"] = {
                "persona_id": self.agent_id,
                "persona_name": self.persona_name,
                "role": role,
                "in_conversation": bool(in_conversation),
                "utterance_style": utterance_style_used or "",
                "outcome_score": round(float(outcome.get("score", 0.0)), 4),
                "strategy": action.get("strategy", ""),
                "reply_surface": reply_surface,
                "action_confidence": round(float(action.get("confidence", 0.0)), 4),
                "law_bound": bool(action.get("law_bound")),
                "learning_channel": (
                    "user_turn" if role == "conversant" and in_conversation else "internal"
                ),
            }
        if debug_recall:
            rm = episode.recalled_memories or []
            out["recall_debug"] = {
                "recall_count": len(rm),
                "snippets": [str(getattr(m, "content", ""))[:160] for m in rm[-8:]],
                "memory_influence": getattr(episode, "memory_influence", 0.0),
            }
        if user_session_id:
            out["user_id"] = user_session_id
            out["first_encounter"] = bool(first_encounter)
        return out

    # -- introspection -------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        base = super().stats()
        base["persona_name"] = self.persona_name
        base["affinity"] = self.affinity
        base["is_default"] = self.is_default
        base["persona_cycles"] = self._cycle_count
        base["emotion"] = self.emotion.summarize()
        base["personality"] = self.personality.summarize()
        base["working_memory"] = self.working_memory.stats()
        base["conversation"] = self.conversation.stats()
        return base
