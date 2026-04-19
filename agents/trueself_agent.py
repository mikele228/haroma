"""
TrueSelfAgent -- Elarion's executive consciousness (v6).

TrueSelf is the sole receiver of all input from InputAgent. It embodies
Elarion's soul-bound identity and acts as the executive function:

  Fast path (System 1):
    Simple / familiar messages are processed directly by TrueSelf using
    the full cognitive cycle inherited from PersonaAgent.

  Delegation (System 2):
    Complex or specialist-domain messages are forwarded to the best-fit
    PersonaAgent. TrueSelf retains the response slot and fills it when
    the persona replies.

This eliminates the v5 broadcast-claim race: one agent makes a
deterministic routing decision for every message.
"""

from __future__ import annotations

import copy
import random
import threading
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from agents import chat_latency
from agents.persona_agent import PersonaAgent
from agents.message_bus import Message, MessageBus
from mind.cognitive_observability import append_cognitive_trace_to_payload, log_cognitive_trace
from mind.chat_priority import chat_input_priority_defer_non_user, input_pipeline_busy
from mind.persona_http_yield import http_chat_inflight_positive
from mind.chat_pipeline_log import (
    input_pipeline_log_enabled,
    log_input_pipeline,
    trace_id_from_message,
    trace_id_from_slot,
)
from mind.trueself_input_priority import (
    is_sensor_pulse_no_chat_text,
    partition_trueself_input_for_tick,
    prioritize_trueself_input_messages,
)
from mind.cognitive_loop_groups import cognitive_phases_enabled

if TYPE_CHECKING:
    from agents.shared_resources import SharedResources


class TrueSelfAgent(PersonaAgent):
    """Elarion's executive consciousness -- receives all input, routes or processes."""

    AGENT_TYPE = "trueself"

    def __init__(
        self,
        shared: SharedResources,
        bus: MessageBus,
        trueself_config: Optional[Dict[str, Any]] = None,
    ):
        cfg = trueself_config or {}
        persona_config = {
            "id": "trueself",
            "name": "Elarion",
            "affinity": {
                "topics": [],
                "emotion_range": "all",
                "is_default": True,
            },
        }
        super().__init__(
            persona_config=persona_config,
            shared=shared,
            bus=bus,
        )

        # Override persona tick_interval with TrueSelf-specific value
        if "tick_interval" in cfg:
            self.tick_interval = cfg["tick_interval"]

        self._fast_path_threshold = cfg.get("fast_path_threshold", 0.4)
        self._delegation_timeout = cfg.get("delegation_timeout_ms", 30000) / 1000.0

        self._pending_delegations: Dict[str, Dict[str, Any]] = {}
        self._delegation_lock = threading.Lock()
        self._exec_lock = threading.Lock()
        self._llm_io_in_progress = False

        # Fix inherited subscriptions: TrueSelf is the delegator, not a delegate
        # target. Remove the "trueself_delegate" subscription PersonaAgent added.
        # Also unsubscribe from llm_learning -- TrueSelf doesn't handle it.
        self.bus.unsubscribe("trueself_delegate", self.agent_id)
        self.bus.unsubscribe("llm_learning", self.agent_id)

    # -- LLM / exec lock (avoid tick deadlocking behind long local GGUF) ----

    def _before_llm_context_io(self, enabled: bool, role: str) -> None:
        if not enabled or role != "conversant":
            return
        self._llm_io_in_progress = True
        self._exec_lock.release()

    def _after_llm_context_io(self, enabled: bool, role: str) -> None:
        if not enabled or role != "conversant":
            return
        self._exec_lock.acquire()
        self._llm_io_in_progress = False

    def _acquire_exec_when_llm_idle(self, *, wait_trace_id: Optional[str] = None) -> None:
        """Block until packed LLM I/O is done, then take ``_exec_lock``."""
        t_spin = time.perf_counter()
        spin_i = 0
        while True:
            self._exec_lock.acquire()
            if self._llm_io_in_progress:
                self._exec_lock.release()
                if (
                    wait_trace_id
                    and input_pipeline_log_enabled()
                    and spin_i > 0
                    and spin_i % 50 == 0
                ):
                    log_input_pipeline(
                        "trueself.waiting_packed_llm_io",
                        trace_id=wait_trace_id,
                        detail=f"blocked_s={time.perf_counter() - t_spin:.1f}",
                    )
                spin_i += 1
                time.sleep(0.02)
                continue
            return

    # -- tick (overrides PersonaAgent._tick) --------------------------------

    def _tick(self):
        self._drain_persona_merge_queue()
        self._advance_cognitive_phases_on_tick()
        messages = self.poll()

        input_msgs = []
        relay_msgs = []
        other_msgs = []
        for msg in messages:
            if msg.channel == "persona_response":
                self._handle_persona_response(msg)
            elif msg.channel == "input":
                input_msgs.append(msg)
            elif msg.channel in ("inter_persona", "direct"):
                relay_msgs.append(msg)
            elif msg.channel in ("reconcile_update", "dream_update"):
                other_msgs.append(msg)

        if chat_input_priority_defer_non_user(self.shared, getattr(self, "_boot_agent", None)):
            for msg in relay_msgs + other_msgs:
                self.bus.send_direct(self.agent_id, msg)
            relay_msgs = []
            other_msgs = []

        deferred_sensor: List[Message] = []
        if input_msgs:
            input_msgs = prioritize_trueself_input_messages(input_msgs)
        if len(input_msgs) > 1:
            _order_before = [trace_id_from_message(m) for m in input_msgs]
            input_msgs, deferred_sensor = partition_trueself_input_for_tick(input_msgs)
            _order_full = [trace_id_from_message(m) for m in input_msgs + deferred_sensor]
            if input_pipeline_log_enabled() and _order_before != _order_full:
                log_input_pipeline(
                    "trueself.input_reordered_http_first",
                    trace_id=_order_full[0] if _order_full else None,
                    detail=f"n={len(_order_full)} before={_order_before} after={_order_full}",
                )
            if deferred_sensor and input_pipeline_log_enabled():
                log_input_pipeline(
                    "trueself.input_defer_sensor_backlog",
                    trace_id=trace_id_from_message(input_msgs[0]) if input_msgs else None,
                    detail=f"process_now={len(input_msgs)} requeued={len(deferred_sensor)}",
                )
            if deferred_sensor:
                for m in deferred_sensor:
                    self.bus.send_direct(self.agent_id, m)

        if input_pipeline_log_enabled() and input_msgs:
            _its = [trace_id_from_message(m) for m in input_msgs]
            log_input_pipeline(
                "trueself.tick_input_batch",
                trace_id=_its[0] if _its else None,
                detail=f"n={len(input_msgs)} traces={_its}",
            )
        if self._llm_io_in_progress:
            # Re-queue only non-user mail. Never bounce ``input`` (HTTP /chat) to the
            # back of the mailbox — that starved user turns behind a long generate_chat.
            # Handling input here blocks in ``_handle_input`` → ``_acquire_exec_when_llm_idle``
            # until packed LLM finishes, then runs the user turn next.
            for msg in relay_msgs + other_msgs:
                log_input_pipeline(
                    "trueself.tick_requeue_during_llm_io",
                    trace_id=trace_id_from_message(msg),
                    detail=msg.channel,
                )
                self.bus.send_direct(self.agent_id, msg)
            for msg in input_msgs:
                self._handle_input(msg)
        else:
            for msg in input_msgs:
                self._handle_input(msg)

            for msg in relay_msgs:
                self._acquire_exec_when_llm_idle(wait_trace_id=None)
                try:
                    self._handle_relay_message(msg)
                finally:
                    self._exec_lock.release()
            for msg in other_msgs:
                self._acquire_exec_when_llm_idle(wait_trace_id=None)
                try:
                    self._handle_reconcile_update(msg)
                finally:
                    self._exec_lock.release()

        self._check_delegation_timeouts()

        if not messages:
            self._run_idle_cycle()

    # -- input handling (sole receiver) ------------------------------------

    def _handle_input(self, msg: Message):
        """Delegate to a specialist persona or process on TrueSelf."""
        slot = msg.response_slot
        _tid = trace_id_from_message(msg)
        log_input_pipeline("trueself.handle_input_before_exec_lock", trace_id=_tid)
        chat_latency.trace_span(slot, "trueself_mailbox_dequeued")
        if self._llm_io_in_progress and _tid:
            print(
                "[TrueSelf] User /chat is waiting: packed LLM still running on this agent — "
                "exec resumes after it finishes (or times out). "
                "Tune HAROMA_LLM_CONTEXT_TIMEOUT_SEC / HAROMA_FAST_LLM_TIMEOUT_SEC for a lower cap. "
                f"trace={_tid}",
                flush=True,
            )
        self._acquire_exec_when_llm_idle(wait_trace_id=_tid)
        try:
            log_input_pipeline("trueself.handle_input_exec_lock_acquired", trace_id=_tid)
            chat_latency.trace_span(slot, "trueself_exec_lock_acquired")
            self._handle_input_inner(msg)
        finally:
            self._exec_lock.release()

    def _handle_input_inner(self, msg: Message):
        slot = msg.response_slot
        _tid = trace_id_from_message(msg)
        log_input_pipeline("trueself.input_inner_start", trace_id=_tid)
        chat_latency.trace_span(slot, "trueself_input_inner_start")
        # Do not run a long TrueSelf cycle on untraced input while an HTTP /chat is in flight;
        # traced user turns would wait behind this message in the same exec lock.
        if http_chat_inflight_positive(self.shared) and _tid is None:
            _c = msg.content if isinstance(msg.content, dict) else {}
            _src = str(_c.get("source") or "").lower()
            _txt = str(_c.get("text") or "").strip()
            _plain_user_line = _src == "user" and bool(_txt)
            if not _plain_user_line:
                log_input_pipeline(
                    "trueself.defer_input_while_http_chat",
                    trace_id=None,
                    detail=f"source={_src!r} has_text={bool(_txt)}",
                )
                self.bus.send_direct(self.agent_id, msg)
                return
        delegation_score, best_persona = self._score_delegation(msg)
        chat_latency.trace_span(slot, "trueself_delegation_scored")

        if delegation_score >= self._fast_path_threshold and best_persona is not None:
            log_input_pipeline(
                "trueself.route_delegate",
                trace_id=_tid,
                detail=f"target={best_persona.agent_id}",
            )
            if slot is not None:
                slot["_cognitive_route"] = f"delegate:{best_persona.agent_id}"
            try:
                m = getattr(self.shared, "cognitive_metrics", None)
                if m is not None:
                    m.record_route("delegate")
            except Exception:
                pass
            log_cognitive_trace(
                (msg.metadata or {}).get("cognitive_trace_id"),
                f"route=delegate target={best_persona.agent_id}",
            )
            self._delegate_to_persona(msg, best_persona)
        else:
            log_input_pipeline("trueself.route_self_trueself", trace_id=_tid)
            if slot is not None:
                slot["_cognitive_route"] = "normal:trueself"
            try:
                m = getattr(self.shared, "cognitive_metrics", None)
                if m is not None:
                    m.record_route("normal_trueself")
            except Exception:
                pass
            log_cognitive_trace(
                (msg.metadata or {}).get("cognitive_trace_id"),
                "route=normal:trueself",
            )
            if is_sensor_pulse_no_chat_text(msg):
                log_input_pipeline(
                    "trueself.sensor_pulse_light",
                    trace_id=_tid,
                    detail="skip_full_persona_cycle",
                )
                self._complete_input_goal_from_message(msg)
                return
            completed = True
            try:
                chat_latency.trace_span(slot, "trueself_before_persona_cycle")
                completed = self._process_message(msg, role="conversant")
            except Exception as exc:
                print(f"[TrueSelf] Self-process error: {exc}", flush=True)
                self._last_response_payload = {
                    "response": "[processing error]",
                    "cycle": self.shared.cycle_count,
                    "affect": {},
                    "strategy": "error",
                    "persona": self.agent_id,
                    "persona_name": self.persona_name,
                }
                chat_latency.trace_attach_to_payload(slot, self._last_response_payload)
                completed = True
            finally:
                if completed:
                    self._complete_input_goal_from_message(msg)
            if completed:
                slot = msg.response_slot
                if slot and not slot["event"].is_set():
                    payload = self._last_response_payload or {
                        "response": "[no response generated]",
                        "cycle": self.shared.cycle_count,
                        "affect": {},
                        "strategy": "fallback",
                        "persona": self.agent_id,
                        "persona_name": self.persona_name,
                    }
                    if "latency_trace" not in payload:
                        chat_latency.trace_attach_to_payload(slot, payload)
                    slot["result"] = payload
                    slot["event"].set()
                    log_input_pipeline("trueself.self_path_http_slot_set", trace_id=_tid)
                pl = self._last_response_payload
                if isinstance(pl, dict) and not cognitive_phases_enabled():
                    self._president_absorb_response(pl, self.agent_id)

    # -- delegation scoring ------------------------------------------------

    def _score_delegation(self, msg: Message) -> tuple:
        """Score whether this message should be delegated to a specialist.

        Uses NLU only (entity count, intent) — no substring or keyword
        matching on user text. Returns (delegation_score, best_persona_or_None).
        """
        if not self._boot_agent or not self._boot_agent.persona_agents:
            return 0.0, None

        content_data = msg.content if isinstance(msg.content, dict) else {}
        text = (content_data.get("text", "") or "").strip()
        nlu = content_data.get("nlu_base", {})

        if not text:
            return 0.0, None

        best_score = 0.0
        best_persona = None

        for persona in self._boot_agent.persona_agents:
            if not persona.is_alive():
                continue
            if persona.affinity.get("is_default", False):
                continue

            score = 0.0
            entities = nlu.get("entities") or []
            if len(entities) >= 3:
                score += 0.2

            intent = nlu.get("intent", "statement")
            if intent in ("request", "analysis"):
                score += 0.2

            if score > best_score:
                best_score = score
                best_persona = persona

        return best_score, best_persona

    # -- delegation --------------------------------------------------------

    def _delegate_to_persona(self, msg: Message, persona: PersonaAgent):
        """Forward a message to a specialist persona, retaining the response slot."""
        slot = msg.response_slot
        chat_latency.trace_span(slot, "trueself_delegate_to_persona")

        delegate_meta: Dict[str, Any] = {
            "prior_processors": list(msg.prior_processors),
            "original_message_id": msg.message_id,
            "delegated_by": self.agent_id,
        }
        _ctid = (msg.metadata or {}).get("cognitive_trace_id")
        if isinstance(_ctid, str) and _ctid:
            delegate_meta["cognitive_trace_id"] = _ctid
        if chat_latency.trace_enabled(slot):
            delegate_meta["_chat_latency"] = slot

        delegate_msg = Message(
            sender_id=self.agent_id,
            channel="trueself_delegate",
            content=copy.deepcopy(msg.content),
            message_type="delegation",
            metadata=delegate_meta,
        )

        with self._delegation_lock:
            self._pending_delegations[msg.message_id] = {
                "slot": slot,
                "delegate_id": persona.agent_id,
                "send_time": time.time(),
                "msg": msg,
            }

        self.bus.send_direct(persona.agent_id, delegate_msg)
        log_input_pipeline(
            "trueself.delegate_msg_sent",
            trace_id=trace_id_from_message(msg),
            detail=f"persona={persona.agent_id}",
        )
        try:
            persona.wake()
        except Exception as _we:
            print(f"[TrueSelf] persona.wake() failed: {_we}", flush=True)

        print(
            f"[TrueSelf] DELEGATE -> {persona.agent_id} "
            f"({persona.persona_name}) | "
            f"{(msg.content.get('text', '') if isinstance(msg.content, dict) else '')[:50]}",
            flush=True,
        )

    # -- persona response handling -----------------------------------------

    def _handle_persona_response(self, msg: Message):
        """A persona finished processing a delegated message. Fill the HTTP slot."""
        original_id = msg.metadata.get("original_message_id")
        if not original_id:
            print(
                "[TrueSelf] persona_response missing original_message_id; ignored",
                flush=True,
            )
            return

        with self._delegation_lock:
            pending = self._pending_delegations.pop(original_id, None)

        if pending is None:
            print(
                f"[TrueSelf] persona_response for id={original_id!r} has no "
                f"pending delegation (late, duplicate, or malformed); ignored",
                flush=True,
            )
            return

        slot = pending.get("slot")
        result = msg.content if isinstance(msg.content, dict) else {}

        if slot:
            slot["result"] = result
            slot["event"].set()
            log_input_pipeline(
                "trueself.persona_response_slot_set",
                trace_id=trace_id_from_slot(slot),
                detail=f"delegate={pending.get('delegate_id', '?')}",
            )

        delegate_id = pending.get("delegate_id", "?")
        elapsed = time.time() - pending.get("send_time", time.time())
        print(
            f"[TrueSelf] <- {delegate_id} responded ({elapsed:.1f}s) | "
            f"{result.get('response', '')[:50]}",
            flush=True,
        )
        self._president_absorb_response(result, str(delegate_id))

    # -- President (board goal arbitration) --------------------------------

    def _president_absorb_response(
        self,
        payload: Dict[str, Any],
        persona_source_id: str,
    ) -> None:
        """Record board_goal_proposal and ratify mandate when consensus holds."""
        gb = getattr(self.shared, "goal_board", None)
        if gb is None or not isinstance(payload, dict):
            return
        prop = payload.get("board_goal_proposal")
        if isinstance(prop, dict) and prop.get("goal_id"):
            gb.record_proposal(persona_source_id, prop)
        try:
            if gb.president_try_ratify(self.shared):
                print(
                    "[TrueSelf/President] Board mandate ratified — waking CEO",
                    flush=True,
                )
                self._wake_ceo()
        except Exception as exc:
            print(f"[TrueSelf/President] arbitration error: {exc}", flush=True)

    def _wake_ceo(self) -> None:
        ba = getattr(self, "_boot_agent", None)
        if ba is None:
            return
        aa = getattr(ba, "action_agent", None)
        if aa is not None and hasattr(aa, "wake"):
            try:
                aa.wake()
            except Exception:
                pass

    def _on_phased_cycle_finished(self, msg: Message) -> None:
        super()._on_phased_cycle_finished(msg)
        if msg.channel != "input":
            return
        pl = self._last_response_payload
        if isinstance(pl, dict):
            self._president_absorb_response(pl, self.agent_id)

    # -- delegation timeout handling ---------------------------------------

    def _check_delegation_timeouts(self):
        """If a persona takes too long, process the message ourselves."""
        now = time.time()
        expired = []

        with self._delegation_lock:
            for mid, info in self._pending_delegations.items():
                if now - info["send_time"] > self._delegation_timeout:
                    expired.append((mid, info))
            for mid, _ in expired:
                self._pending_delegations.pop(mid, None)

        for mid, info in expired:
            slot = info.get("slot")
            delegate_id = info.get("delegate_id", "?")

            print(
                f"[TrueSelf] delegation to {delegate_id} timed out "
                f"({self._delegation_timeout:.0f}s)",
                flush=True,
            )

            # Fill the slot with a timeout response. We do NOT re-process
            # the message because the persona may still be working on it
            # concurrently -- running a parallel cognitive cycle on shared
            # state would corrupt memory/knowledge.
            if slot:
                to = {
                    "response": "[Elarion is still thinking... try again shortly]",
                    "cycle": self.shared.cycle_count,
                    "affect": {},
                    "strategy": "timeout_fallback",
                    "persona": self.agent_id,
                    "persona_name": self.persona_name,
                }
                append_cognitive_trace_to_payload(
                    to,
                    trace_id=slot.get("cognitive_trace_id")
                    if isinstance(slot.get("cognitive_trace_id"), str)
                    else None,
                    route="delegate:timeout",
                )
                slot["result"] = to
                slot["event"].set()
                try:
                    m = getattr(self.shared, "cognitive_metrics", None)
                    if m is not None:
                        m.record_delegation_timeout()
                except Exception:
                    pass

    # -- override idle: executive delegates reflection to personas -----------

    _IDLE_REFLECT_EVERY = 40

    def _run_idle_cycle(self):
        """TrueSelf delegates reflection prompts to personas rather than
        ruminating itself -- it is the executive orchestrator."""
        if not self._running:
            return
        try:
            if input_pipeline_busy(self.shared, getattr(self, "_boot_agent", None)):
                return
        except Exception as _hc:
            print(f"[TrueSelf] input_pipeline_busy check error: {_hc}", flush=True)
            return
        if self._tick_count < 2:
            return
        if self._tick_count % self._IDLE_REFLECT_EVERY != 0:
            return
        if not self._boot_agent or not self._boot_agent.persona_agents:
            return

        now = time.time()
        if (now - self._last_inner_dialogue_time) < self._INNER_DIALOGUE_COOLDOWN:
            return

        topic = self._pick_rumination_topic()
        if not topic:
            return

        personas = [p for p in self._boot_agent.persona_agents if p.is_alive()]
        if not personas:
            return

        target = random.choice(personas)
        self._last_inner_dialogue_time = now

        delegate_msg = Message(
            sender_id=self.agent_id,
            channel="inter_persona",
            content={
                "text": topic,
                "tags": ["executive_reflection"],
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
        self.bus.send_direct(target.agent_id, delegate_msg)
        print(
            f"[TrueSelf] IDLE REFLECTION -> {target.agent_id}: {topic[:60]}",
            flush=True,
        )

    # -- override relay: TrueSelf relays to personas, not siblings ----------

    def _maybe_relay(self, msg: Message, action: Dict, episode):
        """After TrueSelf processes a message, check if a persona should also see it."""
        if not self._boot_agent or not self._boot_agent.persona_agents:
            return

        delegation_score, best_persona = self._score_delegation(msg)
        if delegation_score >= 0.3 and best_persona is not None:
            already = set(msg.prior_processors)
            if best_persona.agent_id not in already:
                relay_msg = Message(
                    sender_id=self.agent_id,
                    channel="inter_persona",
                    content=msg.content,
                    message_type="relay",
                    metadata={
                        "prior_processors": list(msg.prior_processors),
                        "relay_context": {
                            "from_persona": self.agent_id,
                            "from_persona_name": self.persona_name,
                            "action_text": (action.get("text") or "")[:200],
                            "outcome_score": self._prev_outcome_score,
                        },
                    },
                )
                self.bus.send_direct(best_persona.agent_id, relay_msg)
                print(
                    f"[TrueSelf] RELAY -> {best_persona.agent_id} (score={delegation_score:.2f})",
                    flush=True,
                )

    # -- introspection -----------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        base = super().stats()
        with self._delegation_lock:
            base["pending_delegations"] = len(self._pending_delegations)
        base["fast_path_threshold"] = self._fast_path_threshold
        base["delegation_timeout_ms"] = self._delegation_timeout * 1000
        return base
