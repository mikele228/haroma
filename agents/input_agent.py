"""
InputAgent -- collects all peripheral inputs and routes to TrueSelf.

Absorbs SensorBuffer + SensorPoller responsibilities. Performs lightweight,
objective preprocessing (entity extraction, intent classification, neural
embedding) then forwards everything to TrueSelf -- the sole consumer.

v6 change: InputAgent no longer broadcasts to personas or manages claims.
All routing decisions are made by TrueSelf.

Tick loop:
  1. Drain text queue + sensor queue
  2. Run lightweight NLU (objective: entities, relations, intent)
  3. Compute neural embedding once
  4. Forward to TrueSelf via send_direct

User chat uses the **chat** input sensor (see ``sensor_data.chat`` on the
forwarded message), same list-of-readings shape as other channels. Each reading
dict is annotated with ``text_translation`` (see :mod:`mind.sensor_text_translation`).
``sensor_text_translation_digest`` concatenates non-chat modalities for quick
prompt context. Each dispatch adds ``senses_numpy``: canonical per-modality
``float32`` vectors (empty when unavailable) plus ``text_embedding`` — see
:mod:`mind.sense_numpy_bundle`. **User** messages use a priority queue so they
are not stuck behind bulk/internal traffic. Env ``HAROMA_INPUT_TICK_INTERVAL_SEC``
overrides ``soul/agents.json`` input.tick_interval (BootAgent).
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from agents.base import BaseAgent
from utils.coerce_bool import env_flag, json_bool
from agents.chat_latency import (
    dummy_reply_auto_latency_trace,
    trace_attach_to_payload,
    trace_init,
    trace_log_requested,
    trace_requested,
    trace_span,
)
from agents.message_bus import Message, MessageBus
from core.cognitive_null import is_cognitive_null
from mind.user_identity import sanitize_user_id
from mind.cognitive_observability import new_trace_id
from mind.chat_pipeline_log import log_input_pipeline, trace_id_from_slot
from mind.sense_numpy_bundle import build_senses_numpy_bundle
from mind.sensor_text_translation import (
    enrich_sensor_data,
    sensor_text_translation_digest,
)

if TYPE_CHECKING:
    from agents.shared_resources import SharedResources
    from agents.boot_agent import BootAgent


def normalize_chat_inline_sensor_data(raw: Any) -> Optional[Dict[str, Any]]:
    """Normalize ``sensor_data`` / ``sensors`` from POST /chat into the internal
    channel → list-of-readings shape used with :meth:`push_sensor`.

    Each value may be a single object or a list; values are capped for safety.
    """
    if not isinstance(raw, dict) or not raw:
        return None
    out: Dict[str, Any] = {}
    for i, (k, v) in enumerate(raw.items()):
        if i >= 64:
            break
        key = str(k).strip()[:120]
        if not key:
            continue
        if isinstance(v, list):
            out[key] = v[:48]
        elif v is not None:
            out[key] = [v]
    return out or None


class InputAgent(BaseAgent):
    """Peripheral input collector -- preprocesses and forwards to TrueSelf."""

    AGENT_TYPE = "input"

    _TRUESELF_ID = "trueself"

    def __init__(
        self,
        shared: SharedResources,
        bus: MessageBus,
        tick_interval: float = 0.5,
    ):
        super().__init__(
            agent_id="input",
            shared=shared,
            bus=bus,
            tick_interval=tick_interval,
        )

        # Thread-safe input buffers (producers: HTTP handlers, sensor poller)
        self._lock = threading.Lock()
        self._text_queue: deque = deque(maxlen=256)
        # Normal-depth /chat from user — drained before ``_text_queue`` (lower tail latency).
        self._text_queue_priority: deque = deque(maxlen=128)
        self._sensor_queue: deque = deque(maxlen=128)
        self._response_log: deque = deque(maxlen=200)

        # FIFO input-goal tracking
        self._input_goal_counter = 0
        self._fifo_enabled = str(os.environ.get("HAROMA_GOAL_FIFO_INPUT", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._unify_sensor = str(
            os.environ.get("HAROMA_GOAL_UNIFY_SENSOR", "0")
        ).strip().lower() in ("1", "true", "yes", "on")

        # Reference to boot agent (set after boot)
        self._boot_agent: Optional[BootAgent] = None

    def set_boot_agent(self, boot_agent: BootAgent):
        self._boot_agent = boot_agent

    # -- producers (called by HTTP handlers / sensor poller) -----------

    def push_text(
        self,
        message: str,
        source: str = "user",
        depth: str = "normal",
        wake_callback=None,
        debug_recall: bool = False,
        trace_latency: bool = False,
        communication_debug: bool = False,
        deliberative: bool = False,
        user_id: Optional[str] = None,
        display_name: Optional[str] = None,
        experiment_id: Optional[str] = None,
        inline_sensor_data: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Push a text stimulus. Returns a slot dict with an Event.

        *inline_sensor_data* optional map ``channel -> reading | [readings]`` merged
        into this turn's ``sensor_data`` (same contract as POST ``/sensor``), e.g.
        lidar/gps alongside ``message`` when POSTing bundled JSON to ``/chat``.

        *depth* is accepted for API compatibility and normalized to ``normal``.
        *debug_recall* adds ``recall_debug`` to the HTTP result payload.
        *trace_latency* (or env HAROMA_CHAT_TRACE) adds ``latency_trace`` timings
        to the JSON reply.
        *communication_debug* (or env HAROMA_COMMUNICATION_DEBUG) adds a ``growth``
        object to the JSON reply (outcome, strategy, reply surface, persona id).
        *deliberative* enables full state JSON + candidate actions with value impact.
        """
        slot = {"event": threading.Event(), "result": None}
        try:
            cm = getattr(self.shared, "cognitive_metrics", None)
            if cm is not None:
                cm.on_chat_turn_started()
                with self._lock:
                    cm.observe_input_queue_depth(
                        len(self._text_queue_priority),
                        len(self._text_queue),
                    )
        except Exception:
            pass
        _tid = new_trace_id()
        slot["cognitive_trace_id"] = _tid
        _trace_flag = json_bool(trace_latency, False)
        if _trace_flag or trace_requested() or dummy_reply_auto_latency_trace():
            trace_init(
                slot,
                log_to_console=_trace_flag or trace_log_requested(),
            )
        d = str(depth or "normal").lower()
        if d != "normal":
            d = "normal"
        item = {
            "content": message,
            "source": source,
            "channel": "chat",
            "tags": ["conversation"],
            "depth": d,
            "debug_recall": json_bool(debug_recall, False),
            "communication_debug": json_bool(communication_debug, False),
            "deliberative": json_bool(deliberative, False),
            "trace_latency": json_bool(slot.get("_trace_enabled"), False),
            "_slot": slot,
        }
        _uid_s = sanitize_user_id(user_id) if user_id is not None else None
        if _uid_s:
            item["user_id"] = _uid_s
        if display_name is not None and str(display_name).strip():
            item["display_name"] = str(display_name).strip()[:160]
        if experiment_id is not None and str(experiment_id).strip():
            item["experiment_id"] = str(experiment_id).strip()[:200]
        item["cognitive_trace_id"] = _tid
        if inline_sensor_data:
            item["_inline_sensor_data"] = inline_sensor_data
        _pq = d == "normal" and str(source or "").lower() == "user"
        _target = self._text_queue_priority if _pq else self._text_queue
        with self._lock:
            if len(_target) == _target.maxlen:
                evicted = _target[0]
                evicted_slot = evicted.get("_slot")
                if evicted_slot and not evicted_slot["event"].is_set():
                    ev_err = {
                        "response": "[queue overflow — message dropped]",
                        "cycle": self.shared.cycle_count,
                        "affect": {},
                        "strategy": "error",
                    }
                    trace_attach_to_payload(evicted_slot, ev_err)
                    evicted_slot["result"] = ev_err
                    evicted_slot["event"].set()
            _target.append(item)
        if wake_callback:
            wake_callback()
        else:
            self.wake()
        log_input_pipeline(
            "input.push_text_queued",
            trace_id=_tid,
            detail=f"queue={'priority' if _pq else 'normal'}",
        )
        return slot

    def push_sensor(self, channel: str, data: Dict):
        """Push a raw sensor reading."""
        with self._lock:
            self._sensor_queue.append(
                {
                    "channel": channel,
                    "data": data,
                    "timestamp": time.time(),
                }
            )

    # -- drain (called internally each tick) ---------------------------

    def _drain_all(self):
        """Atomically drain everything."""
        with self._lock:
            texts = list(self._text_queue_priority) + list(self._text_queue)
            sensors = list(self._sensor_queue)
            self._text_queue_priority.clear()
            self._text_queue.clear()
            self._sensor_queue.clear()
        return texts, sensors

    # -- response log --------------------------------------------------

    def log_response(self, result: Dict):
        with self._lock:
            self._response_log.append({"result": result, "timestamp": time.time()})

    def recent_responses(self, n: int = 20) -> List:
        with self._lock:
            return list(self._response_log)[-n:]

    # -- FIFO goal helpers -----------------------------------------------

    def _next_input_goal_id(self) -> str:
        self._input_goal_counter += 1
        return f"ig_{self.shared.cycle_count}_{self._input_goal_counter}"

    def _register_goal_for_dispatch(
        self,
        goal_id: str,
        text: str,
        sensor_channels: Optional[List[str]] = None,
    ) -> None:
        """Register a FIFO input goal on the shared GoalManager."""
        desc_parts: List[str] = []
        if text:
            desc_parts.append(text[:120])
        if sensor_channels:
            # Avoid goal descriptions that are only ``sensors:vision`` — those pollute
            # inner-dialogue / rumination topic picks (see PersonaAgent._pick_rumination_topic).
            # Pure user /chat adds only the ``chat`` sensor mirror — do not append
            # ``sensors:chat`` (redundant with the line above; confuses prompts when dummy is off).
            _ch = [str(c).strip() for c in sensor_channels if str(c).strip()]
            _only_chat = len(_ch) == 1 and _ch[0].lower() == "chat"
            if text:
                if not _only_chat:
                    desc_parts.append(f"sensors:{','.join(_ch)}")
            else:
                desc_parts.append(f"Multimodal tick ({','.join(_ch)})")
        description = " | ".join(desc_parts) or "input_cycle"
        try:
            self.shared.goal.register_input_goal(
                goal_id=goal_id,
                description=description,
                source="input",
                meta={"sensor_channels": sensor_channels or []},
            )
        except Exception as exc:
            print(f"[InputAgent] register_input_goal error: {exc}", flush=True)

    # -- tick ----------------------------------------------------------

    def _dispatch_text_item(
        self,
        item: Dict[str, Any],
        merged_sensor_data: Optional[Dict[str, Any]] = None,
        sensor_channels: Optional[List[str]] = None,
    ) -> None:
        """NLU + encoder + forward one text item to TrueSelf."""
        slot = item.pop("_slot", None)
        _ptid = trace_id_from_slot(slot if isinstance(slot, dict) else None)
        log_input_pipeline("input.dispatch_start", trace_id=_ptid)
        trace_span(slot, "input_queue_wait")
        content = item.get("content", "")
        source = item.get("source", "user")
        tags = item.get("tags", [])
        depth = str(item.get("depth", "normal") or "normal").lower()
        if depth != "normal":
            depth = "normal"

        nlu_base = self._lightweight_nlu(content)
        trace_span(slot, "input_nlu")
        log_input_pipeline("input.after_nlu", trace_id=_ptid)

        embedding = None
        _encoder_real = self.shared.encoder is not None and not is_cognitive_null(
            self.shared.encoder
        )
        # Optional: skip SBERT encode for user /chat to cut ~1–3s CPU on large encoders (TrueSelf
        # can still answer from text; persona may skip re-encode via HAROMA_TRUESELF_USER_CHAT_*).
        _skip_enc = env_flag("HAROMA_INPUT_CHAT_SKIP_ENCODER", False) and str(source).lower() in (
            "user",
            "human",
        )
        if _encoder_real and content and not _skip_enc:
            try:
                log_input_pipeline("input.before_encoder_neural_sync", trace_id=_ptid)
                with self.shared.neural_sync("encoder"):
                    raw_emb = self.shared.encoder.encode(content)
                if raw_emb is not None:
                    expected = self.shared.encoder._embed_dim
                    if len(raw_emb) != expected:
                        if len(raw_emb) > expected:
                            embedding = np.array(raw_emb[:expected], dtype=np.float32)
                        else:
                            padded = np.zeros(expected, dtype=np.float32)
                            padded[: len(raw_emb)] = raw_emb
                            embedding = padded
                    else:
                        embedding = raw_emb
            except Exception as exc:
                print(f"[InputAgent] encoder error: {exc}", flush=True)

        trace_span(slot, "input_encoder")
        log_input_pipeline("input.after_encoder", trace_id=_ptid)

        # Merge multimodal sensors with the **chat** input sensor (same shape as ``push_sensor``).
        _sensor_payload: Dict[str, Any] = {}
        if merged_sensor_data:
            _sensor_payload.update(merged_sensor_data)
        _inline = item.pop("_inline_sensor_data", None)
        if isinstance(_inline, dict) and _inline:
            for ch, readings in _inline.items():
                chs = str(ch).strip()[:120]
                if not chs:
                    continue
                if readings is None:
                    continue
                if isinstance(readings, list):
                    _sensor_payload.setdefault(chs, []).extend(
                        x for x in readings[:48] if x is not None
                    )
                else:
                    _sensor_payload.setdefault(chs, []).append(readings)
        if content:
            _sensor_payload.setdefault("chat", []).append(
                {
                    "text": content,
                    "source": source,
                    "channel": "chat",
                    "ts": time.time(),
                }
            )

        enrich_sensor_data(_sensor_payload)
        _sensor_digest = sensor_text_translation_digest(_sensor_payload)

        # FIFO goal registration (chat counts as a sensor channel when text is present).
        _goal_id: Optional[str] = None
        if self._fifo_enabled:
            _goal_id = self._next_input_goal_id()
            _sc = list(sensor_channels or [])
            if content and "chat" not in _sc:
                _sc.append("chat")
            self._register_goal_for_dispatch(
                _goal_id,
                content,
                sensor_channels=_sc,
            )

        _uid = item.get("user_id")
        _dn = item.get("display_name")
        _ctid = item.get("cognitive_trace_id")
        _senses_numpy = build_senses_numpy_bundle(_sensor_payload, text_embedding=embedding)
        _content_payload: Dict[str, Any] = {
            "text": content,
            "source": source,
            "tags": tags,
            "nlu_base": nlu_base,
            "embedding": embedding,
            "sensor_data": _sensor_payload,
            "sensor_text_translation_digest": _sensor_digest,
            "senses_numpy": _senses_numpy,
            "cycle_depth": depth,
            "debug_recall": json_bool(item.get("debug_recall"), False),
            "communication_debug": json_bool(item.get("communication_debug"), False),
            "deliberative": json_bool(item.get("deliberative"), False),
            "input_goal_id": _goal_id,
        }
        if _uid is not None:
            _content_payload["user_id"] = _uid
        if _dn is not None:
            _content_payload["display_name"] = _dn
        _exp = item.get("experiment_id")
        if isinstance(_exp, str) and _exp.strip():
            _content_payload["experiment_id"] = _exp.strip()[:200]
        _meta: Dict[str, Any] = {
            "_response_slot": slot,
            "prior_processors": [],
        }
        if isinstance(_ctid, str) and _ctid:
            _meta["cognitive_trace_id"] = _ctid
        msg = Message(
            sender_id=self.agent_id,
            channel="input",
            content=_content_payload,
            message_type="input",
            metadata=_meta,
        )

        trace_span(slot, "input_to_trueself_handoff")
        log_input_pipeline("input.before_send_to_trueself", trace_id=_ptid)

        self.bus.send_direct(self._TRUESELF_ID, msg)
        log_input_pipeline("input.after_send_to_trueself", trace_id=_ptid)

        if slot is not None and self._boot_agent is not None:
            try:
                self._boot_agent.trueself_agent.wake()
            except Exception as _we:
                print(f"[InputAgent] wake() error: {_we}", flush=True)

        if self._tick_count < 3 or self._tick_count % 10 == 0:
            print(
                f"[InputAgent] -> TrueSelf: {content[:60]}",
                flush=True,
            )

    def _tick(self):
        text_items, sensor_items = self._drain_all()
        if not text_items and not sensor_items:
            return

        # Build merged sensor payload (reused in unified and standalone paths)
        merged_sensor: Dict[str, Any] = {}
        sensor_channels: List[str] = []
        if sensor_items:
            for s in sensor_items:
                ch = s.get("channel", "unknown")
                merged_sensor.setdefault(ch, []).append(s.get("data", {}))
                if ch not in sensor_channels:
                    sensor_channels.append(ch)

        # Unified path: merge sensors into the primary text message,
        # re-queue remaining text items for the next tick.
        if text_items and sensor_items and self._unify_sensor:
            primary = text_items[0]
            rest = text_items[1:]
            _slot = primary.get("_slot")
            try:
                self._dispatch_text_item(
                    primary,
                    merged_sensor_data=merged_sensor,
                    sensor_channels=sensor_channels,
                )
            except Exception as exc:
                print(f"[InputAgent] tick dispatch error: {exc}", flush=True)
                if _slot is not None and not _slot["event"].is_set():
                    err = {
                        "response": "[processing error]",
                        "cycle": self.shared.cycle_count,
                        "affect": {},
                        "strategy": "error",
                    }
                    trace_attach_to_payload(_slot, err)
                    _slot["result"] = err
                    _slot["event"].set()
            # Re-queue remaining texts (prepend so they dispatch first next tick)
            if rest:
                with self._lock:
                    for leftover in reversed(rest):
                        _lpq = str(leftover.get("depth", "normal") or "normal").lower() == "normal" and str(
                            leftover.get("source", "") or ""
                        ).lower() == "user"
                        (self._text_queue_priority if _lpq else self._text_queue).appendleft(leftover)
            return

        # Default path: dispatch each text independently
        for item in text_items:
            _slot = item.get("_slot")
            try:
                self._dispatch_text_item(item)
            except Exception as exc:
                print(f"[InputAgent] tick dispatch error: {exc}", flush=True)
                if _slot is not None and not _slot["event"].is_set():
                    err = {
                        "response": "[processing error]",
                        "cycle": self.shared.cycle_count,
                        "affect": {},
                        "strategy": "error",
                    }
                    trace_attach_to_payload(_slot, err)
                    _slot["result"] = err
                    _slot["event"].set()

        # Standalone sensor message (no text items, or unify not enabled)
        if sensor_items and not (text_items and self._unify_sensor):
            enrich_sensor_data(merged_sensor)
            _standalone_digest = sensor_text_translation_digest(merged_sensor)
            _sensor_goal_id: Optional[str] = None
            if self._fifo_enabled:
                _sensor_goal_id = self._next_input_goal_id()
                self._register_goal_for_dispatch(
                    _sensor_goal_id,
                    "",
                    sensor_channels=sensor_channels,
                )
            msg = Message(
                sender_id=self.agent_id,
                channel="input",
                content={
                    "text": "",
                    "source": "sensor",
                    "tags": ["sensor"],
                    "nlu_base": {},
                    "embedding": None,
                    "sensor_data": merged_sensor,
                    "sensor_text_translation_digest": _standalone_digest,
                    "input_goal_id": _sensor_goal_id,
                },
                message_type="input",
            )
            self.bus.send_direct(self._TRUESELF_ID, msg)

    # -- lightweight NLU (objective, persona-independent) ---------------

    @staticmethod
    def _neutralize_question_intent_labels(intent: Optional[str]) -> str:
        """Fold legacy question-type labels into a neutral intent (no ? / WH heuristics)."""
        i = (intent or "").strip().lower()
        if i in ("interrogative", "question", "inquiry"):
            return "utterance"
        return i or "utterance"

    @staticmethod
    def _minimal_nlu_heuristic(content: str) -> Dict[str, Any]:
        """Lightweight fields only; intent is neutral (no question/statement split)."""
        return {
            "intent": "utterance",
            "entities": [],
            "relations": [],
            "sentiment": {"polarity": 0.0, "subjectivity": 0.0},
        }

    def _lightweight_nlu(self, content: str) -> Dict[str, Any]:
        """Run objective NLU: entities, relations, intent classification.

        Emotional perception and discourse interpretation are left to
        the claiming PersonaAgent. Uses channel ``chat`` (user utterance sensor).
        """
        if not content:
            return {}
        if not self.shared.perception:
            return self._minimal_nlu_heuristic(content)
        try:
            perceived = self.shared.perception.perceive({"content": content}, channel="chat")
            nlu = perceived.get("nlu", {})
            return {
                "intent": InputAgent._neutralize_question_intent_labels(
                    nlu.get("intent", "statement"),
                ),
                "entities": nlu.get("entities", []),
                "relations": nlu.get("relations", []),
                "sentiment": nlu.get("sentiment", {}),
            }
        except Exception:
            return self._minimal_nlu_heuristic(content)

    # -- introspection -------------------------------------------------

    def peek_sensor_queue(self, limit: int = 64) -> List[Dict[str, Any]]:
        """Non-destructive snapshot of recent sensor readings (newest first)."""
        with self._lock:
            items = list(self._sensor_queue)
        items.reverse()
        return items[:limit]

    def buffer_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "text_pending": len(self._text_queue),
                "text_priority_pending": len(self._text_queue_priority),
                "sensor_pending": len(self._sensor_queue),
                "response_log_size": len(self._response_log),
            }

    def stats(self) -> Dict[str, Any]:
        base = super().stats()
        base["buffer"] = self.buffer_stats()
        return base
